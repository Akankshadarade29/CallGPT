from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from typing import Annotated, Sequence

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk

from .. import input_text, chunking, embeddings, retrieval, prompt_templates, llms, vectorstore_supabase


class RAGState(TypedDict, total=False):
    # Inputs / config
    input_path: str
    table_name: str
    query_name: str
    rebuild: bool

    embeddings_model: str

    llm_model: str
    temperature: float

    search_type: str
    k: int
    fetch_k: int
    lambda_mult: float

    template: str
    question: str

    # Conversational memory (accumulated across turns via checkpointer)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Artifacts
    docs: List[Document]
    chunks: List[Document]
    answer: str

# Nodes

def node_load(state: RAGState) -> Dict[str, Any]:
    docs = input_text.load_text_file(state["input_path"])
    return {"docs": docs}


def node_chunk(state: RAGState) -> Dict[str, Any]:
    chunks = chunking.chunk_documents(state["docs"])  # default params
    return {"chunks": chunks}


def node_embeddings(state: RAGState) -> Dict[str, Any]:
    # Deprecated: embeddings object is ephemeral and should not be checkpointed
    return {}


def node_vectorstore(state: RAGState) -> Dict[str, Any]:
    # Supabase-only: upsert when explicitly requested via rebuild
    emb = embeddings.get_embedding_model(state.get("embeddings_model"))
    if state.get("rebuild", False):
        table_name = state.get("table_name", "documents")
        query_name = state.get("query_name", "match_documents")
        vectorstore_supabase.build_supabase_from_documents(
            state["chunks"], emb, table_name=table_name, query_name=query_name
        )
    return {}


def node_retriever(state: RAGState) -> Dict[str, Any]:
    # Deprecated: retriever object is ephemeral and should not be checkpointed
    return {}


def node_llm(state: RAGState) -> Dict[str, Any]: 
    # Deprecated: LLM object is ephemeral and should not be checkpointed
    return {}


def node_answer(state: RAGState):
    prompt = prompt_templates.get_qa_prompt()
    # Ephemerally load vectorstore and create retriever (Supabase)
    emb = embeddings.get_embedding_model(state.get("embeddings_model"))
    vstore = vectorstore_supabase.load_supabase(
        state.get("table_name", "documents"),
        emb,
        query_name=state.get("query_name", "match_documents"),
    )
    if state.get("search_type", "mmr") == "mmr":
        retriever = retrieval.get_retriever(
            vstore,
            search_type="mmr",
            k=state.get("k", 4),
            fetch_k=state.get("fetch_k", 20),
            lambda_mult=state.get("lambda_mult", 0.5),
        )
    else:
        retriever = retrieval.get_retriever(vstore, search_type="similarity", k=state.get("k", 4))

    # Retrieve and prepare context
    docs = retrieval.retrieve(retriever, state["question"])
    context = "\n\n".join(d.page_content for d in docs)

    # Create LLM ephemerally
    temperature = state.get("temperature", 0.1)
    llm = llms.get_groq_llm(model="openai/gpt-oss-120b", temperature=temperature)

    history = list(state.get("messages", []))  # previous turns
    current = prompt.format_messages(context=context, question=state["question"])
    messages = [*history, *current]

    # Stream tokens and yield AI message chunks for smoother UI
    answer_accum = ""
    for chunk in llm.stream(messages):
        delta = getattr(chunk, "content", "")
        if not delta:
            continue
        answer_accum += delta
        # Emit incremental assistant chunks via the messages channel
        yield {"messages": [AIMessageChunk(content=delta)]}

    # Finalize: return full answer and persist the turn into memory
    yield {
        "answer": answer_accum,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer_accum),
        ],
    }


def build_rag_graph(checkpointer: Optional[Any] = None) -> Any:
    builder = StateGraph(RAGState)
    builder.add_node("load", node_load)
    builder.add_node("chunk", node_chunk)
    builder.add_node("vectorstore", node_vectorstore)
    builder.add_node("answer", node_answer)

    builder.add_edge(START, "load")
    builder.add_edge("load", "chunk")
    builder.add_edge("chunk", "vectorstore")
    builder.add_edge("vectorstore", "answer")
    builder.add_edge("answer", END)

    return builder.compile(checkpointer=checkpointer)
