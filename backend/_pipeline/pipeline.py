from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict
from typing import Annotated, Sequence

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from backend.input_text.load_text import load_text_file
from backend.chunking.chunk_text import chunk_documents
from backend.embeddings.generate_embeddings import get_embedding_model
from backend.vectorstore_faiss.build_store import build_faiss_from_documents, load_faiss
from backend.retrieval.retriever import get_retriever, retrieve
from backend.prompt_templates.templates import get_qa_prompt
from backend.llms.init_llms import get_groq_llm
from backend.qa_generation.qa import answer_question


class RAGState(TypedDict, total=False):
    # Inputs / config
    input_path: str
    index_dir: str
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


essential_files = {"index.faiss", "index.pkl"}


def _need_rebuild(index_dir: str) -> bool:
    return not os.path.isdir(index_dir) or any(
        not os.path.exists(os.path.join(index_dir, f)) for f in essential_files
    )


# Nodes

def node_load(state: RAGState) -> Dict[str, Any]:
    docs = load_text_file(state["input_path"])
    return {"docs": docs}


def node_chunk(state: RAGState) -> Dict[str, Any]:
    chunks = chunk_documents(state["docs"])  # default params
    return {"chunks": chunks}


def node_embeddings(state: RAGState) -> Dict[str, Any]:
    # Deprecated: embeddings object is ephemeral and should not be checkpointed
    return {}


def node_vectorstore(state: RAGState) -> Dict[str, Any]:
    index_dir = state.get("index_dir", "faiss_index")
    # Create embeddings model ephemerally
    emb = get_embedding_model(state.get("embeddings_model"))
    if state.get("rebuild", False) or _need_rebuild(index_dir):
        build_faiss_from_documents(state["chunks"], emb, index_dir=index_dir,)
    # Do not return vstore to state; it's not serializable. Loading will be repeated where needed.
    return {}


def node_retriever(state: RAGState) -> Dict[str, Any]:
    # Deprecated: retriever object is ephemeral and should not be checkpointed
    return {}


def node_llm(state: RAGState) -> Dict[str, Any]: 
    # Deprecated: LLM object is ephemeral and should not be checkpointed
    return {}


def node_answer(state: RAGState) -> Dict[str, Any]:
    prompt = get_qa_prompt()
    # Ephemerally load vectorstore and create retriever
    emb = get_embedding_model(state.get("embeddings_model"))
    vstore = load_faiss(state.get("index_dir", "faiss_index"), emb)
    if state.get("search_type", "mmr") == "mmr":
        retriever = get_retriever(
            vstore,
            search_type="mmr",
            k=state.get("k", 4),
            fetch_k=state.get("fetch_k", 20),
            lambda_mult=state.get("lambda_mult", 0.5),
        )
    else:
        retriever = get_retriever(vstore, search_type="similarity", k=state.get("k", 4))

    # Retrieve and answer
    docs = retrieve(retriever, state["question"])
    context = "\n\n".join(d.page_content for d in docs)

    # Create LLM ephemerally
    temperature = state.get("temperature", 0.1)
    llm = get_groq_llm(model="openai/gpt-oss-120b", temperature=temperature)

    history = list(state.get("messages", []))  # previous turns
    current = prompt.format_messages(context=context, question=state["question"])
    messages = [*history, *current]

    resp = llm.invoke(messages)
    ans = getattr(resp, "content", str(resp))

    return {
        "answer": ans,
        # Append the latest human/AI turn so memory persists per thread
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=ans),
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
