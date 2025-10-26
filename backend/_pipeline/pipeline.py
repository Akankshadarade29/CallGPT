from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from backend.input_text.load_text import load_text_file
from backend.chunking.chunk_text import chunk_documents
from backend.embeddings.generate_embeddings import get_embedding_model
from backend.vectorstore_faiss.build_store import build_faiss_from_documents, load_faiss
from backend.retrieval.retriever import get_retriever, retrieve
from backend.prompt_templates.templates import get_qa_prompt
from backend.llms.init_llms import get_groq_llm, get_openai_llm, get_oss_llm
from backend.qa_generation.qa import answer_question


class RAGState(TypedDict, total=False):
    # Inputs / config
    input_path: str
    index_dir: str
    rebuild: bool

    embeddings_provider: str
    embeddings_model: Optional[str]

    llm_provider: str
    llm_model: Optional[str]
    temperature: float

    search_type: str
    k: int
    fetch_k: int
    lambda_mult: float

    template: str
    question: str

    # Artifacts
    docs: List[Document]
    chunks: List[Document]
    embeddings: Embeddings
    vstore: FAISS
    retriever: Any
    llm: Any
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
    emb = get_embedding_model(state.get("embeddings_provider", "huggingface"), state.get("embeddings_model"))
    return {"embeddings": emb}


def node_vectorstore(state: RAGState) -> Dict[str, Any]:
    index_dir = state.get("index_dir", "faiss_index")
    if state.get("rebuild", False) or _need_rebuild(index_dir):
        build_faiss_from_documents(state["chunks"], state["embeddings"], index_dir=index_dir, use_cosine=True)
    vstore = load_faiss(index_dir, state["embeddings"])
    return {"vstore": vstore}


def node_retriever(state: RAGState) -> Dict[str, Any]:
    if state.get("search_type", "mmr") == "mmr":
        retriever = get_retriever(
            state["vstore"],
            search_type="mmr",
            k=state.get("k", 4),
            fetch_k=state.get("fetch_k", 20),
            lambda_mult=state.get("lambda_mult", 0.5),
        )
    else:
        retriever = get_retriever(state["vstore"], search_type="similarity", k=state.get("k", 4))
    return {"retriever": retriever}


def node_llm(state: RAGState) -> Dict[str, Any]:
    provider = state.get("llm_provider", "groq").lower()
    model = state.get("llm_model")
    temperature = state.get("temperature", 0.1)
    if provider == "groq":
        llm = get_groq_llm(model=model or "llama-3.1-8b-instant", temperature=temperature)
    elif provider == "openai":
        llm = get_openai_llm(model=model or "gpt-4o-mini", temperature=temperature)
    else:
        llm = get_oss_llm(model=model or "llama3.1", temperature=temperature)
    return {"llm": llm}


def node_answer(state: RAGState) -> Dict[str, Any]:
    prompt = get_qa_prompt(template=state.get("template", "default"))
    docs = retrieve(state["retriever"], state["question"])
    ans = answer_question(state["llm"], prompt, docs, state["question"])
    return {"answer": ans}


def build_rag_graph() -> Any:
    builder = StateGraph(RAGState)
    builder.add_node("load", node_load)
    builder.add_node("chunk", node_chunk)
    builder.add_node("embeddings", node_embeddings)
    builder.add_node("vectorstore", node_vectorstore)
    builder.add_node("retriever", node_retriever)
    builder.add_node("llm", node_llm)
    builder.add_node("answer", node_answer)

    builder.add_edge(START, "load")
    builder.add_edge("load", "chunk")
    builder.add_edge("chunk", "embeddings")
    builder.add_edge("embeddings", "vectorstore")
    builder.add_edge("vectorstore", "retriever")
    builder.add_edge("retriever", "llm")
    builder.add_edge("llm", "answer")
    builder.add_edge("answer", END)

    return builder.compile()
