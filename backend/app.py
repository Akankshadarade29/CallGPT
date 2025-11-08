"""
Purpose: End-to-end RAG CLI to load a text file, chunk it, embed, persist FAISS, retrieve, and ask an LLM to answer a question.

Usage examples:
python -m backend.app --input input.txt --rebuild --question "Where vector is used?"

Environment:
- GROQ_API_KEY for Groq (if --llm-provider groq) 
"""

import os
import argparse
from dotenv import load_dotenv
from uuid import uuid4
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from ._pipeline import pipeline

 
from . import question_input, streaming, chunking, embeddings, conversation, vectorstore_supabase, input_text
from .storage import supabase_storage as sstore
# from ._pipeline import build_rag_graph

# Re-export build_rag_graph for convenience
build_rag_graph = pipeline.build_rag_graph

__all__ = [
    "run_rag",
    "build_rag_graph",
    # feature modules re-exported for UI convenience
    "chunking",
    "embeddings",
    "vectorstore_supabase",
    "conversation",
    "streaming",
]


def _need_rebuild(index_dir: str) -> bool:
    files = {"index.faiss", "index.pkl"}
    return not os.path.isdir(index_dir) or any(
        not os.path.exists(os.path.join(index_dir, f)) for f in files
    )


def run_rag(
    input_path: str,
    # supabase vector store db
    table_name: str = "documents",
    query_name: str = "match_documents",
    
    rebuild: bool = False,
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
    llm_model: str = "openai/gpt-oss-120b",
    temperature: float = 0.1,
    search_type: str = "mmr",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    question: str | None = None,
    thread_id: str | None = None,
) -> str:
    """
    Purpose: Execute the entire RAG pipeline and return the final answer string.

    Parameters:
    - input_path (str): Path to input .txt file.
    - index_dir (str): Directory to persist/load FAISS index.
    - rebuild (bool): Force rebuild of FAISS index. 
    - embeddings_model (str): embedding model name. 
    - llm_model (str): LLM model name.
    - temperature (float): LLM sampling temperature.
    - search_type (str): One of {"mmr", "similarity"}.
    - k (int): Top-k to retrieve.
    - fetch_k (int): Candidates fetched for MMR.
    - lambda_mult (float): Diversity factor for MMR. 
    - question (Optional[str]): Question text. If None, prompt via stdin.

    Return Value:
    - str: Model's answer.

    Side Effects:
    - Creates/reads local FAISS index directory.
    - Reads environment variables for LLM providers.
    """
    # Load environment (.env) if present
    load_dotenv(override=False)
    # Read question before graph run if not provided
    q = question if question else question_input.read_question()

    # If requested, build vector store and upload file to Supabase Storage
    if rebuild:
        docs = input_text.load_text_file(input_path)
        chunks = chunking.chunk_documents(docs)
        emb = embeddings.get_embedding_model(embeddings_model)
        vectorstore_supabase.build_supabase_from_documents(
            chunks, emb, table_name=table_name, query_name=query_name
        )
        with open(input_path, "rb") as f:
            file_bytes = f.read()
        bucket_name = sstore.get_bucket_name()
        sstore.ensure_bucket_exists(bucket_name, public=True)
        obj_name = os.path.basename(input_path)
        upload_res = sstore.upload_text_bytes(
            bucket_name=bucket_name,
            object_name=obj_name,
            data=file_bytes,
            upsert=True,
        )
        meta = sstore.get_file_metadata(bucket_name=bucket_name, object_name=obj_name)
        sstore.store_metadata_record(
            bucket_name=bucket_name,
            object_name=obj_name,
            meta=meta,
            public_url=upload_res.get("public_url") if isinstance(upload_res, dict) else None,
        )

    # Build graph with persistent Sqlite checkpointer
    db_path = os.path.join('db', 'chatbot.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(database=db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    thread_id = thread_id or f"cli-{uuid4()}"
    app = pipeline.build_rag_graph(checkpointer=checkpointer)

    # Build state for streaming (messages-based)
    base_state = {
        "input_path": input_path,
        "table_name": table_name,
        "query_name": query_name,
        "rebuild": False,
        "embeddings_model": embeddings_model,
        "llm_model": llm_model,
        "temperature": temperature,
        "search_type": search_type,
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
    }
    state = streaming.build_messages_state(q, base_state=base_state)

    # Stream tokens to stdout and accumulate final answer
    answer_parts = []
    for piece in streaming.stream_ai_tokens(app, state, thread_id):
        print(piece, end="", flush=True)
        answer_parts.append(piece)
    print()  # newline after streaming
    return "".join(answer_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="CallGPT")
    parser.add_argument("--input", type=str, default="input.txt", help="Path to input .txt file")
    parser.add_argument("--table-name", type=str, default="documents", help="Supabase table for pgvector")
    parser.add_argument("--query-name", type=str, default="match_documents", help="Supabase RPC name for vector search")
    parser.add_argument("--rebuild", action="store_true", help="Upsert chunks into Supabase (pgvector)")
 
    parser.add_argument("--embeddings-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    parser.add_argument("--llm-model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--temperature", type=float, default=0.5)

    parser.add_argument("--search-type", type=str, choices=["mmr", "similarity"], default="mmr")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--lambda-mult", type=float, default=0.5)
 
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None, help="Existing thread_id to continue the conversation")
    parser.add_argument("--list-threads", action="store_true", help="List all persisted thread IDs and exit")

    args = parser.parse_args()

    if args.list_threads:
        # List persisted thread IDs from sqlite checkpointer
        db_path = os.path.join('db', 'chatbot.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(database=db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        seen = set()
        for cp in checkpointer.list(None):
            try:
                tid = cp.config.get('configurable', {}).get('thread_id')
                if tid:
                    seen.add(tid)
            except Exception:
                continue
        print("\nPersisted threads:")
        for tid in sorted(seen):
            print("-", tid)
        return

    answer = run_rag(
        input_path=args.input,
        table_name=args.table_name,
        query_name=args.query_name,
        rebuild=args.rebuild,
        embeddings_model=args.embeddings_model, 
        llm_model=args.llm_model,
        temperature=args.temperature,
        search_type=args.search_type,
        k=args.k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult, 
        question=args.question,
        thread_id=args.thread_id,
    )

    print("\n=== Answer ===\n")


if __name__ == "__main__":
    main()
