from __future__ import annotations

import os
from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from supabase import create_client, Client

from .custom_store import CustomSupabaseVectorStore


def _client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_API_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_API_KEY (or SERVICE_KEY) must be set")
    return create_client(url, key)


def build_supabase_from_documents(
    docs: List[Document],
    embeddings: Embeddings,
    table_name: str = "documents",
    *,
    query_name: str = "match_documents",
    chunk_size: int = 500,
) -> str:
    """
    Purpose: Upsert chunked documents + embeddings into Supabase (pgvector).

    Return Value:
    - str: Table name used.
    
    Side Effects:
    - Inserts documents with embeddings into Supabase table via direct insert.
    """
    sb = _client()
    
    # Generate embeddings for all documents
    texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)
    
    # Prepare batch insert records
    records = []
    for doc, vector in zip(docs, vectors):
        records.append({
            "id": str(uuid4()),
            "content": doc.page_content,
            "metadata": doc.metadata or {},
            "embedding": vector,
        })
    
    # Batch insert in chunks to avoid payload limits
    for i in range(0, len(records), chunk_size):
        batch = records[i:i + chunk_size]
        sb.table(table_name).insert(batch).execute()
    
    return table_name


essential_params = {"table_name", "query_name"}


def load_supabase(
    table_name: str,
    embeddings: Embeddings,
    *,
    query_name: str = "match_documents",
) -> CustomSupabaseVectorStore:
    """
    Purpose: Load a CustomSupabaseVectorStore for querying.
    
    Parameters:
    - table_name (str): Supabase table name.
    - embeddings (Embeddings): Embedding model instance.
    - query_name (str): RPC function name for vector search.
    
    Return Value:
    - CustomSupabaseVectorStore: Vector store instance for retrieval.
    """
    sb = _client()
    return CustomSupabaseVectorStore(
        client=sb,
        embeddings=embeddings,
        table_name=table_name,
        query_name=query_name,
    )
