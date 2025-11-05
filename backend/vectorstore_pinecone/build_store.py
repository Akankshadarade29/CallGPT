from __future__ import annotations

import os
from uuid import uuid4
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Pinecone + LangChain integration
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def _resolve_dimension(embeddings: Embeddings) -> int:
    """
    Infer embedding dimension from the provided embedding model by embedding a tiny string.
    """
    try:
        vec = embeddings.embed_query("_dim_")
        return len(vec)
    except Exception as e:
        raise RuntimeError(f"Could not infer embedding dimension: {e}")


def build_pinecone_from_documents(
    docs: List[Document],
    embeddings: Embeddings,
    index_name: str,
    *,
    cloud: str = "aws",
    region: str = "us-east-1",
    metric: str = "cosine",
    dimension: Optional[int] = None,
) -> str:
    """
    Purpose: Build/ensure a Pinecone serverless index exists and add documents to it.

    Parameters:
    - docs (List[Document]): Chunked documents to index.
    - embeddings (Embeddings): Embedding model for encoding.
    - index_name (str): Pinecone index name.
    - cloud (str): Cloud provider for serverless index.
    - region (str): Region for serverless index.
    - metric (str): Similarity metric ('cosine' default).
    - dimension (Optional[int]): Embedding dimension; inferred if None.

    Return Value:
    - str: The Pinecone index name used.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment")

    pc = Pinecone(api_key=api_key)

    # Create index if missing
    names = {idx.name for idx in pc.list_indexes()}
    if index_name not in names:
        dim = dimension or _resolve_dimension(embeddings)
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    index = pc.Index(index_name)
    vstore = PineconeVectorStore(index=index, embedding=embeddings)

    # Stable UUID prefix per batch
    prefix = str(uuid4())
    ids = [f"{prefix}-{i}" for i in range(len(docs))]
    vstore.add_documents(documents=docs, ids=ids)
    return index_name


def load_pinecone(index_name: str, embeddings: Embeddings) -> PineconeVectorStore:
    """
    Purpose: Load a PineconeVectorStore for a given index.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)
