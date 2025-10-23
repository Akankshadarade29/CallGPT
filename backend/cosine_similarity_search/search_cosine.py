from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def cosine_similarity_search(vstore: FAISS, query: str, k: int = 4) -> List[Tuple[Document, float]]:
    """
    Purpose: Run a top-k cosine similarity search against a FAISS vector store.

    Parameters:
    - vstore (FAISS): The FAISS vector store (ideally built with cosine distance).
    - query (str): Natural language query to search with.
    - k (int): Number of results to return.

    Return Value:
    - List[Tuple[Document, float]]: Documents with similarity scores (lower may be better depending on distance metric).

    Side Effects:
    - None.

    Examples:
    >>> # results = cosine_similarity_search(vstore, "What is RAG?", k=3)  # doctest: +SKIP
    """
    return vstore.similarity_search_with_score(query, k=k)
