from typing import List, Any
from langchain_core.documents import Document


def mmr_search(
    vstore: Any,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> List[Document]:
    """
    Purpose: Run Max Marginal Relevance (MMR) search for diverse, semantically-relevant results.

    Parameters:
    - vstore (Any): The vector store.
    - query (str): Natural language query.
    - k (int): Number of final results.
    - fetch_k (int): How many candidates to fetch before MMR filtering.
    - lambda_mult (float): Diversity factor in [0,1], higher favors diversity.

    Return Value:
    - List[Document]: Selected documents using MMR.

    Side Effects:
    - None.

    Examples:
    # docs = mmr_search(vstore, "Tell me about RAG", k=5)  # doctest: +SKIP
    """
    return vstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
