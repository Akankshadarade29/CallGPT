from typing import Any, Dict, List, Protocol
from langchain_core.documents import Document


class PineconeVectorStoreProtocol(Protocol):
    def as_retriever(self, *, search_type: str, search_kwargs: Dict[str, Any]) -> Any: ...

    def max_marginal_relevance_search(
        self,
        query: str,
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Document]: ...
