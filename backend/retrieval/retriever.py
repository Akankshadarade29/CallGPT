from typing import Any, Dict, List
from langchain_core.documents import Document
from ..vectorstore_pinecone.types import PineconeVectorStoreProtocol


def get_retriever(vstore: PineconeVectorStoreProtocol, search_type: str = "mmr", **search_kwargs) -> Any:
    """
    Purpose: Create a retriever from a vector store.

    Parameters:
    - vstore (PineconeVectorStoreProtocol): The vector store instance.
    - search_type (str): One of {"mmr", "similarity"}.
    - **search_kwargs: Additional search parameters passed to `as_retriever`.

    Return Value:
    - Any: A LangChain retriever object.

    Side Effects:
    - None.

    Examples:
    # retr = get_retriever(vstore, search_type="similarity", k=4)  
    """
    return vstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def retrieve(retriever: Any, query: str) -> List[Document]:
    """
    Purpose: Retrieve relevant documents for the given query.

    Parameters:
    - retriever (Any): Retriever created via `get_retriever`.
    - query (str): Natural language question/query.

    Return Value:
    - List[Document]: Retrieved documents.

    Side Effects:
    - None.

    Examples:
    # docs = retrieve(retriever, "What is RAG?")  
    """
    return retriever.invoke(query)
