from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = None):
    """
    Purpose: Return a LangChain Embeddings instance for the specified provider.

    Parameters: 
    - model_name (str): model by the provider.

    Return Value:
    - Embeddings: A LangChain-compatible embeddings object.

    Side Effects:
    - Reads provider-specific environment variables.

    Examples:
    emb = get_embedding_model()
    isinstance(emb, object)
    True
    """
    if model_name is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)