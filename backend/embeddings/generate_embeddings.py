from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(provider: str = "huggingface", model_name: Optional[str] = None):
    """
    Purpose: Return a LangChain Embeddings instance for the specified provider.

    Parameters:
    - provider (str): One of {"huggingface", "openai"}.
    - model_name (Optional[str]): Optional model override for the provider.

    Return Value:
    - Embeddings: A LangChain-compatible embeddings object.

    Side Effects:
    - Reads provider-specific environment variables when needed (e.g., OpenAI API key).

    Examples:
    >>> emb = get_embedding_model()
    >>> isinstance(emb, object)
    True
    """
    provider = provider.lower()

    if provider == "huggingface":
        name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=name)

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "langchain-openai is required for OpenAI embeddings. Add `langchain-openai` to requirements and install."
            ) from e
        name = model_name or "text-embedding-3-small"
        return OpenAIEmbeddings(model=name)

    raise ValueError(f"Unsupported embeddings provider: {provider}")
