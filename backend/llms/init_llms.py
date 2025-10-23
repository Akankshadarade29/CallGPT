import os
from typing import Optional


def get_groq_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.1):
    """
    Purpose: Initialize a Groq chat LLM via langchain_groq.

    Parameters:
    - model (str): Groq model name.
    - temperature (float): Sampling temperature.

    Return Value:
    - BaseChatModel: A LangChain ChatModel instance.

    Side Effects:
    - Requires environment variable GROQ_API_KEY.

    Examples:
    >>> # llm = get_groq_llm()  # doctest: +SKIP
    """
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set in environment.")

    return ChatGroq(model=model, temperature=temperature)


def get_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.1):
    """
    Purpose: Initialize an OpenAI chat LLM via langchain-openai.

    Parameters:
    - model (str): OpenAI chat model name.
    - temperature (float): Sampling temperature.

    Return Value:
    - BaseChatModel: A LangChain ChatModel instance.

    Side Effects:
    - Requires environment variable OPENAI_API_KEY and optional OPENAI_BASE_URL.

    Examples:
    >>> # llm = get_openai_llm()  # doctest: +SKIP
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "langchain-openai is required. Add `langchain-openai` to requirements and install."
        ) from e

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in environment.")

    base_url = os.getenv("OPENAI_BASE_URL", None)
    return ChatOpenAI(model=model, temperature=temperature, base_url=base_url)


def get_oss_llm(model: str = "llama3.1", temperature: float = 0.1):
    """
    Purpose: Initialize an OSS chat LLM via local Ollama.

    Parameters:
    - model (str): Ollama model name/tag.
    - temperature (float): Sampling temperature.

    Return Value:
    - BaseChatModel: A LangChain ChatModel instance.

    Side Effects:
    - Requires `ollama` running locally and the specified model pulled.

    Examples:
    >>> # llm = get_oss_llm()  # doctest: +SKIP
    """
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "ChatOllama is unavailable. Ensure `langchain-community` supports it and `ollama` is running."
        ) from e

    return ChatOllama(model=model, temperature=temperature)
