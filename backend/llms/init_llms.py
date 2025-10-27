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
    # llm = get_groq_llm()  
    """
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set in environment.")

    return ChatGroq(model=model, temperature=temperature)


 