from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


def _format_context(docs: List[Document]) -> str:
    """
    Purpose: Create a single context string from retrieved documents.

    Parameters:
    - docs (List[Document]): Retrieved documents.

    Return Value:
    - str: Concatenated context text.

    Side Effects:
    - None.

    Examples:
    >>> _format_context([Document(page_content="A"), Document(page_content="B")])
    'A\n\nB'
    """
    return "\n\n".join(d.page_content for d in docs)


def answer_question(llm, prompt: ChatPromptTemplate, docs: List[Document], question: str) -> str:
    """
    Purpose: Generate an answer from an LLM using retrieved context and a prompt template.

    Parameters:
    - llm: A LangChain ChatModel.
    - prompt (ChatPromptTemplate): Template with {context} and {question} placeholders.
    - docs (List[Document]): Retrieved documents.
    - question (str): User question.

    Return Value:
    - str: The model's answer text.

    Side Effects:
    - None.

    Examples:
    >>> # ans = answer_question(llm, prompt, docs, "What is RAG?")  # doctest: +SKIP
    """
    context = _format_context(docs)
    messages = prompt.format_messages(context=context, question=question)
    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))
