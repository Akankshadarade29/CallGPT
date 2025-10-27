from typing import Literal
from langchain_core.prompts import ChatPromptTemplate


def get_qa_prompt(template: Literal["default", "concise"] = "default") -> ChatPromptTemplate:
    """
    Purpose: Return a chat prompt template for RAG QA.

    Parameters:
    - template (Literal["default", "concise"]): The name of the template variant.

    Return Value:
    - ChatPromptTemplate: A LangChain chat prompt ready to be formatted.

    Side Effects:
    - None.

    Examples:
    prompt = get_qa_prompt()
    isinstance(prompt, ChatPromptTemplate)
    True
    """
    if template == "concise":
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You answer the user's question concisely using only the provided context. If missing, say you don't know."),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Use ONLY the provided context to answer. If the answer is not present, say you don't know."),
            (
                "human",
                (
                    "You are given context to answer a question.\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n"
                    "Provide a helpful, accurate answer with citations to the context when relevant."
                ),
            ),
        ]
    )
