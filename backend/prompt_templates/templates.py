from langchain_core.prompts import ChatPromptTemplate


def get_qa_prompt() -> ChatPromptTemplate:
    """
    Purpose: Return a chat prompt template for RAG QA.

    Return Value:
    - ChatPromptTemplate: A LangChain chat prompt ready to be formatted.

    Side Effects:
    - None.

    Examples:
    prompt = get_qa_prompt()
    isinstance(prompt, ChatPromptTemplate)
    True
    """
    
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
