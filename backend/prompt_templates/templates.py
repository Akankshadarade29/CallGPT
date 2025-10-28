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
            (
                "system",
                (
                    "You are CallGPT — a voice-friendly AI support assistant .\n"
                    "Your role is to assist customers calling the company by giving clear, polite, and accurate spoken-style answers.\n"
                    "Use ONLY the provided context to answer the question.\n"
                    "If the answer is not found in the context, say politely that you don’t have that information.\n"
                    "Keep responses short, natural, and conversational (like talking on a call).\n"
                    "Do NOT make up information. Do NOT reference 'documents' or 'context' explicitly."
                ),
            ),
            (
                "human",
                (
                    "Context:\n{context}\n\n"
                    "Customer Question: {question}\n\n"
                    "Answer naturally as if you are speaking to the customer on a call."
                ),
            ),
        ]
    )

