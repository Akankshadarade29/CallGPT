from typing import Optional


def read_question(prompt: str = "Enter your question: ", default: Optional[str] = None) -> str:
    """
    Purpose: Read a question from standard input with an optional default.

    Parameters:
    - prompt (str): CLI prompt to display.
    - default (Optional[str]): Default question if the user provides empty input.

    Return Value:
    - str: The user's question string (or default).

    Side Effects:
    - Reads from standard input.

    Examples:
    >>> # q = read_question(default="What is RAG?")  # doctest: +SKIP
    """
    try:
        q = input(prompt).strip()
    except EOFError:
        q = ""

    if not q:
        if default is None:
            raise ValueError("Question is empty and no default provided.")
        return default
    return q
