from typing import List
import os
from langchain_core.documents import Document


def load_text_file(file_path: str) -> List[Document]:
    """
    Purpose: Load a single plain-text .txt file and wrap it as LangChain Document(s).

    Parameters:
    - file_path (str): Absolute or relative path to a .txt file.

    Return Value:
    - List[Document]: A list containing a single Document with the file's contents.

    Side Effects:
    - None.

    Examples:
    >>> docs = load_text_file("input.txt")
    >>> len(docs) >= 1
    True
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [Document(page_content=text, metadata={"source": os.path.abspath(file_path)})]
