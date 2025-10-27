from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] | None = None,
) -> List[Document]:
    """
    Purpose: Split input Documents into smaller chunks for downstream embedding and retrieval.

    Parameters:
    - docs (List[Document]): Input documents to split.
    - chunk_size (int): Max characters per chunk.
    - chunk_overlap (int): Overlap in characters between consecutive chunks.
    - separators (Optional[List[str]]): Custom separators to guide splitting. Defaults to sensible values.

    Return Value:
    - List[Document]: Chunked documents with metadata preserved.

    Side Effects:
    - None.

    Examples:
    chunks = chunk_documents([Document(page_content="a" * 3000)])
    len(chunks) >= 2
    True
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return splitter.split_documents(docs)
