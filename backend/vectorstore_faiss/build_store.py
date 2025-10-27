from typing import List, Optional
import os

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


def build_faiss_from_documents(
    docs: List[Document],
    embeddings: Embeddings,
    index_dir: str = "./faiss_index",
     
) -> str:
    """
    Purpose: Build a FAISS vector store from documents and persist it locally.

    Parameters:
    - docs (List[Document]): The chunked documents to index.
    - embeddings (Embeddings): Embedding model used for encoding documents/queries.
    - index_dir (str): Directory path where the FAISS index will be saved.
   
    Return Value:
    - str: The directory path where the FAISS index is saved.

    Side Effects:
    - Creates directories and files under `index_dir`.

    Examples:
    path = build_faiss_from_documents([], embeddings=None) 
    isinstance(path, str)
    True
    """
    os.makedirs(index_dir, exist_ok=True)
    try:
        vstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        print(f"Failed to build FAISS index: {e}")
        return None
    

    vstore.save_local(index_dir)
    return index_dir


essential_files = {"index.faiss", "index.pkl"}


def load_faiss(index_dir: str, embeddings: Embeddings) -> FAISS:
    """
    Purpose: Load a previously saved FAISS vector store from disk.

    Parameters:
    - index_dir (str): Directory path containing the saved FAISS artifacts.
    - embeddings (Embeddings): Embedding model for query-time encoding.

    Return Value:
    - FAISS: The loaded vector store instance.

    Side Effects:
    - None.

    Examples:
    vstore = load_faiss("./faiss_index", emb)  
    """
    missing = [f for f in essential_files if not os.path.exists(os.path.join(index_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"FAISS index missing files in '{index_dir}': {', '.join(missing)}"
        )

    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
