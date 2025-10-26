"""
Purpose: End-to-end RAG CLI to load a text file, chunk it, embed, persist FAISS, retrieve, and ask an LLM to answer a question.

Usage examples:
python app.py --input input.txt --rebuild --llm-provider groq --question "What is RAG?"
python app.py --input input.txt --search-type similarity --k 5 --template concise

Environment:
- GROQ_API_KEY for Groq (if --llm-provider groq)
- OPENAI_API_KEY (and optional OPENAI_BASE_URL) for OpenAI (if --llm-provider openai)
- Ollama must be running locally for OSS (if --llm-provider oss)
"""

import os
import argparse
from dotenv import load_dotenv

from backend.input_text.load_text import load_text_file
from backend.chunking.chunk_text import chunk_documents
from backend.embeddings.generate_embeddings import get_embedding_model
from backend.vectorstore_faiss.build_store import build_faiss_from_documents, load_faiss
from backend.retrieval.retriever import get_retriever, retrieve
from backend.prompt_templates.templates import get_qa_prompt
from backend.llms.init_llms import get_groq_llm, get_openai_llm, get_oss_llm
from backend.question_input.question import read_question
from backend.qa_generation.qa import answer_question
from backend._pipeline.pipeline import build_rag_graph


def _need_rebuild(index_dir: str) -> bool:
    files = {"index.faiss", "index.pkl"}
    return not os.path.isdir(index_dir) or any(
        not os.path.exists(os.path.join(index_dir, f)) for f in files
    )


def run_rag(
    input_path: str,
    index_dir: str = "faiss_index",
    rebuild: bool = False,
    embeddings_provider: str = "huggingface",
    embeddings_model: str | None = None,
    llm_provider: str = "groq",
    llm_model: str | None = None,
    temperature: float = 0.1,
    search_type: str = "mmr",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    template: str = "default",
    question: str | None = None,
) -> str:
    """
    Purpose: Execute the entire RAG pipeline and return the final answer string.

    Parameters:
    - input_path (str): Path to input .txt file.
    - index_dir (str): Directory to persist/load FAISS index.
    - rebuild (bool): Force rebuild of FAISS index.
    - embeddings_provider (str): One of {"huggingface", "openai"}.
    - embeddings_model (Optional[str]): Optional embedding model name.
    - llm_provider (str): One of {"groq", "openai", "oss"}.
    - llm_model (Optional[str]): Optional LLM model name.
    - temperature (float): LLM sampling temperature.
    - search_type (str): One of {"mmr", "similarity"}.
    - k (int): Top-k to retrieve.
    - fetch_k (int): Candidates fetched for MMR.
    - lambda_mult (float): Diversity factor for MMR.
    - template (str): Prompt template variant {"default", "concise"}.
    - question (Optional[str]): Question text. If None, prompt via stdin.

    Return Value:
    - str: Model's answer.

    Side Effects:
    - Creates/reads local FAISS index directory.
    - Reads environment variables for LLM providers.
    """
    # Load environment (.env) if present
    load_dotenv(override=False)

    # Read question before graph run if not provided
    q = question if question else read_question(default="What is the document about?")

    # Build graph and export diagram
    app = build_rag_graph()

    # Invoke graph with initial state
    state = {
        "input_path": input_path,
        "index_dir": index_dir,
        "rebuild": rebuild,
        "embeddings_provider": embeddings_provider,
        "embeddings_model": embeddings_model,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "temperature": temperature,
        "search_type": search_type,
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
        "template": template,
        "question": q,
    }
    result = app.invoke(state)
    return result.get("answer", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG CLI")
    parser.add_argument("--input", type=str, default="input.txt", help="Path to input .txt file")
    parser.add_argument("--index-dir", type=str, default="faiss_index", help="Directory for FAISS index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild FAISS index")

    parser.add_argument("--embeddings-provider", type=str, choices=["huggingface", "openai"], default="huggingface")
    parser.add_argument("--embeddings-model", type=str, default=None)

    parser.add_argument("--llm-provider", type=str, choices=["groq", "openai", "oss"], default="groq")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--search-type", type=str, choices=["mmr", "similarity"], default="mmr")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--lambda-mult", type=float, default=0.5)

    parser.add_argument("--template", type=str, choices=["default", "concise"], default="default")
    parser.add_argument("--question", type=str, default=None)

    args = parser.parse_args()

    answer = run_rag(
        input_path=args.input,
        index_dir=args.index_dir,
        rebuild=args.rebuild,
        embeddings_provider=args.embeddings_provider,
        embeddings_model=args.embeddings_model,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
        search_type=args.search_type,
        k=args.k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        template=args.template,
        question=args.question,
    )

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
