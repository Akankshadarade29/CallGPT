import os
import hashlib
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from backend.chunking.chunk_text import chunk_documents
from backend.embeddings.generate_embeddings import get_embedding_model
from backend.vectorstore_faiss.build_store import build_faiss_from_documents, load_faiss
from backend.retrieval.retriever import get_retriever, retrieve 
from backend.llms.init_llms import get_groq_llm 
from backend.qa_generation.qa import answer_question
from backend._pipeline.pipeline import build_rag_graph


load_dotenv(override=False)

st.set_page_config(page_title="CallGPT", layout="wide")
st.title("CallGPT")

if "vstore" not in st.session_state:
    st.session_state.vstore = None
if "index_dir" not in st.session_state:
    st.session_state.index_dir = None 
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "openai/gpt-oss-120b"

# Sidebar controls
st.sidebar.header("Settings")
 
llm_model = st.sidebar.text_input("LLM Model", value=st.session_state.llm_model)
llm_temperature = st.sidebar.slider("Temperature", 0.5)

emb_model = st.sidebar.text_input("Embeddings Model", value=st.session_state.embeddings_model)

search_type = st.sidebar.radio("Search Type", ["mmr", "similarity"], index=0)
k = st.sidebar.slider("Top-K", 1, 10, 4)
fetch_k = st.sidebar.slider("Fetch-K (MMR)", 5, 50, 20)
lambda_mult = st.sidebar.slider("Lambda (MMR)", 0.0, 1.0, 0.5, 0.05)

persist = st.sidebar.checkbox("Persist FAISS to disk", value=True)

# File uploader
uploaded = st.file_uploader("Upload a .txt file", type=["txt"]) 

# Helpers

def docs_from_upload(uploaded_file) -> List[Document]:
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return [Document(page_content=content, metadata={"source": uploaded_file.name})]


def make_index_dir(file_bytes: bytes) -> str:
    h = hashlib.sha1(file_bytes).hexdigest()[:12]
    p = os.path.join("faiss_index", "ui", h)
    os.makedirs(p, exist_ok=True)
    return p


col1, col2 = st.columns([2, 1])
with col1:
    if uploaded is not None:
        st.subheader("Preview")
        preview = uploaded.getvalue().decode("utf-8", errors="ignore")[:800]
        st.code(preview, language="text")

        if st.button("Build / Update Index", type="primary"):
            try:
                docs = docs_from_upload(uploaded)
                chunks = chunk_documents(docs)
                embeddings = get_embedding_model(emb_model or None)

                if persist:
                    idx_dir = make_index_dir(uploaded.getvalue())
                    build_faiss_from_documents(chunks, embeddings, index_dir=idx_dir)
                    vstore = load_faiss(idx_dir, embeddings)
                    st.session_state.index_dir = idx_dir
                else:
                    vstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.index_dir = None

                st.session_state.vstore = vstore 
                st.session_state.embeddings_model = emb_model or None
                st.session_state.llm_model = llm_model or None

                st.success("Index is ready.")
                # Persist the uploaded content to a stable path for graph-based chat
                try:
                    file_bytes = uploaded.getvalue()
                    content_full = file_bytes.decode("utf-8", errors="ignore")
                    h = hashlib.sha1(file_bytes).hexdigest()[:12]
                    uploads_dir = os.path.join("uploads", "ui")
                    os.makedirs(uploads_dir, exist_ok=True)
                    input_path = os.path.join(uploads_dir, f"{h}.txt")
                    with open(input_path, "w", encoding="utf-8") as f:
                        f.write(content_full)

                    st.session_state.input_path = input_path
                    # Prepare LangGraph chatbot for chat mode
                    st.session_state.chatbot = build_rag_graph()
                except Exception as persist_e:
                    st.info(f"Saved upload for chat failed (chat still usable without graph): {persist_e}")
            except Exception as e:
                st.error(f"Failed to build index: {e}")

with col2:
    st.subheader("Status")
    st.write("Index:", "Ready" if st.session_state.vstore is not None else "Not built")
    if st.session_state.index_dir:
        st.write("Index dir:", st.session_state.index_dir)

st.divider()

# Chat UI using session message history
message_history = st.session_state.get("message_history", [])
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Add user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Ensure index and chatbot are ready
    if not st.session_state.get("index_dir") or not st.session_state.get("input_path"):
        st.warning("Please build the index with an uploaded file first (Persist recommended).")
    else:
        try:
            # Build state for graph invocation
            state = {
                "input_path": st.session_state.get("input_path"),
                "index_dir": st.session_state.get("index_dir"),
                "rebuild": False, 
                "embeddings_model": st.session_state.get("embeddings_model"),
                "llm_model": st.session_state.get("llm_model"),
                "temperature": llm_temperature,
                "search_type": search_type,
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                "question": user_input,
            }
            if "chatbot" not in st.session_state or st.session_state["chatbot"] is None:
                st.session_state["chatbot"] = build_rag_graph()

            result = st.session_state["chatbot"].invoke(state)
            ai_message = result.get("answer", "")

            st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
            with st.chat_message("assistant"):
                st.text(ai_message)
        except Exception as e:
            st.error(f"Chat failed: {e}")
