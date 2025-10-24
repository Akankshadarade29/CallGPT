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
from backend.prompt_templates.templates import get_qa_prompt
from backend.llms.init_llms import get_groq_llm, get_openai_llm, get_oss_llm
from backend.qa_generation.qa import answer_question


load_dotenv(override=False)

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("Simple RAG Demo (Dynamic Upload)")

if "vstore" not in st.session_state:
    st.session_state.vstore = None
if "index_dir" not in st.session_state:
    st.session_state.index_dir = None
if "embeddings_provider" not in st.session_state:
    st.session_state.embeddings_provider = "huggingface"
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = None
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "groq"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None

# Sidebar controls
st.sidebar.header("Settings")

llm_provider = st.sidebar.selectbox("LLM Provider", ["groq", "openai", "oss"], index=["groq", "openai", "oss"].index(st.session_state.llm_provider))
llm_model_default = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
    "oss": "llama3.1",
}[llm_provider]
llm_model = st.sidebar.text_input("LLM Model", value=st.session_state.llm_model or llm_model_default)
llm_temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.1, 0.1)

emb_provider = st.sidebar.selectbox("Embeddings Provider", ["huggingface", "openai"], index=["huggingface", "openai"].index(st.session_state.embeddings_provider))
emb_model = st.sidebar.text_input("Embeddings Model (optional)", value=st.session_state.embeddings_model or "")

search_type = st.sidebar.radio("Search Type", ["mmr", "similarity"], index=0)
k = st.sidebar.slider("Top-K", 1, 10, 4)
fetch_k = st.sidebar.slider("Fetch-K (MMR)", 5, 50, 20)
lambda_mult = st.sidebar.slider("Lambda (MMR)", 0.0, 1.0, 0.5, 0.05)

template = st.sidebar.selectbox("Prompt Template", ["default", "concise"], index=0)

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
                embeddings = get_embedding_model(emb_provider, emb_model or None)

                if persist:
                    idx_dir = make_index_dir(uploaded.getvalue())
                    build_faiss_from_documents(chunks, embeddings, index_dir=idx_dir, use_cosine=True)
                    vstore = load_faiss(idx_dir, embeddings)
                    st.session_state.index_dir = idx_dir
                else:
                    vstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.index_dir = None

                st.session_state.vstore = vstore
                st.session_state.embeddings_provider = emb_provider
                st.session_state.embeddings_model = emb_model or None
                st.session_state.llm_provider = llm_provider
                st.session_state.llm_model = llm_model or None

                st.success("Index is ready.")
            except Exception as e:
                st.error(f"Failed to build index: {e}")

with col2:
    st.subheader("Status")
    st.write("Index:", "Ready" if st.session_state.vstore is not None else "Not built")
    if st.session_state.index_dir:
        st.write("Index dir:", st.session_state.index_dir)

st.divider()

q = st.text_input("Ask a question about the uploaded document")
ask = st.button("Ask")

if ask:
    if st.session_state.vstore is None:
        st.warning("Please upload a file and build the index first.")
    elif not q.strip():
        st.warning("Please enter a question.")
    else:
        try:
            if llm_provider == "groq":
                llm = get_groq_llm(model=llm_model or llm_model_default, temperature=llm_temperature)
            elif llm_provider == "openai":
                llm = get_openai_llm(model=llm_model or llm_model_default, temperature=llm_temperature)
            else:
                llm = get_oss_llm(model=llm_model or llm_model_default, temperature=llm_temperature)

            if search_type == "mmr":
                retriever = get_retriever(
                    st.session_state.vstore,
                    search_type="mmr",
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                )
            else:
                retriever = get_retriever(st.session_state.vstore, search_type="similarity", k=k)

            docs = retrieve(retriever, q)
            prompt = get_qa_prompt(template=template)
            ans = answer_question(llm, prompt, docs, q)

            st.subheader("Answer")
            st.write(ans)

            with st.expander("Retrieved Context"):
                for i, d in enumerate(docs, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(d.page_content)
                    if d.metadata:
                        st.caption(str(d.metadata))
        except Exception as e:
            st.error(f"Failed to answer: {e}")
