import os
import hashlib
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import backend.app as backend_app
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv(override=False)

st.set_page_config(page_title="CallGPT", layout="wide", page_icon="💬")

# Custom CSS for ChatGPT-like UI
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Main title */
    h1 {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 CallGPT")

##################### Utilities #####################

def generate_thread_id():
    return backend_app.conversation.generate_thread_id()

def add_thread(thread_id: str) -> None:
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat_ui() -> None:
    tid = generate_thread_id()
    st.session_state["thread_id"] = tid
    add_thread(tid)
    st.session_state["message_history"] = []

def retrieve_all_threads():
    all_threads = set()
    cp = st.session_state.get("checkpointer")
    if not cp:
        return list(all_threads)
    try:
        for checkpoint in cp.list(None):
            tid = checkpoint.config.get('configurable', {}).get('thread_id')
            if tid:
                all_threads.add(tid)
    except Exception:
        pass
    return list(all_threads)

def load_conversation_ui(thread_id: str):
    if not st.session_state.get("chatbot"):
        return []
    return backend_app.conversation.load_conversation(st.session_state["chatbot"], thread_id)

def docs_from_upload(uploaded_file) -> List[Document]:
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return [Document(page_content=content, metadata={"source": uploaded_file.name})]


def make_index_dir(file_bytes: bytes) -> str:
    h = hashlib.sha1(file_bytes).hexdigest()[:12]
    p = os.path.join("faiss_index", "ui", h)
    os.makedirs(p, exist_ok=True)
    return p


if "vstore" not in st.session_state:
    st.session_state.vstore = None
if "index_dir" not in st.session_state:
    st.session_state.index_dir = None 
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "openai/gpt-oss-120b"
if "checkpointer" not in st.session_state:
    db_path = os.path.join('db', 'chatbot.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(database=db_path, check_same_thread=False)
    st.session_state.checkpointer = SqliteSaver(conn=conn)
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads() or []
if "thread_id" not in st.session_state:
    if st.session_state["chat_threads"]:
        # Prefer an existing persisted thread so its messages can be shown immediately
        st.session_state["thread_id"] = st.session_state["chat_threads"][0]
    else:
        # No persisted threads: create a fresh one and add it to the list
        st.session_state["thread_id"] = backend_app.conversation.generate_thread_id()
        st.session_state["chat_threads"].append(st.session_state["thread_id"])
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {st.session_state.thread_id: []}

# Ensure chatbot is available to read persisted messages on first load
if "chatbot" not in st.session_state or st.session_state["chatbot"] is None:
    try:
        st.session_state["chatbot"] = backend_app.build_rag_graph(checkpointer=st.session_state.checkpointer)
    except Exception:
        st.session_state["chatbot"] = None

# Hydrate the active thread's history from the checkpointer so messages appear after refresh
try:
    if st.session_state.get("chatbot"):
        _msgs = backend_app.conversation.load_conversation(st.session_state["chatbot"], st.session_state["thread_id"])
        _hist = backend_app.conversation.convert_messages_to_chat_history(_msgs)
        st.session_state.chat_histories[st.session_state["thread_id"]] = _hist
except Exception:
    pass

# ============ Sidebar: Conversation History ============
st.sidebar.title("💬 CallGPT")

if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
    new_tid, new_threads, new_histories = backend_app.conversation.reset_chat(
        st.session_state.chat_threads,
        st.session_state.chat_histories,
    )
    st.session_state.thread_id = new_tid
    st.session_state.chat_threads = new_threads
    st.session_state.chat_histories = new_histories
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📚 My Conversations")

# Display conversations in reverse chronological order (newest first)
for thread_id in st.session_state.chat_threads[::-1]:
    # Get preview for thread
    if st.session_state.get("chatbot"):
        preview = backend_app.conversation.get_thread_preview(st.session_state.chatbot, thread_id, max_length=35)
    else:
        # Fallback if chatbot not ready
        preview = f"Thread {thread_id[:8]}..."
    
    # Highlight active thread
    is_active = (thread_id == st.session_state.thread_id)
    button_type = "primary" if is_active else "secondary"
    
    if st.sidebar.button(
        f"{'🟢 ' if is_active else ''}  {preview}",
        key=f"thread_{thread_id}",
        use_container_width=True,
        type=button_type,
    ):
        if thread_id != st.session_state.thread_id:
            # Switch to this thread
            st.session_state.thread_id = thread_id
            
            # Load conversation from checkpointer if chatbot is ready
            if st.session_state.get("chatbot"):
                messages = backend_app.conversation.load_conversation(st.session_state.chatbot, thread_id)
                chat_history = backend_app.conversation.convert_messages_to_chat_history(messages)
                st.session_state.chat_histories[thread_id] = chat_history
            
            st.rerun()

st.sidebar.divider()

# Sidebar controls
with st.sidebar.expander("⚙️ Settings", expanded=False):
    llm_model = st.text_input("LLM Model", value=st.session_state.llm_model)
    llm_temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
    
    emb_model = st.text_input("Embeddings Model", value=st.session_state.embeddings_model)
    
    search_type = st.radio("Search Type", ["mmr", "similarity"], index=0)
    k = st.slider("Top-K", 1, 10, 4)
    fetch_k = st.slider("Fetch-K (MMR)", 5, 50, 20)
    lambda_mult = st.slider("Lambda (MMR)", 0.0, 1.0, 0.5, 0.05)
    
    persist = st.checkbox("Persist FAISS to disk", value=True)

# File uploader
uploaded = st.file_uploader("Upload a .txt file", type=["txt"]) 



col1, col2 = st.columns([2, 1])
with col1:
    if uploaded is not None:
        st.subheader("Preview")
        preview = uploaded.getvalue().decode("utf-8", errors="ignore")[:800]
        st.code(preview, language="text")

        if st.button("Build / Update Index", type="primary"):
            try:
                docs = docs_from_upload(uploaded)
                chunks = backend_app.chunking.chunk_documents(docs)
                emb_model_obj = backend_app.embeddings.get_embedding_model(emb_model or None)

                if persist:
                    idx_dir = make_index_dir(uploaded.getvalue())
                    backend_app.vectorstore_faiss.build_faiss_from_documents(chunks, emb_model_obj, index_dir=idx_dir)
                    vstore = backend_app.vectorstore_faiss.load_faiss(idx_dir, emb_model_obj)
                    st.session_state.index_dir = idx_dir
                else:
                    vstore = FAISS.from_documents(chunks, emb_model_obj)
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
                    st.session_state.chatbot = backend_app.build_rag_graph(checkpointer=st.session_state.checkpointer)
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

# Chat UI using session message history keyed by thread_id
tid = st.session_state.get("thread_id")
if tid not in st.session_state.chat_histories:
    st.session_state.chat_histories[tid] = []
st.session_state["message_history"] = st.session_state.chat_histories[tid]

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Add user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Ensure index and chatbot are ready
    if not st.session_state.get("index_dir") or not st.session_state.get("input_path"):
        st.warning("Please build the index with an uploaded file first (Persist recommended).")
    else:
        try:
            base_state = {
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
            }
            if "chatbot" not in st.session_state or st.session_state["chatbot"] is None:
                st.session_state["chatbot"] = backend_app.build_rag_graph(checkpointer=st.session_state.checkpointer)
            state = backend_app.streaming.build_messages_state(user_input, base_state=base_state)

            def ai_only_stream():
                yield from backend_app.streaming.stream_ai_tokens(
                    st.session_state["chatbot"],
                    state,
                    st.session_state["thread_id"],
                )

            with st.chat_message("assistant"):
                ai_message = st.write_stream(ai_only_stream())
            st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
            st.session_state.chat_histories[tid] = st.session_state["message_history"]
        except Exception as e:
            st.error(f"Chat failed: {e}")
