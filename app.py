import streamlit as st
import tempfile
import os
import shutil
import uuid
from ingest import ingest, ingest_temporary, embeddings, CHROMA_PATH
from query import query
from langchain_community.vectorstores import Chroma

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Q&A Assistant")
st.caption("Upload documents or ask questions about the default knowledge base.")

# ── Session state ─────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "db" not in st.session_state:
    st.session_state.db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = False
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, CSVs, or TXT files",
        type=["pdf", "csv", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Uploaded Files", type="primary"):
            with st.spinner("Processing documents..."):
                temp_dir = tempfile.mkdtemp()
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                st.session_state.db = ingest_temporary(temp_dir, existing_db=st.session_state.db)
                shutil.rmtree(temp_dir)
            st.success(f"✅ Added {len(uploaded_files)} file(s) to knowledge base")

    st.divider()

    if st.button("Use Default Documents"):
        with st.spinner("Loading default knowledge base..."):
            DATA_PATH = r"C:\Users\batla\OneDrive\Desktop\RAGdocumnets"
            st.session_state.db = ingest(DATA_PATH)
            st.session_state.chat_history = []
        st.success("✅ Default documents loaded")

    st.divider()

    # calculate filenames first
    if st.session_state.db:
        all_docs = st.session_state.db.get()
        filenames = list(set([
            m.get("file_name")
            for m in all_docs.get("metadatas", [])
            if m and "file_name" in m
        ]))
        st.subheader("📚 Loaded Documents")
        for f in filenames:
            st.write(f"• {f}")
    else:
        filenames = []  # empty list before any DB is loaded
        st.info("No documents loaded yet. Upload files or click 'Use Default Documents'.")

    st.divider()

    # NOW selectbox has filenames available
    selected_file = st.selectbox(
        "Filter answers to a specific document (optional)",
        options=["All documents"] + filenames
    )

# ── Auto load default DB on first run ─────────────────────
if st.session_state.db is None:
    DATA_PATH = r"C:\Users\batla\OneDrive\Desktop\RAGdocumnets"
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        st.session_state.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

# ── Chat history display ──────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

# ── Chat input ────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    if st.session_state.db is None:
        st.error("Please upload documents or click 'Use Default Documents' first.")
    else:
        # show user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "sources": None
        })

        # get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = query(st.session_state.db, user_input, filter_file=selected_file if selected_file != "All documents" else None)
            st.write(answer)
            if sources:
                with st.expander("📎 Sources"):
                    for source in sources:
                        st.write(f"• {source}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })