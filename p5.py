from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent          # ✅ FIXED import
from langgraph.checkpoint.memory import MemorySaver        # ✅ FIXED import

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – PDF Chatbot",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS  (dark card theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
body, .stApp { background: #0f1117; color: #e0e0e0; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

/* ---------- top header bar ---------- */
.top-bar {
    background: linear-gradient(135deg, #1a1f2e 0%, #1e2640 100%);
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.top-bar h1 { margin: 0; font-size: 1.6rem; color: #7eb8f7; }
.top-bar p  { margin: 0; color: #8a95ab; font-size: 0.9rem; }

/* ---------- upload card ---------- */
.upload-card {
    background: #161b27;
    border: 2px dashed #2e4a7a;
    border-radius: 14px;
    padding: 40px 30px;
    text-align: center;
    margin: 40px auto;
    max-width: 560px;
}
.upload-card h2 { color: #7eb8f7; margin-bottom: 6px; }
.upload-card p  { color: #6b7a99; }

/* ---------- chat bubbles ---------- */
.chat-user {
    background: #1e2d4a;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    margin: 8px 0 8px auto;
    max-width: 75%;
    color: #d4e8ff;
    font-size: 0.95rem;
    word-wrap: break-word;
}
.chat-assistant {
    background: #1a2035;
    border: 1px solid #2a3550;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 18px;
    margin: 8px auto 8px 0;
    max-width: 75%;
    color: #c8d8f0;
    font-size: 0.95rem;
    word-wrap: break-word;
}
.role-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    opacity: 0.65;
}

/* ---------- status badge ---------- */
.badge-success {
    display: inline-block;
    background: #0d3326;
    color: #3ecf8e;
    border: 1px solid #1a5c42;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-info {
    display: inline-block;
    background: #1a2540;
    color: #7eb8f7;
    border: 1px solid #2a3f6a;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
}

/* ---------- streamlit overrides ---------- */
.stChatInputContainer textarea { background: #1a1f2e !important; color: #e0e0e0 !important; }
div[data-testid="stChatMessage"] { background: transparent !important; }
.stButton button {
    background: linear-gradient(135deg, #2d5be3, #1e3fa8);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton button:hover { background: linear-gradient(135deg, #3d6cf5, #2a50c8); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
for key, default in {
    "documents_uploaded": False,
    "agent": None,
    "messages": [],
    "uploaded_file_names": [],
    "processing_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DOC_DIR   = "./doc_files/"
THREAD_ID = "user-session-1"

# ─────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────
def process_files(path: str):
    """Load PDFs → chunk → embed → build agent."""
    try:
        loader = PyPDFDirectoryLoader(path)
        docs   = loader.load()

        if not docs:
            raise ValueError("No text could be extracted from the uploaded PDFs.")

        splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        embeddings   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = InMemoryVectorStore.from_documents(split_docs, embeddings)

        # ✅ Use a valid ollama model name
        llm = ChatOllama(model="gemma3:1b", temperature=0)

        @tool
        def retriever_tool(query: str) -> str:
            """
            Retrieves relevant excerpts from the uploaded PDF documents.
            Always use this tool when answering questions about document content.
            """
            similar_docs = vector_store.similarity_search(query, k=4)
            return "\n\n".join(doc.page_content for doc in similar_docs) or "No relevant content found."

        system_prompt = (
            "You are DocMind, a helpful AI assistant that answers questions based solely "
            "on the uploaded PDF documents. ALWAYS call the 'retriever_tool' before answering "
            "any factual question. If the answer is not found in the documents, say so clearly. "
            "Keep answers concise, accurate, and well-structured."
        )

        memory = MemorySaver()
        agent  = create_react_agent(
            model=llm,
            tools=[retriever_tool],
            prompt=system_prompt,
            checkpointer=memory,
        )

        st.session_state.agent              = agent
        st.session_state.documents_uploaded = True
        st.session_state.processing_error   = None

    except Exception as e:
        st.session_state.processing_error = str(e)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("---")

    if st.session_state.documents_uploaded:
        st.markdown('<span class="badge-success">✔ Documents Ready</span>', unsafe_allow_html=True)
        st.markdown("**Loaded files:**")
        for name in st.session_state.uploaded_file_names:
            st.markdown(f"📄 `{name}`")

        st.markdown("---")
        if st.button("🗑️ Clear & Upload New"):
            for key in ["documents_uploaded", "agent", "messages",
                        "uploaded_file_names", "processing_error"]:
                st.session_state[key] = False if key == "documents_uploaded" else \
                                        None  if key in ("agent", "processing_error") else []
            if os.path.exists(DOC_DIR):
                shutil.rmtree(DOC_DIR)
            st.rerun()
    else:
        st.markdown("Upload one or more PDF files to get started.")

    st.markdown("---")
    st.markdown("**Model:** `llama3-70b-8192`")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
    st.caption("Powered by Groq + LangChain + LangGraph")

# ─────────────────────────────────────────────
# Header bar
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div>
    <h1>🧠 DocMind</h1>
    <p>Upload your PDFs and ask anything — your personal document intelligence assistant</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Upload screen
# ─────────────────────────────────────────────
if not st.session_state.documents_uploaded:
    st.markdown("""
    <div class="upload-card">
      <h2>📂 Upload Your Documents</h2>
      <p>Supports multiple PDFs — all content becomes your searchable knowledge base</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Documents", use_container_width=True):
                os.makedirs(DOC_DIR, exist_ok=True)   # ✅ create dir safely
                with st.spinner("Embedding documents — this may take a moment…"):
                    for f in uploaded_files:
                        with open(os.path.join(DOC_DIR, f.name), "wb") as fp:
                            fp.write(f.getvalue())
                    st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                    process_files(DOC_DIR)

                if st.session_state.processing_error:
                    st.error(f"❌ Error: {st.session_state.processing_error}")
                else:
                    st.success("✅ Documents processed successfully!")
                    st.rerun()   # ✅ rerun AFTER spinner exits, not inside it

# ─────────────────────────────────────────────
# Chat screen
# ─────────────────────────────────────────────
if st.session_state.documents_uploaded and st.session_state.agent:

    # Render message history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                  <div class="role-label">YOU</div>
                  {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-assistant">
                  <div class="role-label">🧠 DOCMIND</div>
                  {msg["content"]}
                </div>""", unsafe_allow_html=True)

    # Chat input
    query = st.chat_input("Ask a question about your documents…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking…"):
            try:
                result = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": THREAD_ID}},
                )
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"⚠️ Sorry, something went wrong: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
