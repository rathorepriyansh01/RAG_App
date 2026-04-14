from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
os.environ["GROQ_API_KEY"] = "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"

st.set_page_config(
    page_title="DocMind – PDF Chatbot",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
body, .stApp { background: #0f1117; color: #e0e0e0; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
.top-bar {
    background: linear-gradient(135deg, #1a1f2e 0%, #1e2640 100%);
    border: 1px solid #2a3550; border-radius: 12px;
    padding: 18px 28px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 14px;
}
.top-bar h1 { margin: 0; font-size: 1.6rem; color: #7eb8f7; }
.top-bar p  { margin: 0; color: #8a95ab; font-size: 0.9rem; }
.upload-card {
    background: #161b27; border: 2px dashed #2e4a7a;
    border-radius: 14px; padding: 40px 30px;
    text-align: center; margin: 40px auto; max-width: 560px;
}
.upload-card h2 { color: #7eb8f7; margin-bottom: 6px; }
.upload-card p  { color: #6b7a99; }
.chat-user {
    background: #1e2d4a; border-radius: 16px 16px 4px 16px;
    padding: 12px 18px; margin: 8px 0 8px auto;
    max-width: 75%; color: #d4e8ff; font-size: 0.95rem; word-wrap: break-word;
}
.chat-assistant {
    background: #1a2035; border: 1px solid #2a3550;
    border-radius: 16px 16px 16px 4px; padding: 12px 18px;
    margin: 8px auto 8px 0; max-width: 75%; color: #c8d8f0;
    font-size: 0.95rem; word-wrap: break-word;
}
.role-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 4px; opacity: 0.65; }
.badge-success {
    display: inline-block; background: #0d3326; color: #3ecf8e;
    border: 1px solid #1a5c42; border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; font-weight: 600;
}
.stChatInputContainer textarea { background: #1a1f2e !important; color: #e0e0e0 !important; }
div[data-testid="stChatMessage"] { background: transparent !important; }
.stButton button {
    background: linear-gradient(135deg, #2d5be3, #1e3fa8);
    color: white; border: none; border-radius: 8px; font-weight: 600;
}
.stButton button:hover { background: linear-gradient(135deg, #3d6cf5, #2a50c8); }
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in {
    "documents_uploaded": False,
    "vector_store": None,
    "messages": [],
    "uploaded_file_names": [],
    "processing_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

DOC_DIR = "./doc_files/"

def process_files(path: str):
    """Load PDFs -> chunk -> embed -> save vector store."""
    try:
        loader = PyPDFDirectoryLoader(path)
        docs   = loader.load()

        if not docs:
            raise ValueError("No text could be extracted from the uploaded PDFs.")

        splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        embeddings   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = InMemoryVectorStore.from_documents(split_docs, embeddings)

        st.session_state.vector_store       = vector_store
        st.session_state.documents_uploaded = True
        st.session_state.processing_error   = None

    except Exception as e:
        st.session_state.processing_error = str(e)


def get_answer(question: str, chat_history: list) -> str:
    """Retrieve context and answer question using LLM directly — no agent needed."""
    try:
        # Step 1: Retrieve relevant chunks
        docs = st.session_state.vector_store.similarity_search(question, k=2)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Step 2: Build history string
        history_str = ""
        for msg in chat_history[-6:]:  # last 3 exchanges only
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        # Step 3: Build prompt
        prompt = ChatPromptTemplate.from_template("""
You are DocMind, a helpful assistant that answers questions based on the provided document context.

Chat History:
{history}

Document Context:
{context}

User Question: {question}

Instructions:
- Answer ONLY based on the document context above
- If the answer is not in the context, say "I couldn't find this in the uploaded documents"
- Be concise and clear
- Do NOT return JSON or tool calls — just answer in plain text

Answer:
""")

        from langchain_groq import ChatGroq

        llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({
            "history": history_str,
            "context": context,
            "question": question,
        })

        return answer.strip()

    except Exception as e:
        return f"⚠️ Error: {e}"


# ── Sidebar ──────────────────────────────────
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
            for key in ["documents_uploaded", "vector_store", "messages",
                        "uploaded_file_names", "processing_error"]:
                st.session_state[key] = False if key == "documents_uploaded" else \
                                        None  if key in ("vector_store", "processing_error") else []
            if os.path.exists(DOC_DIR):
                shutil.rmtree(DOC_DIR)
            st.rerun()
    else:
        st.markdown("Upload one or more PDF files to get started.")
    st.markdown("---")
    st.markdown("**Model:** `gemma3:1b` (Ollama)")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
    st.caption("Powered by Ollama + LangChain")

# ── Header ───────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div>
    <h1>🧠 DocMind</h1>
    <p>Upload your PDFs and ask anything — your personal document intelligence assistant</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload Screen ─────────────────────────────
if not st.session_state.documents_uploaded:
    st.markdown("""
    <div class="upload-card">
      <h2>📂 Upload Your Documents</h2>
      <p>Supports multiple PDFs — all content becomes your searchable knowledge base</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Documents", use_container_width=True):
                os.makedirs(DOC_DIR, exist_ok=True)
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
                    st.rerun()

# ── Chat Screen ───────────────────────────────
if st.session_state.documents_uploaded and st.session_state.vector_store:

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

    query = st.chat_input("Ask a question about your documents…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Thinking…"):
            answer = get_answer(query, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
