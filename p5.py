import os
import streamlit as st

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from langchain_community.document_loaders import PyPDFLoader ,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import os

# check the documents and create the agentic chatbot for question answering task using the uploaded document as knowledge base.

if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "memory" not in st.session_state:
    st.session_state.memory = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []
def process_file(path):

    # load the data
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    #split the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # embeddings the data and create vector store

    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore.from_documents(split_docs, embeddings)

    # create agent need -> tools , llm , system prompt

    llm = ChatGroq(model="openai/gpt-oss-20b")
    
    st.markdown("## 📊 PDF Processing Debug Panel")

    st.markdown(f"""
    ### 📄 Document Status

    - 📁 Total PDFs Loaded: **{len(docs)}**
    - ✂️ Total Chunks Created: **{len(split_docs)}**
    """)

    # Check vector store
    try:
        count = len(vector_store.index_to_docstore_id)
    except:
        count = "Unknown / Error"

    st.markdown(f"""
    ### 🧠 Vector Store Status

    - 📦 Stored Documents: **{count}**
    """)
    st.session_state.debug_info = {
    "docs": len(docs),
    "chunks": len(split_docs),
    "vector_store": len(vector_store.index_to_docstore_id) if hasattr(vector_store, "index_to_docstore_id") else "unknown"
    }
    st.session_state.vector_store = vector_store

    

    @tool
    def retriever_tool(query: str):
        """
        this tool can help you to retrieve the relevent data of the pdf
        documents ,and the document have details about the pandas library in python.
      
        """
        same_docs = vector_store.similarity_search(query ,k = 3)
      
        context = ""

        for doc in same_docs:
            context += doc.page_content + "\n"

        return context


    system_prompt = """
      you are thehelpful assistent that answers question using my knowledge based consist of the uploaded document . ALWAYS use the 'retriever_tool'tool for question requiring external knowledge.  
    """
    # memory 
    memory = InMemorySaver()

    # create agent
    agent = create_agent(model= llm , system_prompt=system_prompt , tools=[retriever_tool], checkpointer=memory)

    st.session_state.agent = agent
    
    st.session_state.documents_uploaded = True

   


# uploading the document UI

if not st.session_state.documents_uploaded:
    uploaded_file = st.file_uploader(label = "Upload a PDF document", type=["pdf"],accept_multiple_files=True)
    
    if uploaded_file:
    # RESET old data
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        st.session_state.messages = []
        
        with st.spinner("Processing the document..."):
            path = "doc_file"
            os.makedirs(path, exist_ok=True)
    
            # clear folder
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
    
            for file in uploaded_file:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())
    
            process_file(path)
            st.rerun()

# chat UI
if st.session_state.documents_uploaded and st.session_state.agent :
    
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)

    
    query = st.chat_input("Ask a question about the uploaded document:")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        res = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "user-1"}}
        )

        answer = res["messages"][-1].content
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})



    

    
