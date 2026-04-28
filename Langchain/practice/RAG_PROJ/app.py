from dotenv import load_dotenv
import tempfile
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = "RAG_PROJ/chroma_db"

prompttemplate = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
    ("human", """
  Context: {context}
  Question: {Question}""")
])

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 RAG Document Assistant")

# ── Sidebar: Upload & Index ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Step 1: Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        if st.button("Create Vector DB"):
            with st.spinner("Processing and indexing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader   = PyPDFLoader(tmp_path)
                docs     = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks   = splitter.split_documents(docs)

                embeddings  = HuggingFaceEmbeddings()
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=CHROMA_DIR
                )
                os.unlink(tmp_path)

                for key in ["retriever", "model", "chat_history"]:
                    st.session_state.pop(key, None)

            st.success(f"✅ {len(chunks)} chunks indexed from {len(docs)} pages!")
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
db_exists = os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)

if not db_exists:
    # No DB yet — show instructions, no chat
    st.info("👈 Please upload a PDF and click **Create Vector DB** from the sidebar to get started.")

else:
    # DB exists — load it and show chat
    if "retriever" not in st.session_state:
        with st.spinner("Loading vector database..."):
            try:
                embeddings  = HuggingFaceEmbeddings()
                vectorstore = Chroma(
                    embedding_function=embeddings,
                    persist_directory=CHROMA_DIR
                )
                st.session_state.retriever    = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
                )
                st.session_state.model        = ChatMistralAI(model="mistral-small-2506")
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Failed to load vector DB: {e}")

    # ── Chat history display ──────────────────────────────────────────────
    if "chat_history" in st.session_state:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────
    query = st.chat_input("Ask anything about your document...")

    if query and "retriever" in st.session_state:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                ans_context = st.session_state.retriever.invoke(query)
                context     = "\n\n".join(ans.page_content for ans in ans_context)
                prompt      = prompttemplate.invoke({"context": context, "Question": query})
                response    = st.session_state.model.invoke(prompt)
            st.markdown(response.content)

        st.session_state.chat_history.append({"role": "assistant", "content": response.content})