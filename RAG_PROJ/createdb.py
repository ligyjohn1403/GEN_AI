from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader#TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

#url=["https://www.apple.com/in/macbook-pro/"] #["",""] to send multiple urls 
#loader = TextLoader("RAG_PROJ/document_loader/notes.txt" )

loader = PyPDFLoader("RAG_PROJ/document_loader/deeplearning.pdf" )

docs=loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings() #model_name="sentence-transformers/all-MiniLM-L6-v2"

vectorstore=Chroma.from_documents(documents=chunks,embedding=embeddings,persist_directory="RAG_PROJ/chroma_db")

