from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

vectorstore=Chroma.from_documents(documents=docs,embedding=embeddings)

ret=vectorstore.as_retriever(search_type ="mmr",search_kwargs={'k':3})

mmr_docs = ret.invoke("What is gradient descent?")

for r in mmr_docs:
     print(r.page_content)
                                  
ret1=vectorstore.as_retriever(search_type ="similarity",search_kwargs={'k':3})

mmr_docs = ret1.invoke("What is gradient descent?")

for r in mmr_docs:
     print(r.page_content)