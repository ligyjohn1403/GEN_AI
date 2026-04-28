from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import TokenTextSplitter #RecursiveCharacterTextSplitter


loader = PyPDFLoader("RAG_PROJ/document_loader/GRU.pdf" )
docs=loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

texts = text_splitter.split_documents(docs)


#print((docs[5].page_content)) # each page in pdf will bcome one document
print(len(texts))