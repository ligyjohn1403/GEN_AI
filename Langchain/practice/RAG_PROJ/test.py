from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",# default /n/n
    chunk_size=10,
    chunk_overlap=1,
  
)

loader = TextLoader("RAG_PROJ/document_loader/notes.txt" )
docs=loader.load()
chunks = text_splitter.split_documents(docs)


for i in chunks:
  print(i.page_content)