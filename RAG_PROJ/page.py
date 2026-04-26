from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.apple.com/in/macbook-pro/" )
docs=loader.load()

print((docs[0].page_content)) # each page in pdf will bcome one document