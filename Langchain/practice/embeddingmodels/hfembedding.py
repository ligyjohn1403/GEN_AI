from dotenv import load_dotenv

load_dotenv()

print("API keys loaded")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", 
                                )  #all-MiniLM-L6-v2 - another free model
text=["ligy john",
      "pljohn"]
#vector = embeddings.embed_query("ligy john")- single query /sentence
vector = embeddings.embed_documents(text) # mul
print(vector)