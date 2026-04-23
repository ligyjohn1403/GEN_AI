from dotenv import load_dotenv

load_dotenv()

print("API keys loaded")
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)
model = ChatHuggingFace(llm=llm)

response = model.invoke("Why do parrots talk?");
print(response.content)