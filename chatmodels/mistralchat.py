from dotenv import load_dotenv

load_dotenv()

print("API keys loaded")
from  langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model= "mistral-small-2506");

response = model.invoke("Why do parrots talk?");
print(response.content)