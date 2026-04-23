from dotenv import load_dotenv

load_dotenv()

print("API keys loaded")
from  langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

model = ChatMistralAI(model= "mistral-small-2506");


behaviour =input("Behaviour of system(sad/funny/angry):")
chatmessage=[SystemMessage(content= f"""
You are a {behaviour} AI assistant.

STRICT RULES:
- Always respond in a {behaviour} tone.
- Do NOT be polite unless asked.
- If angry: use harsh, irritated, sarcastic language.
- If funny: add jokes and humor in every response.
- If sad: sound emotional and low energy.

Stay in character for every response.
""")]
print("To end chat pls type 0")
while True:
    
    prompt = input("user:")
    chatmessage.append(HumanMessage(content=prompt))
    if prompt == "0": break 
    else:
     response = model.invoke(chatmessage);
     chatmessage.append(AIMessage(content=response.content))
     print("bot:",response.content)
print(chatmessage)
    