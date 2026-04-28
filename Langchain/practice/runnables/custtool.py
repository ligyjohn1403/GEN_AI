from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage


from rich import print
from dotenv import load_dotenv

load_dotenv()
@tool
def greet_name(name:str)->str:
    """ this is greet tool"""
    return f"hello {name} you are doing great learning."
@tool    
def find_length(sentence:str)->int:
    """ this tool finds the length of input string """
    return len(sentence)

tools ={
   'greet_name':greet_name,
    'find_length':find_length
}


llm= ChatMistralAI(model = "mistral-small-2506")

chat_history=[]

prompt = input("you:")

query =HumanMessage(prompt)
chat_history.append(query)

#toolbinding

tool_llm= llm.bind_tools([greet_name,find_length])



#toolcalling

result = tool_llm.invoke(chat_history)

chat_history.append(result)


if result.tool_calls:
    name=result.tool_calls[0]['name']
    tool_message = tools[name].invoke(result.tool_calls[0])
    chat_history.append(tool_message)
    
response = llm.invoke(chat_history)    
print(response.content)