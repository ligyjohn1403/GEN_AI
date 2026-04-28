import os
import requests

from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from tavily import TavilyClient 

from rich import print
from dotenv import load_dotenv

load_dotenv()

@tool
def weather_report(city:str)->str:

    """ This tool should provide weather related info about the  city in India"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    
    response = requests.get(url)
    data = response.json()
    
    #print(f"debug:{data}")
    if response.status_code != 200:
        return f"Error: {data.get('message', 'Unable to fetch weather')}"

    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]

    return f""" Weather report of {city} is  temperature  {temp} °C and humidity is  {humidity} . The Weather is {weather} """
#result= weather_report.invoke("aluva")
#print(result)

@tool
def get_city_news(city: str)->str:
    """ This tool should provide latest news about the  city in India"""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    query = f"latest news in {city} India"

    response = client.search(
        query=query,
        search_depth="basic",  # better results-advanced
        max_results=2
    )

    results = response.get("results", [])

    if not results:
        return "No news found."
    
    news_list = []
   
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

#result= get_city_news.invoke("aluva")
#print(result)

tools ={
   'weather_report':weather_report,
    'get_city_news':get_city_news
}

llm= ChatMistralAI(model = "mistral-small-2506")

tool_llm= llm.bind_tools([weather_report,get_city_news])

#AGENT LOOP 
message=[]
print("City intelligence system")
print("type 0 to quit")

while True:
    userchat= input("user:")
    if userchat == '0':break
    message.append(HumanMessage(content = userchat))

    while True:
        results=tool_llm.invoke(message)
        message.append(results)
        if results.tool_calls:
          for toolcall in results.tool_calls:
              toolname= toolcall['name']
              #HUMAN in loop 
              confirmation = input(f"{toolname} Agent to be used yes/no:")
              if confirmation.lower() == 'no': 
                  print(f"Not authorizsed to use the agent {toolname}")
                  message.append( ToolMessage(
            content=f"User denied access to tool {toolname}. Answer without using tools." ,
                                         tool_call_id =toolcall['id'])
    )
              else:   
              #execute the call
               tool_message=tools[toolname].invoke(toolcall)
               message.append(ToolMessage(content=tool_message,
                                         tool_call_id =toolcall['id']))
              continue
        else :
            print("\n\n")
            print(results.content)
            print("\n\n")
            break