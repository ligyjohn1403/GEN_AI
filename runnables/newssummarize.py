from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)

llm = ChatMistralAI(model = "mistral-small-2506")

parser = StrOutputParser()

seq= prompt | llm | parser

search_tool =TavilySearchResults(max_result = 2)

lresult = search_tool.run("Latest IPL news of 2026")


result= seq.invoke({"news":lresult})

print(result)

