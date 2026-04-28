from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = ChatMistralAI(model =  "mistral-small-2506")

template = ChatPromptTemplate.from_template("Explain {topic} in 10 sentences")

parser = StrOutputParser()

chain = template | llm | parser

result  = chain.invoke ("string theory")

print(result)