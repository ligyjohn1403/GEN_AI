from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

template1 = ChatPromptTemplate.from_template("Explain {topic} in short ")

template2 = ChatPromptTemplate.from_template("Explain {topic} in 10 lines ")

llm = ChatMistralAI(model= "mistral-small-2506")

parser = StrOutputParser()

chain = RunnableParallel({
 "short": template1 | llm |parser,
"detailed": template2 | llm |parser,

})

result = chain.invoke("string theory")

print(result)