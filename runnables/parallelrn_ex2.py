from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

template1 = ChatPromptTemplate.from_template("Explain {topic} in short ")

template2 = ChatPromptTemplate.from_template("Explain {topic} in 10 lines ")

llm = ChatMistralAI(model= "mistral-small-2506")

parser = StrOutputParser()

chain = RunnableParallel({
 "short": RunnableLambda(lambda x : x["short"])  |template1 | llm |parser,
"10lines": RunnableLambda(lambda x : x["10lines"])  | template2 | llm |parser,
})

result = chain.invoke({"short": {"topic": "string theory"},"10lines": {"topic": "oxidization"}})

print(result)