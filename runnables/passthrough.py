from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough 

template1 = ChatPromptTemplate.from_template("write a simple python code {topic} ")

template2 = ChatPromptTemplate.from_template("Explain the {code} ")

llm = ChatMistralAI(model= "mistral-small-2506")

parser = StrOutputParser()

seq= template1 |llm | parser

seq2 = RunnableParallel({
    "code1": RunnablePassthrough(),
    "explaination":template2 | llm |parser
})

chain = seq |seq2

result = chain.invoke("write a code to reverse a string")

print(result['code1'])
print(result['explaination'])