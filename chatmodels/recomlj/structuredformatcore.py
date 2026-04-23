from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from  langchain_mistralai import ChatMistralAI
from  pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
model = ChatMistralAI(model= "mistral-small-2506");

class Movie(BaseModel):
    title: str
    cast: List[str]
    realase_year: Optional[int]
    summary: str

parser=PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([("system","""
You are an expert movie analyst.

Your task is to extract useful and meaningful information from the given movie paragraph.

                                            {format_instructions}

"""),("human","""
      Extract information from this paragraph:{paragraph}""")])

para= input("give the para")
final_prompt= prompt.invoke({'paragraph' :para,
                             'format_instructions':parser.get_format_instructions()})
response = model.invoke(final_prompt);

data= parser.parse(response.content)
print(data.model_dump_json(indent=2))