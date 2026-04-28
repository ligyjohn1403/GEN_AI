from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
import streamlit as st

load_dotenv()
model = ChatMistralAI(model="mistral-small-2506")

class Movie(BaseModel):
    title: str
    cast: List[str]
    realase_year: Optional[int]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([("system", """
You are an expert movie analyst.

Your task is to extract useful and meaningful information from the given movie paragraph.

{format_instructions}

"""), ("human", """Extract information from this paragraph:{paragraph}""")])

st.title("🎬 Movie Info Extractor")

para = st.text_area("Give the paragraph:", height=200, placeholder="Paste your movie paragraph here...")

if st.button("Extract"):
    if para.strip():
        with st.spinner("Extracting movie info..."):
            final_prompt = prompt.invoke({
                'paragraph': para,
                'format_instructions': parser.get_format_instructions()
            })
            response = model.invoke(final_prompt)
            data = parser.parse(response.content)

        st.subheader("📋 Extracted Output")
        st.json(data.model_dump())

    else:
        st.warning("Please enter a paragraph first.")