from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages([("system", """
You are an expert movie analyst.

Your task is to extract useful and meaningful information from the given movie paragraph.

Instructions:
- Only use information present in the text. Do NOT guess.
- If something is not available, write "Not mentioned".
- Keep the output clean, structured, and easy to read.

Provide the output in the following format:

Movie Name: 
Release Year: 
Genres: 
Director: 
Writer: 
Cast: 
Language: 
Setting/Location: 
Budget: 
Box Office: 
Key Themes: 
Notable Aspects: 

Quick Summary:
(Write a short 2–3 line summary in simple language)

"""), ("human", """Extract information from this paragraph:{paragraph}""")])

st.title("🎬 Movie Info Extractor")

para = st.text_area("Give the paragraph:", height=200, placeholder="Paste your movie paragraph here...")

if st.button("Extract"):
    if para.strip():
        with st.spinner("Extracting movie info..."):
            final_prompt = prompt.invoke({'paragraph': para})
            response = model.invoke(final_prompt)

        st.subheader("📋 Extracted Information")
        st.text(response.content)
    else:
        st.warning("Please enter a paragraph first.")