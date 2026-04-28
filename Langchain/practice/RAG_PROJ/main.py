from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

embeddings = HuggingFaceEmbeddings() 
vectorstore = Chroma(embedding_function=embeddings,persist_directory="RAG_PROJ/chroma_db")

retriver = vectorstore.as_retriever(search_type ="mmr",
                                    search_kwargs = {"k":3,
                                                   "fetch_k":10,
                                                   "lambda_mult":0.5})

model=ChatMistralAI(model="mistral-small-2506");

prompttemplate= ChatPromptTemplate.from_messages([("system","""You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
                                                  ("human","""
  Context: {context}
  Question:{Question}    """)])

print("RAG created.Press 0 to end")

while True:
    query = input("You query about DLP:")
    if query == "0": break
    else:
        ans_context = retriver.invoke(query) 
        context = ("/n/n").join(ans.page_content for ans in ans_context)
        prompt = prompttemplate.invoke({"context":context,"Question" :query })
        response=model.invoke(prompt)    
        print(response.content)
