from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.title("🤖 AI Chatbot")

model = ChatMistralAI(model="mistral-small-2506")

# Behaviour selector (only shown if chat hasn't started)
if "chatmessage" not in st.session_state:
    st.subheader("Choose AI Behaviour")
    behaviour = st.selectbox("Behaviour of system:", ["funny", "sad", "angry"])

    if st.button("Start Chat"):
        st.session_state.behaviour = behaviour
        st.session_state.chatmessage = [SystemMessage(content=f"""
You are a {behaviour} AI assistant.

STRICT RULES:
- Always respond in a {behaviour} tone.
- Do NOT be polite unless asked.
- If angry: use harsh, irritated, sarcastic language.
- If funny: add jokes and humor in every response.
- If sad: sound emotional and low energy.

Stay in character for every response.
""")]
        st.rerun()

else:
    # Show current behaviour badge
    behaviour_emoji = {"funny": "😂", "sad": "😢", "angry": "😠"}
    st.caption(f"Current mode: {behaviour_emoji.get(st.session_state.behaviour, '🤖')} {st.session_state.behaviour.capitalize()}")

    # Display chat history (skip the SystemMessage)
    for msg in st.session_state.chatmessage[1:]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    # Chat input
    prompt = st.chat_input("Type your message... (type 0 to end)")

    if prompt:
        st.session_state.chatmessage.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
            st.write(prompt)

        if prompt == "0":
            st.info("Chat ended. Here's the full chat history:")
            #st.write(st.session_state.chatmessage)
            st.write("Bye")
        else:
            response = model.invoke(st.session_state.chatmessage)
            st.session_state.chatmessage.append(AIMessage(content=response.content))

            with st.chat_message("assistant"):
                st.write(response.content)

    # Reset button to change behaviour
    if st.button("🔄 Change Behaviour"):
        del st.session_state.chatmessage
        del st.session_state.behaviour
        st.rerun()