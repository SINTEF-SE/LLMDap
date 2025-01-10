import streamlit as st
import json
from llm import LLM

# Load settings from the local file
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        return {'temp': 0.0, 'token': '', 'prompt_template' :''}

def load_context():
    try:
        with open('context.json', 'r') as f:
            context = json.load(f)
        return context
    except FileNotFoundError:
        return {'title': '', 'abstract': ''}

def show():
    st.write("## Display Parameters")
    
    # Load settings
    settings = load_settings()

    # Display settings
    st.write("### Parameters from Configuration")
    st.write(f"Temperature (temp): {settings['temperature']}")
    st.write(f"Token: {settings['maximal number of tokens generated']}")

    # Connect LLM for Q&A
    st.write("### Q&A session - ask LLM what you want to know!")
    question = st.chat_input("Ask the llm anything:")

    if question:
        st.write("Question:")
        st.write(question)

        st.write("... generating ...")

        context = load_context() 
        prompt = settings['prompt_template']
        prompt = question.join(prompt.split("{question}"))
        prompt = context["title"].join(prompt.split("{title}"))
        prompt = context["abstract"].join(prompt.split("{abstract}"))


        llm = LLM()
        output = llm.ask(prompt, max_tokens = settings['maximal number of tokens generated'], temperature = settings['temperature'])

        st.write(output)
