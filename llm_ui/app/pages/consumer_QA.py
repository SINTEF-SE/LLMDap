import streamlit as st
import json
import sys
import os
from llm import LLM

def load_previous_datasets():
    """Laster inn tidligere søkte datasett fra fil."""
    try:
        with open("previous_datasets.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def load_context():
    """Laster inn metadata (title, abstract) fra context.json."""
    try:
        with open("context.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {'title': '', 'abstract': ''}

def load_settings():
    """Laster inn konfigurasjonsparametere fra settings.json."""
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'temperature': 0.0, 'max_tokens': 50, 'prompt_template': ''}

def show(llm):
    st.title("Dataset Discovery & Q&A")

    query = st.text_input("Enter your query to find datasets:")

    if st.button("Search"):
        with st.spinner("Searching for datasets... Please wait."):
            response = llm.ask(f"Find datasets related to: {query}")

            try:
                datasets = json.loads(response)  
            except json.JSONDecodeError:
                datasets = [{"title": response, "url": "#"}]

            with open("previous_datasets.json", "w") as f:
                json.dump(datasets, f)

            st.subheader("Search Results")
            if datasets:
                for dataset in datasets:
                    st.markdown(f"[{dataset['title']}]({dataset['url']})")
            else:
                st.info("No datasets found.")

    previous_datasets = load_previous_datasets()
    if previous_datasets:
        st.subheader("Ask Questions About Your Selected Datasets")

        selected_datasets = st.multiselect("Choose datasets for Q&A", previous_datasets)
        if selected_datasets:
            context = load_context()
            settings = load_settings()

            question = st.chat_input("Enter your question:")

            if question:
                with st.spinner("Generating answer..."):
                    combined_context = "\n".join([d["title"] for d in selected_datasets])
                    prompt = settings['prompt_template']
                    prompt = prompt.replace("{question}", question)
                    prompt = prompt.replace("{title}", context["title"])
                    prompt = prompt.replace("{abstract}", context["abstract"])
                    prompt = prompt.replace("{datasets}", combined_context)

                    output = llm.ask(prompt, max_tokens=settings['max_tokens'], temperature=settings['temperature'])

                    st.subheader("Answer:")
                    st.write(output)


"""import sys
import os
from llm import LLM
import streamlit as st
import json



# Hjelpefunksjoner for å laste inn tidligere data
def load_previous_datasets():
    Laster inn tidligere opplastede datasett fra fil.
    try:
        with open("previous_datasets.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def load_context():
    Laster inn metadata (title, abstract) fra context.json.
    try:
        with open("context.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {'title': '', 'abstract': ''}

def load_settings():
    Laster inn konfigurasjonsparametere fra settings.json.
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'temperature': 0.0, 'max_tokens': 50, 'prompt_template': ''}

# Streamlit UI
st.title("QA View - Ask Questions About Your Datasets")

# Laste inn tidligere datasett
previous_datasets = load_previous_datasets()
if not previous_datasets:
    st.warning("No datasets available. Please upload datasets first in the Provider View.")
    st.stop()

# Brukeren velger datasettene som skal brukes i QA
selected_datasets = st.multiselect("Choose datasets for Q&A", previous_datasets)
if not selected_datasets:
    st.info("Select at least one dataset to proceed with the Q&A session.")
    st.stop()

# Laste inn metadata og innstillinger
context = load_context()
settings = load_settings()

# Spørsmål fra brukeren
st.subheader("Ask the LLM about the selected datasets")
question = st.chat_input("Enter your question:")

if question:
    with st.spinner("Generating answer..."):
        # Opprette prompt basert på valgte datasett og metadata
        combined_context = "\n".join(selected_datasets)
        prompt = settings['prompt_template']
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{title}", context["title"])
        prompt = prompt.replace("{abstract}", context["abstract"])
        prompt = prompt.replace("{datasets}", combined_context)

        # Koble til LLM og generere svar
        llm = LLM()
        output = llm.ask(prompt, max_tokens=settings['max_tokens'], temperature=settings['temperature'])

        # Vise svaret i UI-et
        st.subheader("Answer:")
        st.write(output)"""



