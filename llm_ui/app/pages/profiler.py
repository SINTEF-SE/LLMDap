from llm import LLM  # Importerer LLM-klassen
import streamlit as st
import json
import tempfile
import subprocess
import requests
import os

# Hjelpefunksjon for å håndtere XML-input (fil eller URL)
def handle_input(uploaded_file, xml_url):
    """Håndterer input enten fra filopplasting eller URL."""
    xml_path = None

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_file:
            temp_file.write(uploaded_file.read())
            xml_path = temp_file.name
    elif xml_url:
        response = requests.get(xml_url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_file:
                temp_file.write(response.content)
                xml_path = temp_file.name
        else:
            st.error(f"Failed to download the file from the URL. Status code: {response.status_code}")
    
    return xml_path

# Hjelpefunksjon for å håndtere JSON-schema-opplasting
def handle_schema(schema_file):
    """Lagrer JSON-schema midlertidig hvis opplastet."""
    if schema_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_schema:
            temp_schema.write(schema_file.read())
            return temp_schema.name
    return None

# Hjelpefunksjon for å kjøre pipelinen
def run_pipeline(xml_path, schema_path=None):
    """Kjører run_inference.py med input-filen og et valgfritt schema."""
    if not schema_path:
        schema_path = os.path.abspath("default_schema.json")  # Sett en standard JSON-schema hvis ingen er gitt

    command = ["python", "profiler/run_inference.py", xml_path, schema_path, "output.json"]

    
    if schema_path and schema_path not in command:
        command.append(schema_path)


    try:
        subprocess.run(command, check=True)
        with open("output.json", "r") as json_file:
            return json.load(json_file)
    except subprocess.CalledProcessError as e:
        st.error(f"Error while running the pipeline: {e}")
        return None


"""import streamlit as st
import fitz  # PyMuPDF
import json

def show():
    st.write("## Please provide information associated with the dataset")
    
    # PDF uploader widget
    uploaded_file = st.file_uploader("Upload the paper (PDF)", type="pdf")

    # Text input widget
    title = st.text_input("Enter the title here:")
    abstract = st.text_area("Enter the abstract here:")

    # File uploader widget
    uploaded_dataset = st.file_uploader("Upload the dataset")

    # Submit button
    if st.button("Submit"):
        if uploaded_file is not None:
            # Display file details
            st.write("### PDF File Details")
            st.write("Filename:", uploaded_file.name)
            st.write("File type:", uploaded_file.type)
            st.write("File size:", uploaded_file.size, "bytes")

            # Read and display the first few lines of the PDF file
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_document:
                first_page = pdf_document.load_page(0)  # Load the first page
                pdf_text = first_page.get_text("text")  # Extract text as plain text
                lines = pdf_text.split('\n')  # Split text into lines
                st.write("### First Few Lines of the PDF:")
                st.write('\n'.join(lines[:5]))  # Display the first 5 lines
        
        if abstract:
            # Display the entered text
            st.write("### Entered Text")
            st.write(abstract)


        if uploaded_dataset is not None:
            # Display file details
            st.write("### Dataset File Details")
            st.write("Filename:", uploaded_dataset.name)
            st.write("File type:", uploaded_dataset.type)
            st.write("File size:", uploaded_dataset.size, "bytes")

            # Read and display the content of the dataset file
            file_bytes = uploaded_dataset.read()
            st.write("File content (first 200 bytes):", file_bytes[:200])
 
        with open("context.json", "w") as f:
            json.dump({
                'title':title,
                'abstract':abstract
                }, f)"""

