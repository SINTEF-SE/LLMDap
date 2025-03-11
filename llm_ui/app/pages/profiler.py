from llm import LLM  # Importerer LLM-klassen
import streamlit as st
import json
import tempfile
import subprocess
import requests
import os

import sys
sys.path.append("profiler")

from run_inference import call_inference
import dataset_loader
from metadata_schemas.arxpr2_schema import Metadata_form as schema

def handle_input(uploaded_file, xml_url):
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
            raise ValueError(f"Failed to download file from URL. Status code: {response.status_code}")

    return xml_path

def handle_schema(schema_file):
    if schema_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_schema:
            temp_schema.write(schema_file.read())
            return temp_schema.name
    return None

def run_pipeline(xml_path, schema_path=None):
    """if schema_path is None:
        schema_path = "default_schema.json"
    
    with open (schema_path, "r") as f:
        schema_data = json.load(f)
        loaded_schema = schema.parse_obj(schema_data)"""
    
    #Laster XML-tekst fra fil
    parsed_xml_paper_text = dataset_loader.load_paper_text_from_file_path(xml_path)

    output = call_inference(
        schema,
        parsed_paper_text=parsed_xml_paper_text,
    )

    #Lagrer til uotput.json
    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output

"""# Hjelpefunksjon for å håndtere XML-input (fil eller URL)
def handle_input(uploaded_file, xml_url):
    #Håndterer input enten fra filopplasting eller URL.
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
    #Lagrer JSON-schema midlertidig hvis opplastet.
    if schema_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_schema:
            temp_schema.write(schema_file.read())
            return temp_schema.name
    return None

# Hjelpefunksjon for å kjøre pipelinen
def run_pipeline(xml_path, schema_path=None):
    #Kjører run_inference.py med input-filen og et valgfritt schema.
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
        return None"""


