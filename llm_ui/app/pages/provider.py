import sys
import os
import streamlit as st
import json
import requests
from pages import profiler


# Streamlit UI
st.title("Provider View - Dataset Profiling")

# Last opp XML-fil eller angi URL
st.subheader("Upload a scientific paper (.xml) or provide a URL")

uploaded_file = st.file_uploader("Choose an .xml file", type=["xml"], key="xml_uploader")
xml_url = st.text_input("Or provide a URL to an .xml file")

# (Valgfritt) Last opp JSON-schema-fil
st.subheader("Upload an optional schema file (JSON)")
schema_file = st.file_uploader("Choose a JSON schema file", type=["json"], key="json_uploader")

# Kjør pipelinen
if st.button("Run pipeline"):
    with st.spinner("Running the pipeline... Please wait."):
        xml_path = profiler.handle_input(uploaded_file, xml_url)  # Kaller funksjon fra profiler.py

        if xml_path:
            schema_path = profiler.handle_schema(schema_file)  # Håndterer schema-opplasting

            output_json = profiler.run_pipeline(xml_path, schema_path)  # Kaller pipeline
            if output_json:
                st.subheader("Pipeline Output")
                json_text = st.text_area("Edit the JSON output:", json.dumps(output_json, indent=4), height=300)

                # La brukeren lagre oppdatert JSON
                if st.button("Save Updated JSON"):
                    with open("updated_output.json", "w") as f:
                        f.write(json_text)
                    st.success("Updated JSON saved!")