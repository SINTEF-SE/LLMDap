import streamlit as st
import requests
import io
from typing import Optional
import sys
from pathlib import Path

# Add project root to Python path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add root directory (llmdap/profiler)

from profiler.api.llmdap_options import LLMDapOptions

def main():
    st.title("UPCAST Data Profiling Demo")

    # Get list of available apps
    #TODO: enter the port as a parameter when starting up
    st.subheader("1. Select Profiler")
    response = requests.get("http://localhost:8001/profile/get_profilers")
    available_apps = response.json()

    print(f"returned apps: {available_apps}")

    
    # Display dropdown menu for selecting profiler
    selected_app = st.selectbox("Select Profiler", available_apps)

    st.subheader("2. Provide Input File or URL (the dataset to be profiled or the paper for LLMDap)")
    input_method = st.radio("Choose input method:", ["Upload File", "Enter URL"])

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload a file")
    else:
        file_url = st.text_input("Enter URL")


    # Option 1: User upload a local file
    #uploaded_file = st.file_uploader("Upload a local file")

    # Option 2: User enters a URL
    #file_url = st.text_input("Or enter a file URL")

    # Process the file
    file_content = None
    filename = None

    # 2. Select an App
    #st.subheader("2. Select Profiler")
    #selected_app = st.selectbox("Select Profiler:", available_apps)

    # 3. If App is LLMDap, show extra input fields
    schema_file = None
    similarity_k = None
    field_info_to_compare = None

    if selected_app == "LLMDap":
        st.subheader("3. LLMDap Configuration")
        schema_file = st.file_uploader("Upload Schema File. Omit if you are using default schema.")
        similarity_k = st.number_input("Similarity_k. Omit if you are using default value.", value=5, step=1, format="%d")
        field_info_to_compare = st.text_input("Field_info_to_compare. Omit if you are using default value.", value = "choices")

    # save values in the Options parameter
    options = LLMDapOptions(
        #path=path or None,
        #url=url or None,
        similarity_k=int(similarity_k) if similarity_k else None,

        field_info_to_compare=field_info_to_compare or None,
        schema_file=schema_file.name if schema_file else None
    )

    # Submit button
    if st.button("Generate profile"):
        if uploaded_file:
            # Send request to FastAPI to run selected app with uploaded file
            files = {"file": uploaded_file}
            data = {"selected_app": selected_app}
            response = requests.post("http://localhost:8001/profile/generate_profile", files=files, data=data)

            # Parse JSON response
            response_data = response.json()
            o_filename = response_data["filename"]
            file_content = response_data["file_content"]

            # Display output filename
            st.markdown("### Output Filename")
            st.write(o_filename)

            # Display output file content
            st.markdown("### Output File Content")
            st.code(file_content, language="text")

        elif file_url:
            # Send the URL to FastAPI
            #files = {"file_url": file_url}
            data = {"selected_app": selected_app, "file_url": file_url, "options": options}
            response = requests.post("http://localhost:8001/profile/generate_profile", data=data)

            # Parse JSON response
            response_data = response.json()
            o_filename = response_data["filename"]
            file_content = response_data["file_content"]

            # Display output filename
            st.markdown("### Output Filename")
            st.write(o_filename)

            # Display output file content
            st.markdown("### Output File Content")
            st.code(file_content, language="text")

        else:
            st.error("Please upload a file or enter a URL.")
            response = None

        # Show response from FastAPI
        if response and response.status_code == 200:
            st.success(response.json()["message"])
        elif response:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

        # Check if a file was uploaded
        #if uploaded_file is not None:
        #    # Send request to FastAPI to run selected app with uploaded file
        #    # Read the uploaded file
        #    file_content = uploaded_file.read()
        #    filename = uploaded_file.name
        #    st.success(f"Uploaded: {filename}")
        #    files = {"file": uploaded_file}
        #    data = {"selected_app": selected_app}
        #    response = requests.post("http://localhost:8000/profile/generate_profile", files=files, data=data)

        #elif file_url:
        #    try:
        #        # Fetch file from the URL
        #       response = requests.get(file_url)
        #        response.raise_for_status()  # Check for errors
        #        file_content = response.content  # Read file content
        #        filename = file_url.split("/")[-1] or "downloaded_file"
        #        st.success(f"Fetched file: {filename}")
        #        # Convert content to a file-like object
        #        file_obj = io.BytesIO(response.content)

        #        # Return as UploadFile
        #        uploaded_file = UploadFile(filename=filename, file=file_obj)
        #        files = {"file": uploaded_file}
        #        data = {"selected_app": selected_app}
        #        response = requests.post("http://localhost:8000/profile/generate_profile", files=files, data=data)
        #    except requests.RequestException as e:
        #        st.error(f"Error fetching file: {e}")

        #else:
        #    st.error("Please upload a file or give a url before submitting.")

        #files = {"file": uploaded_file}
        #data = {"selected_app": selected_app}
        #response = requests.post("http://localhost:8000/profile/generate_profile", files=files, data=data)

        # Parse JSON response
        #response_data = response.json()
        #filename = response_data["filename"]
        #file_content = response_data["file_content"]

        # Display output filename
        #st.markdown("### Output Filename")
        #st.write(filename)

        # Display output file content
        #st.markdown("### Output File Content")
        #st.code(file_content, language="text")

if __name__ == "__main__":
    main()
