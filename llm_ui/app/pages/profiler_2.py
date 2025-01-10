import streamlit as st
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
                }, f)

