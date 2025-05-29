Overview
LLM-UI is a Streamlit-based web application for interacting with biomedical research datasets from ArrayExpress/BioStudies using Large Language Models (LLMs). It allows researchers to browse datasets, ask questions about them, process scientific papers, and extract metadata in a structured way.

Features
Dataset Browser: View, search, and select datasets stored in a local SQLite database
Consumer Q&A: Ask natural language questions about selected datasets and get AI-powered answers
Provider Interface: Upload research papers (XML, PDF) or fetch them via PubMed ID/DOI and extract structured metadata
Profiler: Test XML file analysis functionality in a standalone interface
Configuration: Customize LLM behavior (temperature, max tokens, prompt template) and choose between local models or OpenAI API

Installation:
Prerequisites
Python 3.8+
SQLite3
Access to either OpenAI API (optional) or local LLM models
Check the README file in the main folder to see requierments for the profiler and other 

setup
1. clone the repository: 
git clone https://github.com/Shang1607/bacholer-test.git
cd <repository-directory>

2. install dependencies: 
pip install -r requirements.txt ( the the frontend folder.)

3. set up OpenAI API key ( if using openAI) 
# Create a file in llm_ui/app/openai_key.py
echo 'API_KEY = "your-openai-api-key"' > llm_ui/app/openai_key.py 
note: this is not the most secure way and should not be pushed to a local git repo. 

4. Prepare the user_datasets directory:
mkdir -p llm_ui/app/user_datasets


Running the Application
Navigate to the app directory and run:
streamlit run app.py


Application Structure
Pages
Home (home.py)

Landing page with navigation options and application overview
Dataset Browser (datasets.py)

View all datasets in the database
Search datasets by title, description, organism, etc.
Select datasets for Q&A
Rescan directories to update the database

Consumer Q&A (consumer_QA.py)
Ask questions about selected datasets
Get AI-generated answers based on dataset metadata
View chat history and manage conversations

Provider (provider.py)
Upload scientific papers (XML/PDF)
Enter PubMed IDs or DOIs to fetch papers automatically
Process papers to extract structured metadata
Save processed data to the database

Configure (configure.py)
Select between OpenAI API or local models
Adjust LLM parameters (temperature, max tokens)
Customize prompt templates
Configure profiler settings (similarity_k, field_info_to_compare)


Core Components
app.py: Main application entry point, handles page navigation
llm.py: LLM interface that manages communication with OpenAI API or local models
db_utils.py: Database utilities for managing the SQLite database
user_datasets/: Directory for storing processed user-uploaded datasets


Usage Workflow
Start the application using streamlit run app.py

Browse and select datasets:
Navigate to "Dataset Browser"
Use the search function to find relevant datasets
Select datasets using the checkboxes
Click "Continue with Selected" to proceed to Q&A

Ask questions about datasets:
Navigate to "Consumer Q&A"
Type your question in the text area
Click "Send" to get an AI-generated answer
View chat history and previous questions/answers

Process new papers:
Navigate to "Provider"
Upload a paper file or enter a PubMed ID/DOI
Select a schema for processing
Click "Process Input"
Review and edit the extracted data
Save to the database or chat with the LLM about the paper

Configure settings:
Navigate to "Configure"
Adjust LLM settings (model type, temperature, max tokens)
Customize prompt templates
Save changes