import streamlit as st

def show():
    st.title("Welcome to the LLM Data Profiler & Interaction App")

    st.markdown("""
    This application leverages Large Language Models (LLMs) to help you interact with, manage, and profile your medical datasets.
    How to use this application:

    *   **Managing Datasets:** Upload, view, and manage different datasets.
    *   **Configuring Settings:** Adjust application parameters and configurations for how the LLM responds.
    *   **Interacting as Consumer/Provider:** Use the LLM to ask questions about datasets and uploade papers through the provider page to save it in the database.
    *   

    **Limitations:**
    *   LLM performance can vary based on the complexity of the data and the specific task.
    *   The accuracy of profiling depends on the quality of the input data and the underlying model's capabilities.
    *   Ensure your configurations are set correctly for optimal results.
    *   The search engine is able to serach for Name, PMID, Organism, study type, and ID. 
    *   The provider page is able to upload XML files, PDF files (slightly more clunky), pubmed links and some DOIs (50% accuracy as of now).  
                

    Use the sidebar or the buttons below to navigate to different sections of the application.
    """)

    st.subheader("Navigate to:")

    # Define page names
    page_map = {
        "Manage Datasets": "Dataset Browser",
        "Consumer QA": "Consumer Q&A",
        "Configure Settings": "Configure",
        "Provider Interaction": "Provider",
        
    }

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Manage Datasets"):
            st.session_state.show_page = page_map["Manage Datasets"]
            st.rerun()
        if st.button("Consumer QA"):
             st.session_state.show_page = page_map["Consumer QA"]
             st.rerun()

    with col2:
        if st.button("Configure Settings"):
            st.session_state.show_page = page_map["Configure Settings"]
            st.rerun()
        if st.button("Provider Interaction"):
            st.session_state.show_page = page_map["Provider Interaction"]
            st.rerun()

    
