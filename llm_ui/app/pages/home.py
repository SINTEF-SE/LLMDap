import streamlit as st

def show():
    st.title("Welcome to the LLM Data Profiler & Interaction App")

    st.markdown("""
    This application leverages Large Language Models (LLMs) to help you interact with, manage, and profile your datasets.
    You can perform various tasks such as:

    *   **Managing Datasets:** Upload, view, and manage different datasets.
    *   **Configuring Settings:** Adjust application parameters and configurations.
    *   **Interacting as Consumer/Provider:** Engage with the system through dedicated interfaces for different user roles (Consumer QA, Provider).
    *   **Profiling Data:** Run profiling tasks to gain insights into your data using LLMs.

    **Limitations:**
    *   LLM performance can vary based on the complexity of the data and the specific task.
    *   The accuracy of profiling depends on the quality of the input data and the underlying model's capabilities.
    *   Ensure your configurations are set correctly for optimal results.

    Use the sidebar or the buttons below to navigate to different sections of the application.
    """)

    st.subheader("Navigate to:")

    # Define page names as they appear in app.py's 'pages' dictionary
    page_map = {
        "Manage Datasets": "Dataset Browser",
        "Consumer QA": "Consumer Q&A",
        "Configure Settings": "Configure",
        "Provider Interaction": "Provider",
        "Run Profiler": "Profiler"
    }

    col1, col2, col3 = st.columns(3)

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

    with col3:
        if st.button("Run Profiler"):
            st.session_state.show_page = page_map["Run Profiler"]
            st.rerun()
