import streamlit as st
import sys
import os

# Add the project root to the path so imports work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Also add the profiler directory specifically to fix the load_modules import
profiler_dir = os.path.join(project_root, 'profiler')
sys.path.append(profiler_dir)

# Configure Streamlit page
st.set_page_config(page_title="UPCAST Profiler", page_icon=":bar_chart:", layout="wide")

# Your homepage content here
st.title("UPCAST Profiler")
st.write("Welcome to the UPCAST Profiler! Use the sidebar to navigate to different pages.")

# Import pages with relative imports
from pages import home, configure, profiler, provider

# Check if consumer_QA.py exists and import it if present
consumer_qa_exists = os.path.exists(os.path.join(os.path.dirname(__file__), 'pages', 'consumer_QA.py'))
if consumer_qa_exists:
    from pages import consumer_QA

def main():
    # Sidebar for navigation
    pages = ["Home", "Configure", "Profiler", "Provider"]
    
    # Add Consumer QA to navigation if it exists
    if consumer_qa_exists:
        pages.append("Consumer Q&A")
    
    page = st.sidebar.selectbox("Select Page", pages)

    if page == "Home":
        home.show()
    elif page == "Configure":
        configure.show()
    elif page == "Profiler":
        profiler.show()
    elif page == "Provider":
        provider.show()
    elif page == "Consumer Q&A" and consumer_qa_exists:
        consumer_QA.show()

if __name__ == "__main__":
    main()

