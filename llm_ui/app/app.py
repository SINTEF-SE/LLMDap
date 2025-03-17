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

# Import pages with relative imports
from pages import home, configure, profiler, provider

def main():
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select Page", ["Home", "Configure", "Profiler", "Provider"])

    if page == "Home":
        home.show()
    elif page == "Configure":
        configure.show()
    elif page == "Profiler":
        profiler.show()
    elif page == "Provider":
        provider.show()

if __name__ == "__main__":
    main()

