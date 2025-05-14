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

# Stabilize the sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 200px;
        max-width: 300px;
    }
</style>
""", unsafe_allow_html=True)

# IMPORTANT: Print current directory and check file existence
print(f"Current directory: {os.getcwd()}")
print(f"Pages directory content: {os.listdir(os.path.join(os.path.dirname(__file__), 'pages'))}")

# using consistent imports for all pages
from llm_ui.app.pages import home, configure, profiler, provider, datasets, consumer_QA

def main():
    # Initialize session state variables
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = "Home"
    
    # Create a consistent container for the sidebar
    sidebar = st.sidebar.container()
    
    with sidebar:
        st.title("Navigation")
        
        # debug information
        with st.expander("Debug Info", expanded=False):
            st.write("Session state keys:", list(st.session_state.keys()))
            if 'show_page' in st.session_state:
                st.write("Pending page switch:", st.session_state.show_page)
            if 'selected_datasets' in st.session_state:
                st.write("Selected datasets:", len(st.session_state.selected_datasets))
            if 'nav_page' in st.session_state:
                st.write("Current sidebar selection:", st.session_state.nav_page)
        
        # Define available pages 
        pages = {
            "Home": home.show,
            "Dataset Browser": datasets.show,  
            "Consumer Q&A": consumer_QA.show,  
            "Provider": provider.show,
            "Profiler": profiler.show,
            "Configure": configure.show
        }
        
        # Check if we need to navigate to a new page first
        if 'show_page' in st.session_state:
            requested_page = st.session_state.show_page
            if requested_page in pages:
                # Update the nav_page immediately
                st.session_state.nav_page = requested_page
                # Clear the temporary navigation state
                del st.session_state.show_page
                # Force rerun to refresh the UI with the new page
                st.rerun()
        
        # Default to Home if no page is selected
        default_page = "Home"
        
        # Use the saved selection if available 
        if 'nav_page' in st.session_state and st.session_state.nav_page in pages:
            default_page = st.session_state.nav_page
        
        # Get the page index for the selectbox
        page_options = list(pages.keys())
        try:
            default_index = page_options.index(default_page)
        except (ValueError, KeyError):
            default_index = 0
        
        # Create the sidebar selectbox 
        current_page = st.selectbox(
            "Select Page", 
            page_options,
            index=default_index,
            key="page_selectbox"
        )
        
        # Check if the selection has changed
        if current_page != st.session_state.nav_page:
            st.session_state.nav_page = current_page
            st.rerun()
    
    # Display the current page
    pages[st.session_state.nav_page]()

if __name__ == "__main__":
    print("Project root:", project_root)
    print("Profiler directory:", profiler_dir)
    main()

