import streamlit as st
from pages import home_0, configure_1, profiler_2, QA_3

def main():
    st.title("UPCAST Profiler")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select Page", ["Home", "Configure", "Profiler", "Q&A with LLM"])

    if page == "Home":
        home_0.show()
    elif page == "Configure":
        configure_1.show()
    elif page == "Profiler":
        profiler_2.show()
    elif page == "Q&A with LLM":
        QA_3.show()

if __name__ == "__main__":
    main()
