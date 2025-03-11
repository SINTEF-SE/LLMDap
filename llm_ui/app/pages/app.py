import streamlit as st
from llm_ui.app.pages import consumer_QA
st.set_page_config(page_title="UPCAST Profiler", page_icon=":bar_chart:", layout="wide")

from pages import home, configure, profiler, provider, consumer_QA

def main():
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select Page", ["Home", "Configure", "Profiler", "Provider", "Consumer", "Q&A with LLM"])

    if page == "Home":
        home.show()
    elif page == "Configure":
        configure.show()
    elif page == "Profiler":
        profiler.show()
    elif page == "Provider":
        provider.show()
    elif page == "Consumer":
        consumer_QA.show()
    elif page == "Q&A with LLM":
        consumer_QA.show()

if __name__ == "__main__":
    main()

