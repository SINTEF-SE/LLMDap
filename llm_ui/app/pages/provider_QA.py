import streamlit as st
import json
import sys
import os
from llm_ui.app.llm import LLM

def show():
    st.title("Chat with LLM about Medical Papers")
    
    # Initialize LLM
    if 'llm' not in st.session_state:
        try:
            st.session_state.llm = LLM()
            st.success("LLM loaded successfully!")
        except Exception as e:
            st.error(f"Error loading LLM: {str(e)}")
            return
    
    # Check if we have processed papers
    if 'processed_papers' not in st.session_state or not st.session_state.processed_papers:
        st.warning("No processed papers available. Please go to the Provider page to process papers first.")
        return
    
    # Paper selection
    paper_options = {f"{paper['title']} (ID: {paper_id})": paper_id 
                     for paper_id, paper in st.session_state.processed_papers.items()}
    
    selected_paper_name = st.selectbox("Select a paper to discuss:", 
                                       options=list(paper_options.keys()),
                                       index=0)
    
    selected_paper_id = paper_options[selected_paper_name]
    paper_data = st.session_state.processed_papers[selected_paper_id]
    
    # Display paper information
    with st.expander("Paper details", expanded=False):
        st.json(paper_data["data"])
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    
    if selected_paper_id not in st.session_state.chat_history:
        st.session_state.chat_history[selected_paper_id] = []
    
    # Display chat history
    for message in st.session_state.chat_history[selected_paper_id]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_query = st.chat_input("Ask about this paper...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history[selected_paper_id].append({
            "role": "user",
            "content": user_query
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Create context from paper data
        paper_context = json.dumps(paper_data["data"], indent=2)
        
        # Create prompt for LLM
        prompt = f"""You are a medical research assistant helping to analyze scientific papers.
You are discussing a paper that has been processed into structured data.
Below is the structured data from the paper:

{paper_context}

Based on this data, please answer the following question:
{user_query}

If the answer cannot be determined from the provided data, say so clearly.
"""
        
        # Get response from LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm.ask(prompt, max_tokens=500, temperature=0.3)
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history[selected_paper_id].append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    
                    # Add error to chat history
                    st.session_state.chat_history[selected_paper_id].append({
                        "role": "assistant",
                        "content": error_message
                    })
    
    # Add option to clear chat history
    if st.button("Clear chat history"):
        st.session_state.chat_history[selected_paper_id] = []
        st.rerun()