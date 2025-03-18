import sys
import os
import streamlit as st
import json
import requests

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Add profiler directory to path
profiler_dir = os.path.join(project_root, 'profiler')
sys.path.append(profiler_dir)

# Import the LLM class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm import LLM

# Import the specific functions from profiler.py
from pages.profiler import handle_input, handle_schema, run_pipeline

def load_settings():
    """Load configuration parameters from settings.json."""
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'temperature': 0.3, 
            'max_tokens': 500,
            'similarity_k': 5,
            'model': 'llama3.1I-8b-q4',
            'use_openai': False,
            'profiler_options': {
                'field_info_to_compare': 'choices'
            }
        }

def show():
    """Main function to display the provider page"""
    st.title("Provider View - Dataset Profiling and LLM Interaction")

    # Initialize session state if not already done
    if 'processed_papers' not in st.session_state:
        st.session_state.processed_papers = {}

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}

    if 'json_text' not in st.session_state:
        st.session_state.json_text = {}

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Process Papers", "Chat with LLM"])

    with tab1:
        st.subheader("Upload a scientific paper (.xml) or provide a URL")

        uploaded_file = st.file_uploader("Choose an .xml file", type=["xml"], key="xml_uploader")
        xml_url = st.text_input("Or provide a URL to an .xml file")

        # Optional schema file upload
        st.subheader("Upload an optional schema file (JSON)")
        schema_file = st.file_uploader("Choose a JSON schema file", type=["json"], key="json_uploader")
        
        # Model selection for processing
        processing_model = st.radio(
            "Select model for paper processing:",
            ["OpenAI GPT-4o (faster, more accurate)", "Local Llama (slower, no API costs)"],
            horizontal=True,
            index=1  # Default to local model
        )

        # Run pipeline button
        if st.button("Run pipeline"):
            with st.spinner("Running the pipeline... Please wait."):
                xml_path = handle_input(uploaded_file, xml_url)

                if xml_path:
                    schema_path = handle_schema(schema_file)

                    # Load settings
                    settings = load_settings()

                    # Also load the raw paper text to preserve it
                    try:
                        from profiler.dataset_loader import load_paper_text_from_file_path
                        raw_paper_text = load_paper_text_from_file_path(xml_path)
                    except Exception as e:
                        st.warning(f"Could not load raw paper text: {str(e)}")
                        raw_paper_text = "Raw text not available"

                    # Choose the model based on UI selection
                    if processing_model.startswith("OpenAI"):
                        ff_model = "4o"
                        st.info(f"Using OpenAI GPT-4o for paper processing")
                    else:
                        ff_model = "llama3.1I-8b-q4"
                        st.info(f"Using local Llama model for paper processing")
                    
                    # Get settings from configuration
                    similarity_k = settings.get('similarity_k', 5)
                    field_info_to_compare = settings.get('profiler_options', {}).get('field_info_to_compare', 'choices')
                    
                    # Pass the model identifier and settings to run_pipeline
                    output_json = run_pipeline(
                        xml_path, 
                        schema_path, 
                        ff_model=ff_model
                    )
                    
                    if output_json:
                        st.subheader("Pipeline Output")
                        
                        # Store the JSON text in session state
                        if 'current_paper_id' not in st.session_state:
                            st.session_state.current_paper_id = 1
                        else:
                            st.session_state.current_paper_id += 1
                        
                        paper_id = st.session_state.current_paper_id
                        
                        # Format the JSON for display
                        formatted_json = json.dumps(output_json, indent=4)
                        st.session_state.json_text[paper_id] = formatted_json
                        
                        # Show the JSON in a text area
                        json_text_area = st.text_area(
                            "Edit the JSON output:", 
                            st.session_state.json_text[paper_id], 
                            height=300,
                            key=f"json_area_{paper_id}"
                        )
                        
                        # Update the stored JSON text when edited
                        st.session_state.json_text[paper_id] = json_text_area
                        
                        # Determine paper title
                        paper_title = f"Paper {paper_id}"
                        if uploaded_file:
                            paper_title = uploaded_file.name
                        elif xml_url:
                            paper_title = xml_url.split('/')[-1]
                        
                        # Extract the paper data and context
                        paper_data = output_json
                        context_info = {}
                        
                        # If the structure is like {"0": {"filled_form": {...}, "context": {...}}}, extract properly
                        if "0" in output_json:
                            if "filled_form" in output_json["0"]:
                                paper_data = output_json["0"]["filled_form"]
                            if "context" in output_json["0"]:
                                context_info = output_json["0"]["context"]
                        
                        # Store everything in session state
                        st.session_state.processed_papers[paper_id] = {
                            "title": paper_title,
                            "data": paper_data,
                            "context": context_info,
                            "raw_text": raw_paper_text,
                            "source": xml_path,
                            "full_output": output_json
                        }
                        
                        # Success message
                        st.success(f"Paper processed successfully! You can now chat with the LLM about this paper.")
                else:
                    st.error("Failed to process the input. Please check your file or URL.")
        
        # Handle save buttons for each paper after initial processing
        for paper_id in st.session_state.json_text:
            if st.button(f"Save Updated JSON for Paper {paper_id}", key=f"save_btn_{paper_id}"):
                try:
                    json_text = st.session_state.json_text[paper_id]
                    updated_json = json.loads(json_text)
                    
                    # Save to a file with paper ID in the filename
                    output_path = f"updated_output_{paper_id}.json"
                    with open(output_path, "w") as f:
                        f.write(json_text)
                    
                    # Update the stored version
                    paper_data = updated_json
                    context_info = {}
                    
                    # If the structure is like {"0": {"filled_form": {...}, "context": {...}}}, extract properly
                    if "0" in updated_json:
                        if "filled_form" in updated_json["0"]:
                            paper_data = updated_json["0"]["filled_form"]
                        if "context" in updated_json["0"]:
                            context_info = updated_json["0"]["context"]
                    
                    st.session_state.processed_papers[paper_id]["data"] = paper_data
                    st.session_state.processed_papers[paper_id]["context"] = context_info
                    st.session_state.processed_papers[paper_id]["full_output"] = updated_json
                    
                    st.success(f"Updated JSON saved to {output_path}!")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check your edits.")

        # Display list of processed papers
        if st.session_state.processed_papers:
            st.subheader("Processed Papers")
            for paper_id, paper in st.session_state.processed_papers.items():
                st.write(f"- {paper['title']} (ID: {paper_id})")

    with tab2:
        st.subheader("Chat with LLM about Medical Papers")
        
        # Initialize LLM
        if 'llm' not in st.session_state:
            try:
                with st.spinner("Initializing LLM... This may take a moment."):
                    st.session_state.llm = LLM()
                st.success("LLM initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing LLM: {str(e)}")
        
        # Check if we have processed papers
        if not st.session_state.processed_papers:
            st.warning("No processed papers available. Please process papers in the first tab.")
        else:
            # Paper selection options
            paper_options = {"All processed papers": "all"}
            paper_options.update({f"{paper['title']} (ID: {paper_id})": paper_id 
                                for paper_id, paper in st.session_state.processed_papers.items()})
            
            selected_paper_option = st.selectbox("Select papers to discuss:", 
                                                options=list(paper_options.keys()),
                                                index=0)
            
            selected_paper_id = paper_options[selected_paper_option]
            
            # Setup chat history for the selection
            if selected_paper_id not in st.session_state.chat_history:
                st.session_state.chat_history[selected_paper_id] = []
            
            # View options
            view_options = st.radio(
                "Information to view:",
                ["Processed Data", "Context Information", "Raw Paper Text", "All Information"],
                horizontal=True
            )
                
            # Display paper information based on selection
            if selected_paper_id != "all":
                paper_data = st.session_state.processed_papers[selected_paper_id]
                
                if view_options == "Processed Data":
                    with st.expander("Paper processed data", expanded=True):
                        st.json(paper_data["data"])
                elif view_options == "Context Information":
                    with st.expander("Context information", expanded=True):
                        if "context" in paper_data and paper_data["context"]:
                            st.json(paper_data["context"])
                        else:
                            st.info("No context information available for this paper.")
                elif view_options == "Raw Paper Text":
                    with st.expander("Raw paper text", expanded=True):
                        if "raw_text" in paper_data and paper_data["raw_text"]:
                            st.text_area("Full paper text:", paper_data["raw_text"], height=400)
                        else:
                            st.info("Raw text not available for this paper.")
                else:  # All Information
                    with st.expander("All paper information", expanded=True):
                        st.subheader("Processed Data")
                        st.json(paper_data["data"])
                        st.subheader("Context Information")
                        if "context" in paper_data and paper_data["context"]:
                            st.json(paper_data["context"])
                        else:
                            st.info("No context information available for this paper.")
                        st.subheader("Raw Paper Text")
                        if "raw_text" in paper_data and paper_data["raw_text"]:
                            st.text_area("Full paper text:", paper_data["raw_text"], height=300)
                        else:
                            st.info("Raw text not available for this paper.")
            else:
                with st.expander("All processed papers", expanded=False):
                    for paper_id, paper in st.session_state.processed_papers.items():
                        st.subheader(paper["title"])
                        if view_options == "Processed Data":
                            st.json(paper["data"])
                        elif view_options == "Context Information":
                            if "context" in paper and paper["context"]:
                                st.json(paper["context"])
                            else:
                                st.info("No context information available for this paper.")
                        elif view_options == "Raw Paper Text":
                            if "raw_text" in paper and paper["raw_text"]:
                                st.text_area(f"Full text of {paper['title']}", paper["raw_text"], height=200)
                            else:
                                st.info("Raw text not available for this paper.")
                        else:  # All Information
                            st.subheader("Processed Data")
                            st.json(paper["data"])
                            st.subheader("Context Information")
                            if "context" in paper and paper["context"]:
                                st.json(paper["context"])
                            else:
                                st.info("No context information available for this paper.")
                            st.subheader("Raw Paper Text")
                            if "raw_text" in paper and paper["raw_text"]:
                                st.text_area(f"Full text of {paper['title']}", paper["raw_text"], height=200)
                            else:
                                st.info("Raw text not available for this paper.")
                        st.divider()
            
            # LLM context options
            st.subheader("LLM Query Options")
            context_options = st.multiselect(
                "Include in LLM context:",
                ["Processed Data", "Context Information", "Raw Paper Text"],
                default=["Processed Data"]
            )
            
            # Display chat history
            st.subheader("Chat History")
            chat_container = st.container(height=400)
            with chat_container:
                for message in st.session_state.chat_history[selected_paper_id]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            
            # Chat input
            user_query = st.chat_input("Ask about the paper(s)...")
            
            if user_query:
                # Add user message to chat history
                st.session_state.chat_history[selected_paper_id].append({
                    "role": "user",
                    "content": user_query
                })
                
                # Display user message in chat history
                with chat_container:
                    with st.chat_message("user"):
                        st.write(user_query)
                
                # Create context from paper data based on selected options
                context_parts = []

                if selected_paper_id == "all":
                    # Handling multiple papers
                    for paper_id, paper in st.session_state.processed_papers.items():
                        paper_context = f"--- PAPER {paper_id}: {paper['title']} ---\n"
                        
                        if "Processed Data" in context_options:
                            paper_context += f"\nPROCESSED DATA:\n{json.dumps(paper['data'], indent=2)}"
                        
                        if "Context Information" in context_options and "context" in paper and paper["context"]:
                            paper_context += f"\nCONTEXT INFORMATION:\n{json.dumps(paper['context'], indent=2)}"
                        
                        if "Raw Paper Text" in context_options and "raw_text" in paper and paper["raw_text"]:
                            # Truncate if too long to fit in context window
                            raw_text = paper["raw_text"]
                            if len(raw_text) > 10000:
                                raw_text = raw_text[:10000] + "... [text truncated]"
                            paper_context += f"\nRAW PAPER TEXT:\n{raw_text}"
                        
                        context_parts.append(paper_context)
                    
                    full_context = "\n\n".join(context_parts)
                    context_intro = "Below is the information from multiple papers:"
                    
                else:
                    # Single paper
                    paper = st.session_state.processed_papers[selected_paper_id]
                    
                    if "Processed Data" in context_options:
                        context_parts.append(f"PROCESSED DATA:\n{json.dumps(paper['data'], indent=2)}")
                    
                    if "Context Information" in context_options and "context" in paper and paper["context"]:
                        context_parts.append(f"CONTEXT INFORMATION:\n{json.dumps(paper['context'], indent=2)}")
                    
                    if "Raw Paper Text" in context_options and "raw_text" in paper and paper["raw_text"]:
                        # Truncate if too long to fit in context window
                        raw_text = paper["raw_text"]
                        if len(raw_text) > 10000:
                            raw_text = raw_text[:10000] + "... [text truncated]"
                        context_parts.append(f"RAW PAPER TEXT:\n{raw_text}")
                    
                    full_context = "\n\n".join(context_parts)
                    context_intro = "Below is the information from the paper:"

                # Create prompt for LLM
                prompt = f"""You are a medical research assistant helping to analyze scientific papers.
                You are discussing {'multiple papers' if selected_paper_id == 'all' else 'a paper'}.
                {context_intro}

                {full_context}

                Based on this information, please answer the following question:
                {user_query}

                If the answer cannot be determined from the provided information, say so clearly.
                """

                # Get response from LLM
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                response = st.session_state.llm.ask(prompt, max_tokens=800, temperature=0.3)
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