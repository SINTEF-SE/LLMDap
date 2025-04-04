import sys
import os
import streamlit as st
import json
import requests
import re

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Add profiler directory to path
profiler_dir = os.path.join(project_root, 'profiler')
sys.path.append(profiler_dir)

# Import the LLM class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm import LLM

# Import the specific functions from profiler.py and db_utils
from pages.profiler import handle_input, handle_schema
from db_utils import init_db, upsert_dataset # Import DB functions

try:
    from profiler.run_inference import call_inference
except ImportError:
    st.error("Could not import call_inference function. Please check your project structure.")
    call_inference = None

def load_settings():
    """Load configuration parameters from settings.json."""
    try:
        # Ensure the path is relative to the project root
        settings_path = os.path.join(project_root, 'settings.json')
        with open(settings_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("settings.json not found, using default settings.")
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
    except json.JSONDecodeError:
        st.error("Error decoding settings.json, using default settings.")
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

# Define run_pipeline function locally as it was previously imported from profiler page
def run_pipeline(xml_path, schema_path, ff_model="llama3.1I-8b-q4", similarity_k=5, field_info_to_compare="choices"):
    """Run the profiling pipeline with better text processing"""
    if call_inference is None:
        st.error("call_inference function could not be imported")
        return None

    try:
        # Determine the schema to use
        schema_to_use = None
        if schema_path: # If a custom schema file path is provided
             if isinstance(schema_path, str) and os.path.exists(schema_path):
                 # Load schema from file - Requires converting JSON schema to Pydantic model
                 # This part needs implementation if custom JSON schemas are to be supported fully
                 st.error("Loading schema from custom JSON file is not fully implemented yet.")
                 # As a placeholder, try to load a default schema if available
                 try:
                     from profiler.metadata_schemas import arxpr2_schema # Example default
                     schema_to_use = arxpr2_schema.Metadata_form
                     st.warning("Using default ARXPR2 schema as custom schema loading is not implemented.")
                 except ImportError:
                     st.error("Could not load default schema.")
                     return None
             elif hasattr(schema_path, '__fields__'): # Check if it's already a Pydantic model class
                 schema_to_use = schema_path
             else:
                 st.error("Invalid schema provided.")
                 return None
        else:
             # Use a default schema if none provided
             try:
                 from profiler.metadata_schemas import arxpr2_schema # Example default
                 schema_to_use = arxpr2_schema.Metadata_form
                 st.info("No custom schema provided, using default ARXPR2 schema.")
             except ImportError:
                 st.error("Could not load default schema.")
                 return None

        if schema_to_use is None:
             st.error("Schema could not be determined.")
             return None

        # Run the inference with the specified model and schema
        st.info(f"Using {ff_model} for paper analysis with schema: {schema_to_use.__name__}")

        # Call the inference function with proper parameters
        output = call_inference(
            schema=schema_to_use,
            paper_path=xml_path,
            similarity_k=similarity_k,
            field_info_to_compare=field_info_to_compare,
            ff_model=ff_model
        )

        # Check and clean up the output - this helps fix garbled context
        if output and len(output) > 0:
            # Check if we need to clean context sections
            for key in output:
                if isinstance(output[key], dict) and "context" in output[key]:
                    context = output[key]["context"]
                    # Clean each field in the context
                    for field in context:
                        if isinstance(context[field], str):
                            # Add spaces between words (simple heuristic)
                            cleaned_text = context[field]
                            # Replace "convertedFont" markers which appear frequently
                            cleaned_text = cleaned_text.replace("convertedFont", " ")
                            # Add space after periods and commas if not followed by space
                            cleaned_text = re.sub(r'\.(?! )', '. ', cleaned_text)
                            cleaned_text = re.sub(r',(?! )', ', ', cleaned_text)
                            # Clean up multiple spaces
                            cleaned_text = re.sub(r' +', ' ', cleaned_text)
                            context[field] = cleaned_text
            return output
        else:
            st.error("No output generated from pipeline")
            return None

    except Exception as e:
        st.error(f"Error running pipeline: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def show():
    """Main function to display the provider page"""
    st.title("Provider View - Dataset Profiling and LLM Interaction")

    # Initialize database on first run
    init_db()

    # Initialize session state if not already done
    if 'processed_papers' not in st.session_state:
        st.session_state.processed_papers = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    if 'json_text' not in st.session_state:
        st.session_state.json_text = {}
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    if 'current_paper_id' not in st.session_state:
         st.session_state.current_paper_id = 0 # Start IDs from 0 or 1 consistently

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Process Papers", "Chat with LLM"])

    with tab1:
        st.subheader("Process a Scientific Paper")

        # Add a nice description
        st.markdown("*Upload an XML paper file or provide a URL to analyze the dataset described in the paper.*")

        # Create a visually distinct upload area
        with st.container(border=True):
            st.markdown("##### Upload Paper")
            uploaded_file = st.file_uploader("Choose an .xml file", type=["xml"], key="xml_uploader")

            # Add a divider between upload and URL
            if not uploaded_file:
                st.markdown("*OR*")
                xml_url = st.text_input("Provide a URL to an .xml file")
            else:
                xml_url = None # Clear URL if file is uploaded

        # Make the schema upload section more distinct
        with st.container(border=True):
            st.markdown("##### Schema Configuration")
            st.markdown("*Optional: Upload a custom schema file (JSON) or select a predefined one.*")
            schema_file = st.file_uploader("Upload JSON schema", type=["json"], key="json_uploader")
            # Add option for predefined schemas later if needed

        # Make the model selection more visually distinct
        with st.container(border=True):
            st.markdown("##### Model Selection")
            processing_model = st.radio(
                "Select model for paper processing:",
                ["OpenAI GPT-4o (faster, more accurate)", "Local Llama (slower, no API costs)"],
                horizontal=True,
                index=1  # Default to local model
            )

        # Add a clearer button
        st.markdown("---")
        if st.button("▶️ Run Pipeline", use_container_width=True):
            with st.spinner("Running the pipeline... Please wait."):
                xml_path = handle_input(uploaded_file, xml_url)

                if xml_path:
                    schema_path_or_obj = handle_schema(schema_file) # This might return path or Pydantic model

                    # Load settings
                    settings = load_settings()

                    # Load the raw paper text
                    raw_paper_text = "Raw text not available"
                    try:
                        from profiler.dataset_loader import load_paper_text_from_file_path
                        raw_paper_text = load_paper_text_from_file_path(xml_path)
                    except Exception as e:
                        st.warning(f"Could not load raw paper text: {str(e)}")

                    # Choose the model based on UI selection
                    ff_model = "4o" if processing_model.startswith("OpenAI") else "llama3.1I-8b-q4"

                    # Get settings from configuration
                    similarity_k = settings.get('similarity_k', 5)
                    field_info_to_compare = settings.get('profiler_options', {}).get('field_info_to_compare', 'choices')

                    # Pass the model identifier and settings to run_pipeline
                    output_json = run_pipeline(
                        xml_path,
                        schema_path_or_obj, # Pass the schema path or object
                        ff_model=ff_model,
                        similarity_k=similarity_k,
                        field_info_to_compare=field_info_to_compare
                    )

                    if output_json:
                        st.subheader("Pipeline Output")

                        # Increment paper ID
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
                            # Extract filename from URL more robustly
                            try:
                                from urllib.parse import urlparse
                                parsed_url = urlparse(xml_url)
                                paper_title = os.path.basename(parsed_url.path) or f"URL_Paper_{paper_id}"
                            except:
                                paper_title = f"URL_Paper_{paper_id}"


                        # Extract the paper data and context
                        paper_data = output_json
                        context_info = {}

                        # If the structure is like {"0": {"filled_form": {...}, "context": {...}}}, extract properly
                        # Assuming call_inference now returns a dict where keys are paper identifiers (like "0" or maybe the pmid)
                        # Let's find the first key if it's nested like this
                        first_key = next(iter(output_json), None)
                        if first_key is not None and isinstance(output_json[first_key], dict):
                             nested_data = output_json[first_key]
                             if "filled_form" in nested_data:
                                 paper_data = nested_data["filled_form"]
                             if "context" in nested_data:
                                 context_info = nested_data["context"]
                        # If it's not nested, paper_data remains output_json

                        # Store everything in session state
                        st.session_state.processed_papers[paper_id] = {
                            "title": paper_title,
                            "data": paper_data, # Store the potentially nested 'filled_form' data
                            "context": context_info,
                            "raw_text": raw_paper_text,
                            "source": xml_path,
                            "full_output": output_json # Store the raw output from run_pipeline
                        }

                        # Success message and action buttons in columns
                        st.success(f"Paper processed successfully!")

                        # Create three columns for the action buttons
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # Save raw output button
                            if st.button("💾 Save Full Output", key=f"save_full_btn_{paper_id}"):
                                try:
                                    output_path = f"full_output_{paper_id}.json" # Save in root dir for now
                                    with open(output_path, "w") as f:
                                        # Save the potentially edited JSON from the text area
                                        f.write(st.session_state.json_text[paper_id])
                                    st.success(f"Full output saved to {output_path}!")
                                except Exception as e:
                                    st.error(f"Error saving full output: {str(e)}")

                        with col2:
                             # Save to Datasets button
                            if st.button("➕ Save to My Datasets", key=f"save_dataset_btn_{paper_id}"):
                                try:
                                    # Define the path for user datasets
                                    user_datasets_dir = os.path.join(project_root, 'llm_ui', 'app', 'user_datasets')
                                    os.makedirs(user_datasets_dir, exist_ok=True)

                                    # Get the processed data (potentially edited by user)
                                    processed_data_json = json.loads(st.session_state.json_text[paper_id])

                                    # Extract the relevant part ('filled_form' if nested)
                                    first_key = next(iter(processed_data_json), None)
                                    if first_key is not None and isinstance(processed_data_json[first_key], dict) and "filled_form" in processed_data_json[first_key]:
                                         paper_data_to_save = processed_data_json[first_key]["filled_form"]
                                    else:
                                         paper_data_to_save = processed_data_json # Assume flat structure otherwise

                                    # Extract key metadata for the dataset browser view
                                    # Use .get() for safety, provide defaults
                                    # Try to get pmid/accession from the processed data first, then from the stored session state
                                    pmid = paper_data_to_save.get('pmid', st.session_state.processed_papers[paper_id].get('pmid'))
                                    if not pmid or pmid == 'unknown': pmid = f'provider_{paper_id}' # Fallback pmid

                                    accession = paper_data_to_save.get('accession', st.session_state.processed_papers[paper_id].get('accession'))
                                    if not accession or accession == 'Unknown': accession = f'USER-{pmid}' # Fallback accession

                                    dataset_for_browser = {
                                        'title': paper_data_to_save.get('title', st.session_state.processed_papers[paper_id].get('title', f"User Dataset {pmid}")),
                                        'accession': accession,
                                        'pmid': pmid,
                                        'description': paper_data_to_save.get('description', 'User-provided dataset processed via Provider page.'),
                                        'organism': paper_data_to_save.get('organism', 'Unknown'),
                                        'study_type': paper_data_to_save.get('study_type', 'Unknown'),
                                        'source': 'user_provider', # Mark as user-provided
                                        'original_file': st.session_state.processed_papers[paper_id].get('source', 'Unknown'),
                                        # Include other relevant fields if available in paper_data_to_save
                                        'hardware': paper_data_to_save.get('hardware'),
                                        'organism_part': paper_data_to_save.get('organism_part'),
                                        'experimental_designs': paper_data_to_save.get('experimental_designs'),
                                        'assay_by_molecule': paper_data_to_save.get('assay_by_molecule'),
                                        'technology': paper_data_to_save.get('technology'),
                                        'sample_count': paper_data_to_save.get('sample_count'),
                                        'release_date': paper_data_to_save.get('release_date'),
                                        'experimental_factors': paper_data_to_save.get('experimental_factors'),
                                        # Optionally store the full processed data within this file too
                                        # 'full_processed_data': paper_data_to_save
                                    }

                                    # Generate filename
                                    safe_filename = re.sub(r'[^\w\-_\. ]', '_', f"user_dataset_{accession}.json")
                                    save_path = os.path.join(user_datasets_dir, safe_filename)

                                    # Save the standardized JSON file
                                    with open(save_path, "w") as f:
                                        json.dump(dataset_for_browser, f, indent=4)

                                    # Also add/update the metadata in the SQLite database
                                    # Ensure file_path is included for the primary key
                                    dataset_for_browser['file_path'] = save_path
                                    print(f"[PROVIDER_PAGE] Attempting to upsert dataset to DB: {dataset_for_browser}") # Debug print
                                    upsert_dataset(dataset_for_browser)

                                    # Show full path in success message
                                    st.success(f"Processed data saved to My Datasets: {save_path}")
                                    st.info("Database index updated.") # Add DB feedback

                                    # Optionally clear the cache in datasets.py if it exists
                                    # Note: With DB, caching might be less necessary or handled differently
                                    cache_file = os.path.join(project_root, 'cached_datasets.json')
                                    if os.path.exists(cache_file):
                                         try:
                                             os.remove(cache_file)
                                             st.info("Cleared dataset cache. Refresh the Datasets page to see changes.")
                                         except OSError as e:
                                             st.warning(f"Could not clear dataset cache: {e}")

                                except json.JSONDecodeError:
                                     st.error("Could not save: Edited JSON is invalid.")
                                except Exception as e:
                                    st.error(f"Error saving dataset: {str(e)}")

                        with col3:
                            # Chat button that changes the active tab
                            if st.button("💬 Chat with LLM", key=f"chat_btn_{paper_id}"):
                                st.session_state.active_tab = 1
                                st.rerun()
                    else:
                        st.error("Failed to process the input. Please check your file or URL.")

        # Handle save buttons for each paper after initial processing (This seems redundant now?)
        # Commenting out as the save buttons are now shown immediately after processing.
        # for paper_id in st.session_state.json_text:
        #     if st.button(f"Save Updated JSON for Paper {paper_id}", key=f"save_btn_{paper_id}"):
        #         # ... (logic was here) ...

        # Display list of processed papers
        if st.session_state.processed_papers:
            st.subheader("Processed Papers History (Current Session)")
            for paper_id, paper in st.session_state.processed_papers.items():
                st.write(f"- {paper['title']} (ID: {paper_id})")

    with tab2:
        st.subheader("Chat with LLM about Medical Papers")

        # If we just came from tab1 via the Chat button, show a helpful message
        if st.session_state.active_tab == 1 and 'current_paper_id' in st.session_state:
            paper_id = st.session_state.current_paper_id
            if paper_id in st.session_state.processed_papers:
                paper_title = st.session_state.processed_papers[paper_id]["title"]
                st.info(f"Ready to chat about '{paper_title}'. Select the paper below and ask questions.")

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
            # Use paper_id as key for uniqueness
            paper_options.update({f"{paper['title']} (ID: {paper_id})": paper_id
                                for paper_id, paper in st.session_state.processed_papers.items()})

            selected_paper_option = st.selectbox("Select papers to discuss:",
                                                options=list(paper_options.keys()),
                                                index=0)

            selected_paper_id_key = paper_options[selected_paper_option] # This is the paper_id or 'all'

            # Setup chat history for the selection
            if selected_paper_id_key not in st.session_state.chat_history:
                st.session_state.chat_history[selected_paper_id_key] = []

            # View options
            view_options = st.radio(
                "Information to view:",
                ["Processed Data", "Context Information", "Raw Paper Text", "All Information"],
                horizontal=True
            )

            # Display paper information based on selection
            if selected_paper_id_key != "all":
                # Ensure the selected paper ID exists before accessing
                if selected_paper_id_key in st.session_state.processed_papers:
                    paper_info = st.session_state.processed_papers[selected_paper_id_key]

                    # Display paper information sections
                    if view_options == "Processed Data":
                        with st.expander("Paper processed data", expanded=True):
                            st.json(paper_info.get("data", {})) # Use .get for safety
                    elif view_options == "Context Information":
                        with st.expander("Context information", expanded=True):
                            if "context" in paper_info and paper_info["context"]:
                                st.json(paper_info["context"])
                            else:
                                st.info("No context information available for this paper.")
                    elif view_options == "Raw Paper Text":
                        with st.expander("Raw paper text", expanded=True):
                            if "raw_text" in paper_info and paper_info["raw_text"]:
                                st.text_area("Full paper text:", paper_info["raw_text"], height=400)
                            else:
                                st.info("Raw text not available for this paper.")
                    else:  # All Information
                        with st.expander("All paper information", expanded=True):
                            st.subheader("Processed Data")
                            st.json(paper_info.get("data", {}))
                            st.subheader("Context Information")
                            if "context" in paper_info and paper_info["context"]:
                                st.json(paper_info["context"])
                            else:
                                st.info("No context information available for this paper.")
                            st.subheader("Raw Paper Text")
                            if "raw_text" in paper_info and paper_info["raw_text"]:
                                st.text_area("Full paper text:", paper_info["raw_text"], height=300)
                            else:
                                st.info("Raw text not available for this paper.")
                else:
                    st.warning(f"Selected paper (ID: {selected_paper_id_key}) not found in processed list.")

            else: # "All processed papers" selected
                with st.expander("All processed papers", expanded=False):
                    for paper_id, paper in st.session_state.processed_papers.items():
                        st.subheader(paper["title"])
                        if view_options == "Processed Data":
                            st.json(paper.get("data", {}))
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
                            st.json(paper.get("data", {}))
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
                for message in st.session_state.chat_history[selected_paper_id_key]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

            # Chat input
            user_query = st.chat_input("Ask about the paper(s)...")

            if user_query:
                # Add user message to chat history
                st.session_state.chat_history[selected_paper_id_key].append({
                    "role": "user",
                    "content": user_query
                })

                # Display user message in chat history
                with chat_container:
                    with st.chat_message("user"):
                        st.write(user_query)

                # Create context from paper data based on selected options
                context_parts = []
                context_intro = ""

                if selected_paper_id_key == "all":
                    # Handling multiple papers
                    context_intro = "Below is the information from multiple papers:"
                    for paper_id, paper in st.session_state.processed_papers.items():
                        paper_context = f"--- PAPER {paper_id}: {paper['title']} ---\n"
                        if "Processed Data" in context_options:
                            paper_context += f"\nPROCESSED DATA:\n{json.dumps(paper.get('data', {}), indent=2)}"
                        if "Context Information" in context_options and "context" in paper and paper["context"]:
                            paper_context += f"\nCONTEXT INFORMATION:\n{json.dumps(paper['context'], indent=2)}"
                        if "Raw Paper Text" in context_options and "raw_text" in paper and paper["raw_text"]:
                            raw_text = paper["raw_text"]
                            if len(raw_text) > 10000: raw_text = raw_text[:10000] + "... [text truncated]"
                            paper_context += f"\nRAW PAPER TEXT:\n{raw_text}"
                        context_parts.append(paper_context)
                    full_context = "\n\n".join(context_parts)

                elif selected_paper_id_key in st.session_state.processed_papers:
                    # Single paper
                    paper = st.session_state.processed_papers[selected_paper_id_key]
                    context_intro = f"Below is the information from the paper '{paper['title']}':"
                    if "Processed Data" in context_options:
                        context_parts.append(f"PROCESSED DATA:\n{json.dumps(paper.get('data', {}), indent=2)}")
                    if "Context Information" in context_options and "context" in paper and paper["context"]:
                        context_parts.append(f"CONTEXT INFORMATION:\n{json.dumps(paper['context'], indent=2)}")
                    if "Raw Paper Text" in context_options and "raw_text" in paper and paper["raw_text"]:
                        raw_text = paper["raw_text"]
                        if len(raw_text) > 10000: raw_text = raw_text[:10000] + "... [text truncated]"
                        context_parts.append(f"RAW PAPER TEXT:\n{raw_text}")
                    full_context = "\n\n".join(context_parts)
                else:
                    full_context = "Error: Selected paper data not found."


                # Create prompt for LLM only if context was built
                if full_context:
                    prompt = f"""You are a medical research assistant helping to analyze scientific papers.
                    You are discussing {'multiple papers' if selected_paper_id_key == 'all' else 'a paper'}.
                    {context_intro}

                    {full_context}

                    Based ONLY on the information provided above, please answer the following question:
                    {user_query}

                    If the answer cannot be determined from the provided information, state that clearly. Do not make assumptions or use external knowledge.
                    """

                    # Get response from LLM
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                try:
                                    if 'llm' in st.session_state: # Check if LLM initialized
                                        response = st.session_state.llm.ask(prompt, max_tokens=800, temperature=0.3)
                                        st.write(response)
                                        # Add assistant response to chat history
                                        st.session_state.chat_history[selected_paper_id_key].append({
                                            "role": "assistant",
                                            "content": response
                                        })
                                    else:
                                        st.error("LLM not initialized. Cannot generate response.")
                                        st.session_state.chat_history[selected_paper_id_key].append({
                                            "role": "assistant",
                                            "content": "LLM not initialized."
                                        })
                                except Exception as e:
                                    error_message = f"Error generating response: {str(e)}"
                                    st.error(error_message)
                                    # Add error to chat history
                                    st.session_state.chat_history[selected_paper_id_key].append({
                                        "role": "assistant",
                                        "content": error_message
                                    })
                else:
                     # Handle case where no context could be built
                     with chat_container:
                         with st.chat_message("assistant"):
                             no_context_message = "Could not build context for the query based on selected options or paper data."
                             st.warning(no_context_message)
                             st.session_state.chat_history[selected_paper_id_key].append({
                                 "role": "assistant",
                                 "content": no_context_message
                             })
