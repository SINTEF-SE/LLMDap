import streamlit as st
import json
import sys
import os
import glob
import random
from datetime import datetime

# Add paths for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm import LLM

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """You are an AI assistant specializing in biomedical research datasets. Your task is to answer questions about the provided datasets from ArrayExpress/BioStudies.

Title: {title}
Abstract: {abstract}

Available datasets:
{datasets}

Question: {question}

Please provide a comprehensive, accurate answer based on the dataset information provided above.
"""

def find_dataset_files(data_dir=None):
    """Find all ArrayExpress dataset files in the data directory"""
    if not data_dir:
        # Try with the default location first
        default_dirs = [
            "/mnt/data/upcast/data/arxpr",  # Original data location
            os.path.join(project_root, "data", "arxpr"),  # Local data directory
            os.path.join(project_root, "data")  # Fallback
        ]
        
        for dir_path in default_dirs:
            if os.path.exists(dir_path):
                print(f"Found data directory: {dir_path}")
                data_dir = dir_path
                break
    
    if not data_dir or not os.path.exists(data_dir):
        print("No data directory found")
        return []
    
    # Look for JSON files with the ArrayExpress pattern (PMID___E-XXXX-NNNN.json)
    dataset_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"Found {len(dataset_files)} JSON files in {data_dir}")
    
    if len(dataset_files) > 0:
        print(f"Example files: {dataset_files[:3]}")
    
    return dataset_files

def load_dataset_sample(dataset_files, max_samples=10):
    """Load a sample of datasets from the files"""
    if not dataset_files:
        return []
    
    # If there are too many files, sample a subset
    if len(dataset_files) > max_samples:
        sample_files = random.sample(dataset_files, max_samples)
    else:
        sample_files = dataset_files
        
    datasets = []
    for file_path in sample_files:
        try:
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Try to parse as JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                st.warning(f"Couldn't parse JSON from {file_path}")
                continue
            
            # If data is a string, something went wrong
            if isinstance(data, str):
                st.warning(f"File content is a string, not a JSON object: {file_path}")
                # Try to parse it again in case it's double-encoded
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    continue  # Skip this file
                
            # Extract the basename without extension
            basename = os.path.basename(file_path)
            try:
                pmid, accession = basename.split("___")
                accession = accession.replace(".json", "")
            except ValueError:
                # Handle files with unexpected naming format
                pmid = "unknown"
                accession = basename.replace(".json", "")
            
            # Extract useful metadata
            title = ""
            description = ""
            organism = ""
            study_type = ""
            release_date = ""
            experimental_factors = ""
            technology = ""
            
            # Navigate through the ArrayExpress JSON structure
            if isinstance(data, dict):
                # Try different ways to get the title
                if 'submissions' in data and isinstance(data['submissions'], list) and len(data['submissions']) > 0:
                    title = data['submissions'][0].get('title', '')
                
                # Extract study info
                if 'section' in data and isinstance(data['section'], list):
                    sections = data['section']
                    for section in sections:
                        if isinstance(section, dict) and section.get('type') == 'study':
                            attributes = section.get('attributes', [])
                            for attr in attributes:
                                if isinstance(attr, dict):
                                    if attr.get('name') == 'description':
                                        description = attr.get('value', '')
                                    elif attr.get('name') == 'organism':
                                        organism = attr.get('value', '')
                                    elif attr.get('name') == 'study type':
                                        study_type = attr.get('value', '')
                                    elif attr.get('name') == 'release date':
                                        release_date = attr.get('value', '')
                        
                        # Look for experimental factors in subsections
                        if isinstance(section, dict) and 'subsections' in section:
                            for subsection in section['subsections']:
                                if isinstance(subsection, dict) and subsection.get('type') == 'samples':
                                    for attr in subsection.get('attributes', []):
                                        if isinstance(attr, dict) and attr.get('name') == 'experimental factors':
                                            experimental_factors = attr.get('value', '')
                                        
                                if isinstance(subsection, dict) and subsection.get('type') == 'protocols':
                                    for subsubsection in subsection.get('subsections', []):
                                        if isinstance(subsubsection, dict) and 'attributes' in subsubsection:
                                            for attr in subsubsection['attributes']:
                                                if attr.get('name') == 'hardware' or attr.get('name') == 'technology':
                                                    technology = attr.get('value', '')
            
            # If we don't have a good title, try to fetch it from PubMed
            if not title or title.strip() == "" or title.startswith("Dataset E-"):
                try:
                    if pmid and pmid != "unknown":
                        import requests
                        from xml.etree import ElementTree as ET
                        
                        pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
                        
                        response = requests.get(pubmed_url, timeout=5)
                        if response.status_code == 200:
                            root = ET.fromstring(response.text)
                            
                            # Extract article title from PubMed
                            article_title = root.find(".//ArticleTitle")
                            if article_title is not None and article_title.text:
                                title = article_title.text
                except Exception as e:
                    # If PubMed lookup fails, use accession as title
                    if not title:
                        title = f"Dataset {accession}"
            
            # Create a dataset object with improved title
            dataset = {
                "title": title or f"Dataset {accession}",
                "accession": accession,
                "pmid": pmid,
                "url": f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}",
                "file_path": file_path,
                "description": description,
                "organism": organism,
                "study_type": study_type,
                "release_date": release_date,
                "experimental_factors": experimental_factors,
                "technology": technology
            }
            
            datasets.append(dataset)
            
        except Exception as e:
            st.warning(f"Error loading dataset from {file_path}: {str(e)}")
            continue
    
    return datasets

def load_cached_datasets():
    """Load cached datasets from file if available."""
    try:
        with open("cached_datasets.json", "r") as f:
            content = f.read().strip()
            if not content or content == "[]" or content == "{}":
                print("Cached datasets file is empty")
                return None
                
            datasets = json.loads(content)
            if not datasets or not isinstance(datasets, list) or len(datasets) == 0:
                print("No valid datasets in cache")
                return None
                
            print(f"Loaded {len(datasets)} cached datasets")
            return datasets
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading cached datasets: {e}")
        return None

def save_cached_datasets(datasets):
    """Save datasets to cache file for future use."""
    try:
        with open("cached_datasets.json", "w") as f:
            json.dump(datasets, f)
        print(f"Saved {len(datasets)} datasets to cache")
        return True
    except Exception as e:
        print(f"Error saving datasets to cache: {e}")
        return False

def load_settings():
    """Load settings from settings.json or use defaults."""
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
            
            # Check if the template is the old one and replace it
            if 'prompt_template' in settings:
                current_template = settings['prompt_template']
                if "You are an AI language model assistant" in current_template:
                    # Replace with the new template
                    settings['prompt_template'] = DEFAULT_PROMPT_TEMPLATE
                    # Save the updated settings
                    with open('settings.json', 'w') as f_write:
                        json.dump(settings, f_write, indent=2)
                    print("Updated prompt template in settings.json")
            
            return settings
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'temperature': 0.3, 
            'max_tokens': 800, 
            'prompt_template': DEFAULT_PROMPT_TEMPLATE
        }

def extract_dataset_metadata(dataset):
    """Extract and format metadata from a dataset for the LLM."""
    import re
    import os
    import requests
    
    metadata = []
    
    # Basic metadata always available
    if dataset.get("title"):
        metadata.append(f"Title: {dataset['title']}")
    
    if dataset.get("accession"):
        metadata.append(f"Accession: {dataset['accession']}")
        
    if dataset.get("pmid"):
        metadata.append(f"PMID: {dataset['pmid']}")
        metadata.append(f"PubMed Link: https://pubmed.ncbi.nlm.nih.gov/{dataset['pmid']}/")
    
    # STEP 1: PUBMED FETCHER - works well already
    if dataset.get("pmid") and dataset["pmid"] != "unknown":
        try:
            import requests
            from xml.etree import ElementTree as ET
            
            pmid = dataset["pmid"]
            pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            
            response = requests.get(pubmed_url, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                
                # Extract article title
                article_title = root.find(".//ArticleTitle")
                if article_title is not None and article_title.text:
                    metadata.append(f"\n### Publication Information:")
                    metadata.append(f"Article Title: {article_title.text}")
                
                # Extract abstract text
                abstract_texts = root.findall(".//AbstractText")
                if abstract_texts:
                    metadata.append(f"Abstract:")
                    abstract_content = []
                    for abstract_part in abstract_texts:
                        label = abstract_part.get("Label")
                        if label:
                            abstract_content.append(f"**{label}**: {abstract_part.text}")
                        else:
                            abstract_content.append(abstract_part.text)
                    metadata.append("\n".join(abstract_content))
                
                # Extract publication date
                pub_date = root.find(".//PubDate")
                if pub_date is not None:
                    year = pub_date.find("Year")
                    month = pub_date.find("Month")
                    day = pub_date.find("Day")
                    pub_date_str = ""
                    if year is not None and year.text:
                        pub_date_str += year.text
                    if month is not None and month.text:
                        pub_date_str += f"-{month.text}"
                    if day is not None and day.text:
                        pub_date_str += f"-{day.text}"
                    if pub_date_str:
                        metadata.append(f"Publication Date: {pub_date_str}")
                
                # Extract authors
                authors = root.findall(".//Author")
                if authors:
                    author_names = []
                    for author in authors[:5]:  # Limit to first 5 authors
                        last_name = author.find("LastName")
                        first_name = author.find("ForeName")
                        if last_name is not None and last_name.text:
                            author_name = last_name.text
                            if first_name is not None and first_name.text:
                                author_name = f"{first_name.text} {author_name}"
                            author_names.append(author_name)
                    if author_names:
                        if len(authors) > 5:
                            author_names.append("et al.")
                        metadata.append(f"Authors: {', '.join(author_names)}")
                
                # Extract journal info
                journal = root.find(".//Journal/Title")
                if journal is not None and journal.text:
                    metadata.append(f"Journal: {journal.text}")
                    
                # NEW: Extract MeSH terms for better context
                mesh_headings = root.findall(".//MeshHeading")
                if mesh_headings:
                    mesh_terms = []
                    for heading in mesh_headings[:10]:  # Limit to 10 terms
                        descriptor = heading.find("DescriptorName")
                        if descriptor is not None and descriptor.text:
                            mesh_terms.append(descriptor.text)
                    if mesh_terms:
                        metadata.append(f"MeSH Terms: {', '.join(mesh_terms)}")
        except Exception as e:
            metadata.append(f"Error fetching PubMed data: {str(e)}")
    
    # STEP 2: ARRAYEXPRESS DIRECT API - Different endpoints for more data
    try:
        import requests
        accession = dataset['accession']
        
        # Try the main BioStudies API first
        ae_url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}"
        response = requests.get(ae_url, timeout=10)
        
        if response.status_code == 200:
            ae_data = response.json()
            metadata.append(f"\n### ArrayExpress Dataset Information:")
            
            if 'name' in ae_data:
                metadata.append(f"Dataset Name: {ae_data['name']}")
            
            # Extract key metadata
            if 'section' in ae_data:
                for section in ae_data['section']:
                    if isinstance(section, dict) and section.get('type') == 'study':
                        if 'attributes' in section:
                            for attr in section['attributes']:
                                if isinstance(attr, dict) and 'name' in attr and 'value' in attr:
                                    name = attr['name'].capitalize()
                                    value = attr['value']
                                    metadata.append(f"{name}: {value}")
        
        # Try the ArrayExpress legacy API (more detailed for older datasets)
        legacy_url = f"https://www.ebi.ac.uk/arrayexpress/json/v3/experiments/{accession}"
        response = requests.get(legacy_url, timeout=10)
        
        if response.status_code == 200:
            try:
                legacy_data = response.json()
                if 'experiments' in legacy_data and 'experiment' in legacy_data['experiments'] and legacy_data['experiments']['experiment']:
                    exp = legacy_data['experiments']['experiment'][0]
                    metadata.append(f"\n### ArrayExpress Detailed Information:")
                    
                    # Extract key fields
                    for field in ['name', 'experimenttype', 'description', 'organism']:
                        if field in exp:
                            metadata.append(f"{field.capitalize()}: {exp[field]}")
                    
                    # Extract sample attributes
                    if 'sampleattribute' in exp:
                        metadata.append("Sample Attributes:")
                        for attr in exp['sampleattribute']:
                            if 'category' in attr and 'value' in attr:
                                metadata.append(f"  {attr['category']}: {attr['value']}")
                    
                    # Extract experimental factors
                    if 'experimentalfactor' in exp:
                        metadata.append("Experimental Factors:")
                        for factor in exp['experimentalfactor']:
                            if 'name' in factor:
                                metadata.append(f"  {factor['name']}")
                    
                    # Extract protocols
                    if 'protocol' in exp:
                        metadata.append("Protocols:")
                        for protocol in exp['protocol']:
                            if 'text' in protocol:
                                text = protocol['text']
                                if len(text) > 200:
                                    text = text[:200] + "..."
                                metadata.append(f"  {protocol.get('type', 'Protocol')}: {text}")
            except:
                pass
                
        # Try the detailed IDF download (contains experiment design)
        try:
            idf_url = f"https://www.ebi.ac.uk/arrayexpress/files/{accession}/{accession}.idf.txt"
            idf_response = requests.get(idf_url, timeout=10)
            
            if idf_response.status_code == 200:
                idf_content = idf_response.text
                metadata.append(f"\n### IDF File Content (Experiment Design):")
                lines = idf_content.strip().split('\n')[:20]  # First 20 lines
                for line in lines:
                    metadata.append(line)
                metadata.append("...")
        except:
            pass
            
        # Try the SDRF download (contains sample details)
        try:
            sdrf_url = f"https://www.ebi.ac.uk/arrayexpress/files/{accession}/{accession}.sdrf.txt"
            sdrf_response = requests.get(sdrf_url, timeout=10)
            
            if sdrf_response.status_code == 200:
                sdrf_content = sdrf_response.text
                metadata.append(f"\n### SDRF File Content (Sample Details):")
                lines = sdrf_content.strip().split('\n')[:10]  # First 10 lines
                for line in lines:
                    if len(line) > 200:
                        line = line[:200] + "..."
                    metadata.append(line)
                metadata.append("...")
        except:
            pass
    except Exception as e:
        metadata.append(f"Error fetching ArrayExpress data: {str(e)}")
    
    # STEP 3: EUROPEPMC API - For additional experimental details
    try:
        if dataset.get("pmid") and dataset["pmid"] != "unknown":
            pmid = dataset["pmid"]
            europepmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&resultType=core&format=json"
            
            response = requests.get(europepmc_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'resultList' in data and 'result' in data['resultList'] and data['resultList']['result']:
                    article = data['resultList']['result'][0]
                    
                    metadata.append(f"\n### Additional Publication Details:")
                    
                    # Get publication type
                    if 'pubTypeList' in article and 'pubType' in article['pubTypeList']:
                        pub_types = ", ".join(article['pubTypeList']['pubType'])
                        metadata.append(f"Publication Type: {pub_types}")
                    
                    # Get grants
                    if 'grantsList' in article and 'grant' in article['grantsList']:
                        grants = []
                        for grant in article['grantsList']['grant'][:3]:
                            if 'agency' in grant:
                                grant_info = grant['agency']
                                if 'grantId' in grant:
                                    grant_info += f" ({grant['grantId']})"
                                grants.append(grant_info)
                        if grants:
                            metadata.append(f"Funding: {', '.join(grants)}")
                    
                    # Get chemicals/substances
                    if 'chemicalList' in article and 'chemical' in article['chemicalList']:
                        chemicals = []
                        for chemical in article['chemicalList']['chemical'][:5]:
                            if 'name' in chemical:
                                chemicals.append(chemical['name'])
                        if chemicals:
                            metadata.append(f"Chemicals/Substances: {', '.join(chemicals)}")
    except Exception as e:
        pass  # Silently continue
    
    # STEP 4: Try to extract minimal info from our JSON file
    try:
        with open(dataset["file_path"], 'r') as f:
            file_content = f.read()
            data = json.loads(file_content)
            
            # Get basic file names
            if 'files' in data and isinstance(data['files'], list):
                file_paths = [file['path'] for file in data['files'][:10] if isinstance(file, dict) and 'path' in file]
                
                if file_paths:
                    metadata.append("\n### Dataset Files:")
                    for file_path in file_paths:
                        metadata.append(f"- {file_path}")
                        
            # Use regex as backup if JSON parsing fails for some sections
            if not file_paths:
                # Extract files using regex
                file_matches = re.findall(r'"path"\s*:\s*"([^"]+\.(txt|cel|gpr|csv|tsv|idf|sdrf))"', file_content)
                if file_matches:
                    metadata.append("\n### Dataset Files (Extracted):")
                    for match in file_matches[:10]:
                        metadata.append(f"- {match[0]}")
    except Exception as e:
        pass  # Silently continue if file can't be read
    
    return "\n".join(metadata)

def filter_datasets(datasets, search_term):
    """Filter datasets based on a search term."""
    if not search_term:
        return datasets
        
    search_term = search_term.lower()
    filtered = []
    
    for dataset in datasets:
        # Search in title, description, organism, and study_type
        searchable_text = (
            (dataset.get('title', '') or '') + ' ' +
            (dataset.get('description', '') or '') + ' ' +
            (dataset.get('organism', '') or '') + ' ' +
            (dataset.get('study_type', '') or '')
        ).lower()
        
        if search_term in searchable_text:
            filtered.append(dataset)
    
    return filtered

def fetch_all_pubmed_titles(datasets):
    """Fetch all PubMed titles in bulk to improve display efficiency"""
    import requests
    from xml.etree import ElementTree as ET
    
    # Get all PMIDs that need titles
    pmids_to_fetch = []
    pmid_indices = {}  # Map PMIDs to dataset indices
    
    for i, dataset in enumerate(datasets):
        if dataset.get("pmid") and dataset["pmid"] != "unknown":
            # Only fetch if title is missing or generic
            if not dataset.get("title") or dataset["title"].startswith("Dataset E-"):
                pmids_to_fetch.append(dataset["pmid"])
                pmid_indices[dataset["pmid"]] = i
    
    if not pmids_to_fetch:
        return datasets  # No titles to fetch
    
    # Split into batches of 50 to avoid overloading PubMed API
    batch_size = 50
    batches = [pmids_to_fetch[i:i+batch_size] for i in range(0, len(pmids_to_fetch), batch_size)]
    
    for batch in batches:
        try:
            # Create a comma-separated list of PMIDs
            pmid_list = ",".join(batch)
            
            # Fetch data from PubMed for this batch
            pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid_list}&retmode=xml"
            response = requests.get(pubmed_url, timeout=15)
            
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                
                # Process each PubMedArticle element
                for article in root.findall(".//PubmedArticle"):
                    # Get the PMID
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is not None and pmid_elem.text:
                        pmid = pmid_elem.text
                        
                        # If this PMID is in our list, extract the title
                        if pmid in pmid_indices:
                            article_title = article.find(".//ArticleTitle")
                            if article_title is not None and article_title.text:
                                # Update the dataset's title
                                datasets[pmid_indices[pmid]]["title"] = article_title.text
        except Exception as e:
            print(f"Error fetching batch of PubMed titles: {e}")
            continue
    
    return datasets

def show():
    st.title("Dataset Q&A")
    
    # Initialize LLM if not already in session state
    if 'llm' not in st.session_state:
        try:
            with st.spinner("Initializing LLM... This may take a moment."):
                st.session_state.llm = LLM()
            st.success("LLM initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            st.stop()
    
    # Check if we have selected datasets from the Datasets page
    if 'selected_datasets' not in st.session_state or not st.session_state.selected_datasets:
        st.warning("No datasets selected. Please select datasets in the Dataset Browser first.")
        
        # Add a button to navigate to the Datasets page
        if st.button("Go to Dataset Browser"):
            st.session_state.show_page = "Dataset Browser"
            st.rerun()
        return
    
    selected_datasets = st.session_state.selected_datasets
    
    # Initialize chat history if not already in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Show selected datasets with badges
    st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True)
    for ds in selected_datasets:
        title = ds['title'] or ds['accession']
        if len(title) > 40:
            title = title[:40] + "..."
        st.markdown(f"<span class='datasets-badge'>📊 {title}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Chat Display Area ---
    st.markdown("<h3>Chat</h3>", unsafe_allow_html=True)
    # Create a container for the chat messages
    chat_display_container = st.container()
    # Apply the chat-container style to this container specifically
    # Note: Applying height/overflow directly via st.container might not work reliably,
    # so we rely on the CSS class defined earlier. We add a key for potential future JS interaction.
    chat_display_container.markdown("<div class='chat-container' id='chat-container-div'>", unsafe_allow_html=True) # Start the styled div

    with chat_display_container:
        if not st.session_state.chat_history:
            st.markdown("<div style='text-align: center; color: #888; padding: 20px;'>Ask a question about the selected datasets to get started</div>", unsafe_allow_html=True)
        else:
            for i, (time_stamp, question_text, answer_text) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class='message-container' style='align-items: flex-end;'>
                    <div class='user-message'>
                        {question_text}
                        
                </div>
                """, unsafe_allow_html=True)

                # Assistant message
                st.markdown(f"""
                <div class='message-container' style='align-items: flex-start;'>
                    <div class='assistant-message'>
                        {answer_text}
                        <div class='message-time'>Assistant</div> 
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) # Close the chat-container-div
    
    # Input area below the chat
    st.markdown("<h3>Ask a Question</h3>", unsafe_allow_html=True)
    st.markdown("<div class='input-area'>", unsafe_allow_html=True)
    
    # Combine both simple and complex questions into a single input
    # Use columns to create a layout with input and buttons
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col1:
        question = st.text_area("Enter your question:", 
                              height=80,
                              placeholder="Ask anything about the selected datasets - experimental design, organisms, methods, etc.",
                              key="question_input",
                              label_visibility="collapsed")
    
    with col2:
        clear_button = st.button("🧹 Clear", key="clear_input")
    
    with col3:
        submit_button = st.button("📤 Send", key="submit_question", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear input if requested
    if clear_button:
        st.session_state.question_input = ""
        st.experimental_rerun()
    
    # Clear entire chat history if requested
    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    # Check for submission action from previous run
    if 'submit_question_pressed' in st.session_state and st.session_state.submit_question_pressed and 'current_question' in st.session_state and st.session_state.current_question:
        # Get the question from session state
        question = st.session_state.current_question
        
        # Reset the flags
        st.session_state.submit_question_pressed = False
        st.session_state.current_question = ""
        
        # Temporarily display a spinner at the end of the chat
        with st.spinner("Generating answer..."):
            # Format the datasets for the prompt with detailed metadata
            formatted_datasets = []
            
            # Get progress info
            progress_text = "Fetching dataset metadata..."
            progress_bar = st.progress(0, text=progress_text)
            
            for i, dataset in enumerate(selected_datasets):
                progress_value = (i + 1) / len(selected_datasets)
                progress_bar.progress(progress_value, text=f"{progress_text} ({i+1}/{len(selected_datasets)})")
                
                # Get detailed metadata using our enhanced function
                dataset_info = extract_dataset_metadata(dataset)
                
                # Add to formatted datasets with proper numbering and spacing
                formatted_datasets.append(f"\n### DATASET {i+1}: {dataset['title'] or dataset['accession']}\n")
                formatted_datasets.append(dataset_info)
                formatted_datasets.append("\n---\n")
            
            # Remove progress bar when done
            progress_bar.empty()
            
            # Join all dataset information
            formatted_dataset_text = "".join(formatted_datasets)
            
            # Create title and abstract for template
            if len(selected_datasets) == 1:
                title = selected_datasets[0].get("title", "") or f"Dataset {selected_datasets[0]['accession']}"
                abstract = selected_datasets[0].get("description", "") or "No abstract available in dataset metadata"
            else:
                title = f"Collection of {len(selected_datasets)} Datasets"
                abstract = "Multiple datasets selected. See details below."
            
            # Create the prompt with explicit instructions for metadata focus
            prompt = f"""You are an AI assistant specializing in biomedical research datasets. You are answering questions about ArrayExpress/BioStudies datasets.

TITLE: {title}

ABSTRACT: {abstract}

QUESTION: {question}

AVAILABLE DATASETS:
{formatted_dataset_text}

Based on the detailed dataset information above, which includes publication abstracts, experimental metadata, and file descriptions, provide a comprehensive answer to the question. 
When addressing metadata specifically, focus on the experimental design, sample information, protocols, and technical details of the dataset itself.
"""
            
            # Show debug information
            with st.expander("Debug: Prompt being sent to LLM", expanded=False):
                st.text(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
                st.text(f"Total prompt length: {len(prompt)} characters")
            
            # Get the response with higher max_tokens to allow for detailed answers
            settings = load_settings()
            output = st.session_state.llm.ask(
                prompt, 
                max_tokens=settings.get('max_tokens', 1000),  # Increased max tokens
                temperature=settings.get('temperature', 0.2)  # Slightly lower temperature for more factual answers
            )

            # Store Q&A in chat history
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.chat_history.append((current_time, question, output))
            
            # Force a rerun to update the UI
            st.rerun()
    
    # Handle the submit button click by setting flags in session state
    if submit_button and question:
        st.session_state.submit_question_pressed = True
        st.session_state.current_question = question
        st.rerun()
    
    # Debug information
    with st.expander("Debug Information", expanded=False):
        st.write(f"Total datasets found: {len(st.session_state.datasets)}")
        st.write(f"Selected datasets: {len(st.session_state.selected_datasets)}")
        st.write(f"Selected datasets: {len(selected_datasets)}")
        
        if os.path.exists("cached_datasets.json"):
            st.info(f"cached_datasets.json: {os.path.getsize('cached_datasets.json')} bytes, " 
                   f"Last modified: {datetime.fromtimestamp(os.path.getmtime('cached_datasets.json'))}")
        
        # Show a sample of the dataset structure
        if st.session_state.datasets:
            st.subheader("Sample Dataset Structure")
            st.json(st.session_state.datasets[0])
