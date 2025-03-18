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
            
            # Create a dataset object
            dataset = {
                "title": title or f"Dataset {accession}",
                "accession": accession,
                "pmid": pmid,
                "url": f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}",
                "file_path": file_path,
                "description": description,
                "organism": organism,
                "study_type": study_type,
                "release_date": release_date
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
    metadata = []
    
    if dataset["title"]:
        metadata.append(f"Title: {dataset['title']}")
    
    if dataset["accession"]:
        metadata.append(f"Accession: {dataset['accession']}")
    
    if dataset["description"]:
        # Limit description length
        desc = dataset["description"]
        if len(desc) > 200:
            desc = desc[:200] + "..."
        metadata.append(f"Description: {desc}")
    
    if dataset["organism"]:
        metadata.append(f"Organism: {dataset['organism']}")
    
    if dataset["study_type"]:
        metadata.append(f"Study Type: {dataset['study_type']}")
    
    # Try to load detailed info from the file
    try:
        with open(dataset["file_path"], 'r') as f:
            data = json.load(f)
            
        # Extract experimental factors if available
        if 'section' in data:
            for section in data['section']:
                if 'subsections' in section:
                    for subsec in section['subsections']:
                        if isinstance(subsec, dict) and subsec.get('type') == 'samples':
                            for attr in subsec.get('attributes', []):
                                if attr.get('name') == 'experimental factors':
                                    factors = attr.get('value', '')
                                    if factors:
                                        metadata.append(f"Experimental Factors: {factors}")
    except:
        pass
    
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

def show():
    st.title("ArrayExpress Dataset Explorer & Q&A")
    
    # Initialize LLM if not already in session state
    if 'llm' not in st.session_state:
        try:
            with st.spinner("Initializing LLM... This may take a moment."):
                st.session_state.llm = LLM()
            st.success("LLM initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            st.stop()
    
    # Try to load cached datasets first
    if 'datasets' not in st.session_state:
        cached_datasets = load_cached_datasets()
        if (cached_datasets):
            st.session_state.datasets = cached_datasets
        else:
            # Find dataset files
            with st.spinner("Searching for ArrayExpress datasets..."):
                dataset_files = find_dataset_files()
                
                if not dataset_files:
                    st.error("No ArrayExpress dataset files found. Please check your data directory.")
                    st.stop()
                    
                st.session_state.datasets = load_dataset_sample(dataset_files, max_samples=50)
                save_cached_datasets(st.session_state.datasets)
    
    # Dataset search and filtering
    st.header("Browse ArrayExpress Datasets")
    
    search_term = st.text_input("Search datasets:", "")
    filtered_datasets = filter_datasets(st.session_state.datasets, search_term)
    
    st.write(f"Found {len(filtered_datasets)} matching datasets")
    
    # Display datasets in an expander
    with st.expander("Available Datasets", expanded=True):
        for i, dataset in enumerate(filtered_datasets):
            col1, col2 = st.columns([8, 2])
            with col1:
                st.markdown(f"**{i+1}. [{dataset['title'] or dataset['accession']}]({dataset['url']})**")
                if dataset['description']:
                    st.write(dataset['description'][:100] + "..." if len(dataset['description']) > 100 else dataset['description'])
            with col2:
                dataset_key = f"select_dataset_{i}"
                if dataset_key not in st.session_state:
                    st.session_state[dataset_key] = False
                st.session_state[dataset_key] = st.checkbox("Select", key=f"cb_{i}", value=st.session_state[dataset_key])
    
    # Get selected datasets
    selected_datasets = [
        dataset for i, dataset in enumerate(filtered_datasets) 
        if st.session_state.get(f"select_dataset_{i}", False)
    ]
    
    # Q&A section
    st.header("Ask Questions About Selected Datasets")
    
    if not selected_datasets:
        st.info("Please select at least one dataset to ask questions about.")
    else:
        # Show selected datasets
        st.subheader(f"Selected Datasets ({len(selected_datasets)})")
        for ds in selected_datasets:
            st.markdown(f"- **{ds['title'] or ds['accession']}**")
        
        # Get a question from the user
        question = st.text_area("Enter your question about these datasets:", height=100)
        settings = load_settings()

        # In the section where you process the user's question:
        if question and st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                # Format the datasets for the prompt
                dataset_info = []
                for i, ds in enumerate(selected_datasets):
                    dataset_info.append(f"Dataset {i+1}: {ds['title']}")
                    
                    # Add more details about the dataset
                    if ds["description"]:
                        dataset_info.append(f"Description: {ds['description']}")
                    
                    if ds["organism"]:
                        dataset_info.append(f"Organism: {ds['organism']}")
                        
                    if ds["study_type"]:
                        dataset_info.append(f"Study Type: {ds['study_type']}")
                        
                    dataset_info.append(f"Accession: {ds['accession']}")
                    dataset_info.append(f"PMID: {ds['pmid']}")
                    dataset_info.append("---")
                
                formatted_dataset_text = "\n".join(dataset_info)
                
                # Create a complete context with title/abstract for template placeholders
                title = ""
                abstract = ""
                
                # If there's just one dataset, use its title as the main title
                if len(selected_datasets) == 1:
                    title = selected_datasets[0]["title"]
                    abstract = selected_datasets[0]["description"] or "No abstract available"
                else:
                    # For multiple datasets, create a summary title
                    title = f"Collection of {len(selected_datasets)} Datasets"
                    abstract = "Multiple datasets selected. See details below."
                
                # Prepare the prompt
                settings = load_settings()
                
                # FORCE the correct prompt template regardless of settings
                prompt_template = """You are an AI assistant specializing in biomedical research datasets. Your task is to answer questions about the provided datasets from ArrayExpress/BioStudies.

Title: {title}
Abstract: {abstract}

Available datasets:
{datasets}

Question: {question}

Please provide a comprehensive, accurate answer based on the dataset information provided above.
"""
                
                # Replace all placeholders with actual values
                prompt = prompt_template.format(
                    title=title,
                    abstract=abstract,
                    question=question,
                    datasets=formatted_dataset_text
                )
                
                # Debug: Show the prompt being sent
                with st.expander("Debug: Prompt being sent to LLM", expanded=False):
                    st.text(prompt)
                
                # Get the response
                output = st.session_state.llm.ask(
                    prompt, 
                    max_tokens=settings.get('max_tokens', 800), 
                    temperature=settings.get('temperature', 0.3)
                )

                # Display the answer
                st.subheader("Answer:")
                st.markdown(output)
    
    # Debug information
    with st.expander("Debug Information", expanded=False):
        st.write(f"Total datasets found: {len(st.session_state.datasets)}")
        st.write(f"Filtered datasets: {len(filtered_datasets)}")
        st.write(f"Selected datasets: {len(selected_datasets)}")
        
        if os.path.exists("cached_datasets.json"):
            st.info(f"cached_datasets.json: {os.path.getsize('cached_datasets.json')} bytes, " 
                   f"Last modified: {datetime.fromtimestamp(os.path.getmtime('cached_datasets.json'))}")
        
        # Show a sample of the dataset structure
        if st.session_state.datasets:
            st.subheader("Sample Dataset Structure")
            st.json(st.session_state.datasets[0])






