import streamlit as st
import sys
import os
import json
import glob
import requests
import time
import re

def load_cached_datasets():
    """Load cached datasets from file if available."""
    try:
        with open('cached_datasets.json', 'r') as f:
            datasets = json.load(f)
        return datasets
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.warning(f"Could not load cached datasets: {str(e)}")
        return None

def save_cached_datasets(datasets):
    """Save datasets to cache file for future use."""
    try:
        with open('cached_datasets.json', 'w') as f:
            json.dump(datasets, f)
    except Exception as e:
        st.warning(f"Could not save datasets to cache: {str(e)}")

def find_dataset_files(data_dir=None):
    """Find all ArrayExpress dataset files in the data directory"""
    if not data_dir:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    
    if not data_dir or not os.path.exists(data_dir):
        st.error(f"Data directory not found: {data_dir}")
        return []
    
    # Look for JSON files with the ArrayExpress pattern (PMID___E-XXXX-NNNN.json)
    dataset_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"Found {len(dataset_files)} JSON files in {data_dir}")
    
    if len(dataset_files) > 0:
        st.success(f"Found {len(dataset_files)} dataset files")
    
    return dataset_files

def load_dataset_sample(dataset_files, max_samples=10):
    """Load a sample of datasets from the files"""
    import random
    
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
            with open(file_path, 'r') as f:
                dataset = json.load(f)
                
                # Extract basic metadata
                filename = os.path.basename(file_path)
                accession_match = re.search(r'(E-\w+-\d+)', filename)
                
                dataset_info = {
                    'title': dataset.get('title', 'Unknown title'),
                    'accession': accession_match.group(1) if accession_match else 'Unknown',
                    'pmid': dataset.get('pmid', 'unknown'),
                    'description': dataset.get('description', ''),
                    'organism': dataset.get('organism', 'Unknown'),
                    'study_type': dataset.get('study_type', 'Unknown'),
                    'url': f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession_match.group(1)}" if accession_match else "#"
                }
                datasets.append(dataset_info)
        except Exception as e:
            print(f"Error loading dataset {file_path}: {str(e)}")
    
    return datasets

def filter_datasets(datasets, search_term):
    """Filter datasets based on a search term."""
    if not search_term:
        return datasets
        
    search_term = search_term.lower()
    filtered = []
    
    for dataset in datasets:
        # Check if search term appears in any of these fields
        if (search_term in dataset.get('title', '').lower() or
            search_term in dataset.get('description', '').lower() or
            search_term in dataset.get('organism', '').lower() or
            search_term in dataset.get('study_type', '').lower() or
            search_term in dataset.get('accession', '').lower()):
            filtered.append(dataset)
    
    return filtered

def fetch_all_pubmed_titles(datasets):
    """Fetch all PubMed titles in bulk to improve display efficiency"""
    from xml.etree import ElementTree as ET
    
    # Get all PMIDs that need titles
    pmids_to_fetch = []
    pmid_indices = {}  # Map PMIDs to dataset indices
    
    for i, dataset in enumerate(datasets):
        if dataset.get('pmid') and dataset['pmid'] != 'unknown' and not dataset.get('pubmed_title'):
            pmids_to_fetch.append(dataset['pmid'])
            pmid_indices[dataset['pmid']] = i
    
    if not pmids_to_fetch:
        return datasets
    
    # Split into batches of 50 to avoid overloading PubMed API
    batch_size = 50
    batches = [pmids_to_fetch[i:i+batch_size] for i in range(0, len(pmids_to_fetch), batch_size)]
    
    for batch in batches:
        try:
            # Use PubMed API to fetch titles
            pmids_str = ",".join(batch)
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmids_str}"
            response = requests.get(url)
            
            if response.status_code == 200:
                xml = ET.fromstring(response.text)
                
                for doc in xml.findall(".//DocSum"):
                    pmid = doc.find("Id").text
                    title_elem = None
                    
                    for item in doc.findall(".//Item"):
                        if item.attrib.get('Name') == 'Title':
                            title_elem = item
                            break
                    
                    if title_elem is not None and pmid in pmid_indices:
                        datasets[pmid_indices[pmid]]['pubmed_title'] = title_elem.text
                        # Use PubMed title to supplement dataset title if needed
                        if not datasets[pmid_indices[pmid]]['title'] or datasets[pmid_indices[pmid]]['title'] == 'Unknown title':
                            datasets[pmid_indices[pmid]]['title'] = title_elem.text
            
            # Be nice to the PubMed API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching PubMed titles: {str(e)}")
    
    return datasets

def show():
    st.title("Dataset Browser")
    st.write("Browse and select datasets for analysis")
    
    # Initialize session state for selected datasets
    if 'selected_datasets' not in st.session_state:
        st.session_state.selected_datasets = []
    
    # Try to load cached datasets first
    if 'datasets' not in st.session_state:
        cached_datasets = load_cached_datasets()
        if cached_datasets:
            st.session_state.datasets = cached_datasets
        else:
            # Find dataset files
            with st.spinner("Searching for ArrayExpress datasets..."):
                dataset_files = find_dataset_files()
                
                if not dataset_files:
                    st.error("No ArrayExpress dataset files found. Please check your data directory.")
                    st.stop()
                
                # Allow user to specify how many datasets to load
                with st.expander("Advanced Settings", expanded=False):
                    max_datasets = st.slider("Number of datasets to load:", min_value=10, max_value=200, value=50, step=10)
                    st.session_state.datasets = load_dataset_sample(dataset_files, max_samples=max_datasets)
                    save_cached_datasets(st.session_state.datasets)

        # Add this new section to fetch all titles at once
        with st.spinner("Fetching publication titles..."):
            st.session_state.datasets = fetch_all_pubmed_titles(st.session_state.datasets)
            # Update the cached datasets with the improved titles
            save_cached_datasets(st.session_state.datasets)
    
    # Dataset search and filtering
    st.header("Browse Datasets")
    
    search_term = st.text_input("Search datasets:", "")
    filtered_datasets = filter_datasets(st.session_state.datasets, search_term)
    
    st.write(f"Found {len(filtered_datasets)} matching datasets")
    
    # Implement pagination for datasets
    datasets_per_page = 10
    total_pages = (len(filtered_datasets) + datasets_per_page - 1) // datasets_per_page

    # Create columns for navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        if st.button("Previous", disabled=st.session_state.current_page == 0):
            st.session_state.current_page -= 1
            st.rerun()
            
    with col2:
        st.write(f"Page {st.session_state.current_page + 1} of {max(1, total_pages)}")
        
    with col3:
        if st.button("Next", disabled=st.session_state.current_page >= total_pages - 1):
            st.session_state.current_page += 1
            st.rerun()

    # Calculate page start and end
    start_idx = st.session_state.current_page * datasets_per_page
    end_idx = min(start_idx + datasets_per_page, len(filtered_datasets))

    # Display only the current page of datasets
    with st.expander("Available Datasets", expanded=True):
        for i in range(start_idx, end_idx):
            dataset = filtered_datasets[i]
            col1, col2, col3 = st.columns([6, 3, 1])
            
            with col1:
                # Better title display
                title = dataset['title'] or f"Dataset {dataset['accession']}"
                st.markdown(f"**{i+1}. [{title}]({dataset['url']})**")
                
                # Show a snippet of description if available
                if dataset['description']:
                    desc = dataset['description'][:100] + "..." if len(dataset['description']) > 100 else dataset['description']
                    st.write(desc)
            
            with col2:
                # Show key metadata that helps identify the dataset
                metadata = []
                if dataset.get('organism'):
                    metadata.append(f"Organism: {dataset['organism']}")
                if dataset.get('study_type'):
                    metadata.append(f"Study type: {dataset['study_type']}")
                if metadata:
                    st.write("\n".join(metadata))
                st.write(f"Accession: {dataset['accession']}")
            
            with col3:
                # Selection checkbox
                dataset_key = f"select_dataset_{i}"
                if dataset_key not in st.session_state:
                    st.session_state[dataset_key] = False
                st.session_state[dataset_key] = st.checkbox("Select", key=f"cb_{i}", value=st.session_state[dataset_key])
            
            # Add a separator between entries
            st.markdown("---")
    
    # Get selected datasets
    st.session_state.selected_datasets = [
        dataset for i, dataset in enumerate(filtered_datasets) 
        if st.session_state.get(f"select_dataset_{i}", False)
    ]
    
    # Navigation to Consumer Q&A page
    st.header("Selected Datasets", divider="gray")
    
    if st.session_state.selected_datasets:
        st.write(f"You have selected {len(st.session_state.selected_datasets)} datasets:")
        
        # Display selected datasets in a compact grid
        cols = st.columns(3)
        for i, ds in enumerate(st.session_state.selected_datasets):
            col_idx = i % 3
            with cols[col_idx]:
                st.markdown(f"- **{ds['title'] or ds['accession']}**")
        
        # Center the button with columns
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("Ask Questions About Selected Datasets", use_container_width=True):
                # Set the page to navigate to
                st.session_state.show_page = "Consumer Q&A"
                st.rerun()
    else:
        st.info("Please select at least one dataset to continue")