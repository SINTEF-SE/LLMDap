import streamlit as st
import json
import os
import sys
import re
import glob
import random
import importlib.util
import inspect
import math
from datetime import datetime
from typing import List, Dict, Any

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'profiler'))

# Try to import metadata schemas
try:
    from profiler.metadata_schemas import (
        ega_schema, nhrf_schema, nhrf_qa_schema, 
        nhrf_qa_schema_2, arxpr_schema, study_type_schema,
        arxpr2_schemas, get_shuffled_arxpr2
    )
    SCHEMAS_IMPORTED = True
except ImportError as e:
    print(f"Warning: Could not import metadata schemas: {e}")
    SCHEMAS_IMPORTED = False

def load_cached_datasets():
    """Load cached datasets from file if available."""
    try:
        cache_file = os.path.join(project_root, 'cached_datasets.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                datasets = json.load(f)
            print(f"Loaded {len(datasets)} datasets from cache")
            return datasets
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load cached datasets: {str(e)}")
    return None

def save_cached_datasets(datasets):
    """Save datasets to cache file for future use."""
    try:
        cache_file = os.path.join(project_root, 'cached_datasets.json')
        with open(cache_file, 'w') as f:
            json.dump(datasets, f)
        print(f"Saved {len(datasets)} datasets to cache")
    except Exception as e:
        print(f"Could not save datasets to cache: {str(e)}")

def find_dataset_files():
    """Find all dataset files from various sources"""
    # Result collections
    all_dataset_files = []
    found_dirs = []
    
    # Try multiple potential data directory locations
    potential_paths = [
        # Current implementation paths
        os.path.join(project_root, 'data'),
        os.path.join(project_root, 'data', 'arxpr'),
        "/mnt/data/upcast/data",
        "/mnt/data/upcast/data/arxpr",
        # Absolute path you specified
        "/home/upcast-shang/snap/snapd-desktop-integration/253/Documents/bacholer/ver1/bacholer-test/data",
        # Root project directory for loose files
        project_root
    ]
    
    print(f"Looking for datasets in the following directories:")
    for path in potential_paths:
        print(f"- {path}")
        if os.path.exists(path):
            found_dirs.append(path)
            json_files = glob.glob(os.path.join(path, "*.json"))
            print(f"  Found {len(json_files)} JSON files in {path}")
            all_dataset_files.extend(json_files)
    
    # Filter out configuration files from root directory
    config_files = ['cached_datasets.json', 'settings.json', 'previous_datasets.json']
    all_dataset_files = [f for f in all_dataset_files if os.path.basename(f) not in config_files]
    
    # Look for preprocessed dataset files mentioned in your code
    preprocessed_files = [
        "arxpr_simplified.json",
        "arxpr_metadataset_train.json", 
        "arxpr_metadataset_holdout.json",
        "arxpr2_25_metadataset_train.json",
        "arxpr2_25_metadataset_holdout.json"
    ]
    
    for filename in preprocessed_files:
        # Check across all potential directories
        for path in found_dirs:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path) and full_path not in all_dataset_files:
                all_dataset_files.append(full_path)
                print(f"Found special preprocessed file: {full_path}")
    
    # Handle schema files separately
    schema_files = []
    schema_dir = os.path.join(project_root, 'profiler', 'metadata_schemas')
    if os.path.exists(schema_dir):
        found_dirs.append(schema_dir)
        for filename in os.listdir(schema_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                schema_path = os.path.join(schema_dir, filename)
                schema_files.append(schema_path)
        print(f"Found {len(schema_files)} schema files in {schema_dir}")
    
    # Add schema files with special marking
    for schema_file in schema_files:
        all_dataset_files.append(f"schema:{schema_file}")
    
    # Report findings
    if found_dirs:
        st.success(f"Searched data in {len(found_dirs)} directories and found {len(all_dataset_files)} files")
        
        # Display first 5 files as a sample
        file_samples = [os.path.basename(f.replace('schema:', '')) for f in all_dataset_files[:5]]
        print(f"Sample files: {file_samples}")
    else:
        st.error("No data directories found. Please check your project structure.")
    
    return all_dataset_files

def find_all_dataset_files():
    """Find all dataset files without limiting the number"""
    # Use the existing find_dataset_files function but rename for clarity
    return find_dataset_files()

def load_schema_examples(schema_file_path):
    """Load examples from a schema file using importlib"""
    try:
        # Extract just the filename without extension
        filename = os.path.basename(schema_file_path).replace('.py', '')
        print(f"Loading schema from {schema_file_path}")
        
        # First try to load using direct imports if already imported
        if SCHEMAS_IMPORTED:
            schema_obj = None
            schema_name = filename.replace('_schema', '')
            
            if 'ega' in filename:
                schema_obj = ega_schema.Metadata_form
                schema_name = 'EGA Schema'
            elif 'nhrf_qa_schema_2' in filename:
                schema_obj = nhrf_qa_schema_2.Metadata_form
                schema_name = 'NHRF QA Schema 2'
            elif 'nhrf_qa' in filename:
                schema_obj = nhrf_qa_schema.Metadata_form
                schema_name = 'NHRF QA Schema'
            elif 'nhrf' in filename:
                schema_obj = nhrf_schema.Metadata_form
                schema_name = 'NHRF Schema'
            elif 'arxpr2' in filename:
                # Use first schema in the dict
                if hasattr(arxpr2_schemas, 'schemas') and len(arxpr2_schemas.schemas) > 0:
                    first_key = list(arxpr2_schemas.schemas.keys())[0]
                    schema_obj = arxpr2_schemas.schemas[first_key]
                    schema_name = f'ARXPR2 Schema (length {first_key})'
            elif 'study_type' in filename:
                schema_obj = study_type_schema.Metadata_form
                schema_name = 'Study Type Schema'
            elif 'arxpr' in filename:
                schema_obj = arxpr_schema.Metadata_form
                schema_name = 'ARXPR Schema'
            elif 'example' in filename:
                try:
                    from profiler.metadata_schemas import example_schema
                    schema_obj = example_schema.Metadata_form
                    schema_name = 'Example Schema'
                except ImportError:
                    pass
            
            if schema_obj:
                print(f"Successfully loaded schema: {schema_name}")
                # Get the schema fields
                fields = {}
                for field_name, field in schema_obj.__fields__.items():
                    field_description = field.field_info.description or "No description"
                    field_examples = field.field_info.examples or []
                    fields[field_name] = {
                        "description": field_description,
                        "examples": field_examples,
                        "type": str(field.type_)
                    }
                
                # Create virtual datasets from the schema fields
                sample_datasets = []
                sample_datasets.append({
                    'title': f"{schema_name} Definition",
                    'accession': 'SCHEMA-DEF',
                    'pmid': 'schema',
                    'description': getattr(schema_obj, '__doc__', 'Schema definition'),
                    'organism': 'N/A',
                    'study_type': 'Schema',
                    'url': "#",
                    'schema_path': schema_file_path,
                    'is_schema': True,
                    'fields': fields
                })
                
                return sample_datasets
        
        # Fallback: try to load the module directly from file
        spec = importlib.util.spec_from_file_location(filename, schema_file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find schema classes (BaseModel subclasses with __fields__ attribute)
            schemas = []
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and hasattr(obj, '__fields__'):
                    schemas.append((name, obj))
            
            if not schemas:
                return [{
                    'title': f"Schema File: {filename}",
                    'accession': 'UNKNOWN',
                    'pmid': 'schema',
                    'description': "Schema file but no identifiable schema classes found",
                    'organism': 'N/A',
                    'study_type': 'Schema',
                    'url': "#",
                    'schema_path': schema_file_path,
                    'is_schema': True
                }]
            
            results = []
            for name, schema_class in schemas:
                fields = {}
                for field_name, field in schema_class.__fields__.items():
                    field_description = field.field_info.description or "No description"
                    field_examples = field.field_info.examples or []
                    fields[field_name] = {
                        "description": field_description,
                        "examples": field_examples,
                        "type": str(field.type_)
                    }
                
                results.append({
                    'title': f"Schema: {name}",
                    'accession': 'SCHEMA',
                    'pmid': 'schema',
                    'description': getattr(schema_class, '__doc__', 'Schema definition'),
                    'organism': 'N/A',
                    'study_type': 'Schema',
                    'url': "#",
                    'schema_path': schema_file_path,
                    'is_schema': True,
                    'fields': fields
                })
            
            return results
        
    except Exception as e:
        print(f"Error loading schema file {schema_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Return a basic entry if all else fails
    return [{
        'title': f"Schema: {os.path.basename(schema_file_path)}",
        'accession': 'ERROR',
        'pmid': 'schema',
        'description': f"Error loading schema file: {schema_file_path}",
        'organism': 'N/A',
        'study_type': 'Error',
        'url': "#",
        'schema_path': schema_file_path,
        'is_schema': True
    }]

def load_dataset_sample(dataset_files, max_samples=10):
    """Load a sample of datasets from the files, excluding schema files"""
    if not dataset_files:
        return []
    
    # Filter out schema files completely
    regular_files = [f for f in dataset_files if not f.startswith('schema:')]
    
    # If there are too many regular files, sample a subset
    if len(regular_files) > max_samples:
        sample_files = random.sample(regular_files, max_samples)
    else:
        sample_files = regular_files
    
    # Process regular dataset files only
    regular_datasets = []
    bulk_files_processed = set()
    
    for file_path in sample_files:
        # Special handling for preprocessed metadata files
        file_basename = os.path.basename(file_path)
        if file_basename in [
            "arxpr_simplified.json",
            "arxpr_metadataset_train.json", 
            "arxpr_metadataset_holdout.json",
            "arxpr2_25_metadataset_train.json",
            "arxpr2_25_metadataset_holdout.json"
        ]:
            if file_path in bulk_files_processed:
                continue  # Skip already processed bulk files
                
            bulk_files_processed.add(file_path)
            
            try:
                with open(file_path, 'r') as f:
                    bulk_data = json.load(f)
                
                print(f"Loading bulk file {file_basename} with {len(bulk_data)} entries")
                
                # Take a sample of the datasets in the bulk file
                if isinstance(bulk_data, dict):
                    pmids = list(bulk_data.keys())
                    sample_size = min(10, len(pmids))  # Take more entries from bulk files since schemas are gone
                    pmid_sample = random.sample(pmids, sample_size)
                    
                    for pmid in pmid_sample:
                        data = bulk_data[pmid]
                        
                        # Extract metadata
                        organism = "Unknown"
                        study_type = "Unknown"
                        
                        # Try to determine study type & organism
                        if isinstance(data, dict):
                            for key, value in data.items():
                                if key.lower().startswith("study_type") and value:
                                    if isinstance(value, list) and len(value) > 0:
                                        study_type = value[0]
                                    else:
                                        study_type = str(value)
                                elif key.lower().startswith("organism") and value:
                                    if isinstance(value, list) and len(value) > 0:
                                        organism = value[0]
                                    else:
                                        organism = str(value)
                        
                        regular_datasets.append({
                            'title': f"PMID: {pmid} dataset from {file_basename}",
                            'accession': f'BULK-{pmid}',
                            'pmid': pmid,
                            'description': f"Entry from bulk dataset file: {file_basename}",
                            'organism': organism,
                            'study_type': study_type,
                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            'file_path': file_path,
                            'is_bulk': True,
                            'raw_data': data
                        })
                else:
                    # Just add one entry for the file itself
                    regular_datasets.append({
                        'title': f"Bulk dataset: {file_basename}",
                        'accession': 'BULK',
                        'pmid': 'bulk',
                        'description': f"Bulk dataset containing multiple entries",
                        'organism': 'Various',
                        'study_type': 'Various',
                        'url': "#",
                        'file_path': file_path,
                        'is_bulk': True
                    })
                
                continue
            except Exception as e:
                print(f"Error processing bulk file {file_path}: {str(e)}")
                # Continue to regular processing
        
        # Regular file processing for individual dataset files
        try:
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Try to parse as JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"Couldn't parse JSON from {file_path}")
                continue
            
            # If data is a string, try to parse it again (double-encoded)
            if isinstance(data, str):
                print(f"File content is a string, not a JSON object: {file_path}")
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    continue
                    
            # Extract basic metadata
            filename = os.path.basename(file_path)
            accession_match = re.search(r'(E-\w+-\d+)', filename)
            pmid = 'unknown'
            pmid_match = re.search(r'^(\d+)___', filename)
            if pmid_match:
                pmid = pmid_match.group(1)
            
            # Try to extract title from data
            title = None
            description = None
            organism = None
            study_type = None
            
            # Extract from sections based on data structure
            if isinstance(data, dict):
                # Try to find title
                if 'title' in data:
                    title = data['title']
                elif 'section' in data and isinstance(data['section'], list):
                    for section in data['section']:
                        if isinstance(section, dict) and 'type' in section and section['type'] == 'Study Title':
                            if 'attribute' in section and isinstance(section['attribute'], list):
                                for attr in section['attribute']:
                                    if isinstance(attr, dict) and 'name' in attr and attr['name'] == 'Title':
                                        if 'value' in attr:
                                            title = attr['value']
                
                # Try to find description
                if 'description' in data:
                    description = data['description']
                elif 'section' in data and isinstance(data['section'], list):
                    for section in data['section']:
                        if isinstance(section, dict) and 'type' in section and section['type'] == 'Study Description':
                            if 'attribute' in section and isinstance(section['attribute'], list):
                                for attr in section['attribute']:
                                    if isinstance(attr, dict) and 'name' in attr and attr['name'] == 'Description':
                                        if 'value' in attr:
                                            description = attr['value']
                
                # Try to extract other fields
                if 'organism' in data:
                    organism = data['organism']
                if 'study_type' in data:
                    study_type = data['study_type']
            
            # Default values if not found
            title = title or f"Dataset {accession_match.group(1) if accession_match else filename}"
            
            # Create a dataset info object 
            dataset_info = {
                'title': title,
                'accession': accession_match.group(1) if accession_match else 'Unknown',
                'pmid': pmid,
                'file_path': file_path,
                'description': description,
                'organism': organism,
                'study_type': study_type,
                'url': f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession_match.group(1)}" if accession_match else "#"
            }
            
            regular_datasets.append(dataset_info)
            
        except Exception as e:
            print(f"Error loading dataset from {file_path}: {str(e)}")
            continue
    
    # Only return regular datasets, no schemas
    return regular_datasets

def load_datasets_paged(dataset_files, page_number=0, per_page=50):
    """Load just a specific page of datasets"""
    if not dataset_files:
        return []
    
    # Calculate start and end indices
    start_idx = page_number * per_page
    end_idx = min(start_idx + per_page, len(dataset_files))
    
    # Get just the files for this page
    page_files = dataset_files[start_idx:end_idx]
    
    # Use existing function to load just these files
    return load_dataset_sample(page_files, max_samples=per_page)

def filter_datasets(datasets, search_term):
    """Filter datasets based on a search term."""
    if not search_term:
        return datasets
        
    search_term = search_term.lower()
    filtered = []
    
    for dataset in datasets:
        # Check if search term appears in any of these fields
        if (search_term in str(dataset.get('title', '')).lower() or
            search_term in str(dataset.get('description', '')).lower() or
            search_term in str(dataset.get('organism', '')).lower() or
            search_term in str(dataset.get('study_type', '')).lower() or
            search_term in str(dataset.get('accession', '')).lower() or
            search_term in str(dataset.get('pmid', '')).lower()):
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
        if dataset.get('pmid') and dataset['pmid'] != 'unknown' and dataset['pmid'] != 'schema' and dataset['pmid'] != 'bulk':
            # Only fetch if needed
            if (not dataset.get('title') or 
                dataset['title'].startswith("Dataset") or
                dataset['title'].startswith("PMID:")):
                pmids_to_fetch.append(dataset['pmid'])
                pmid_indices[dataset['pmid']] = i
    
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
            
            # Be nice to the PubMed API
            import time
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching batch of PubMed titles: {e}")
            continue
    
    return datasets

def generate_sample_datasets(count=5):
    """Generate sample datasets when no real data is available"""
    sample_data = []
    
    organisms = ["Homo sapiens", "Mus musculus", "Rattus norvegicus", "Drosophila melanogaster", "Saccharomyces cerevisiae"]
    study_types = ["RNA-seq", "ChIP-seq", "Microarray", "Proteomics", "Metabolomics"]
    
    for i in range(count):
        sample_data.append({
            'title': f"Sample Dataset {i+1}: {study_types[i % len(study_types)]} study of {organisms[i % len(organisms)]}",
            'accession': f"E-SAMPLE-{i+1}",
            'pmid': f"PMID{30000000 + i}",
            'description': f"This is a sample dataset for demonstration purposes. It contains {(i+1)*10} samples of {organisms[i % len(organisms)]} data.",
            'organism': organisms[i % len(organisms)],
            'study_type': study_types[i % len(study_types)],
            'url': "#",
            'is_sample': True
        })
    
    return sample_data

def show():
    st.title("Dataset Browser")
    st.write("Browse and select datasets for analysis")
    
    # Initialize session state for selected datasets
    if 'selected_datasets' not in st.session_state:
        st.session_state.selected_datasets = []
    
    # Initialize session state for pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'current_dataset_page' not in st.session_state:
        st.session_state.current_dataset_page = 0

    # Initialize session state for pagination properly
    if 'all_dataset_files' not in st.session_state:
        st.session_state.all_dataset_files = []
    if 'total_datasets' not in st.session_state:
        st.session_state.total_datasets = 0
    if 'current_dataset_page' not in st.session_state:
        st.session_state.current_dataset_page = 0
        
    # Default values for settings (moved to top of function)
    initial_display = 50
    datasets_per_page = 10
    show_all = True
    
    # Advanced settings expander (always show this)
    with st.expander("Advanced Settings", expanded=False):
        # Keep user friendly sample limit for initial display
        initial_display = st.slider("Number of datasets to display initially:", 
                            min_value=10, max_value=100, value=initial_display, step=10)
        
        # Add option to see all datasets
        show_all = st.checkbox("Enable full dataset navigation (1000+ papers)", value=show_all,
                            help="When enabled, you can navigate through all available papers")
        
        # Datasets per page control
        datasets_per_page = st.slider("Datasets per page:", 
                               min_value=5, max_value=50, value=datasets_per_page, step=5)
        
        st.info("For large datasets, papers will load one page at a time to improve performance.")
    
    # Demo mode checkbox - show at the top
    demo_mode = st.checkbox("Use demo datasets", value=False, 
                      help="Enable this option if you don't have dataset files available")
    
    # File uploader option
    uploaded_files = st.file_uploader("Or upload your own dataset JSON files", 
                               accept_multiple_files=True, type=["json"])
    
    # Process datasets based on source
    if demo_mode:
        st.info("Using demo datasets for demonstration purposes")
        st.session_state.datasets = generate_sample_datasets(10)
    elif uploaded_files:
        # Handle uploaded files
        st.session_state.datasets = []
        for uploaded_file in uploaded_files:
            try:
                dataset = json.loads(uploaded_file.getvalue())
                filename = uploaded_file.name
                accession_match = re.search(r'(E-\w+-\d+)', filename)
                
                dataset_info = {
                    'title': dataset.get('title', os.path.splitext(filename)[0]),
                    'accession': accession_match.group(1) if accession_match else 'User-uploaded',
                    'pmid': dataset.get('pmid', 'unknown'),
                    'description': dataset.get('description', 'User uploaded dataset'),
                    'organism': dataset.get('organism', 'Unknown'),
                    'study_type': dataset.get('study_type', 'Unknown'),
                    'url': "#",
                    'is_uploaded': True
                }
                st.session_state.datasets.append(dataset_info)
            except json.JSONDecodeError:
                st.error(f"Could not parse {uploaded_file.name} as valid JSON")
    elif 'datasets' not in st.session_state:
        # Load from cache or discover files
        cached_datasets = load_cached_datasets()
        
        # Always find ALL dataset files to ensure pagination works
        with st.spinner("Finding all available datasets..."):
            all_dataset_files = find_dataset_files()
            st.session_state.all_dataset_files = all_dataset_files
            st.session_state.total_datasets = len(all_dataset_files)
            
            if len(all_dataset_files) > 0:
                st.success(f"Found {len(all_dataset_files)} total dataset files")
        
        # If we have cached datasets, load those for initial display
        if cached_datasets:
            st.session_state.datasets = cached_datasets
            st.success(f"Loaded {len(cached_datasets)} datasets from cache")
            st.info(f"You can navigate through all {len(all_dataset_files)} datasets using the pagination controls")
        else:
            # No cache, so load initial datasets
            with st.spinner("Loading initial datasets..."):
                if not all_dataset_files:
                    st.warning("No dataset files found. Please use the demo mode or upload your own files.")
                    st.session_state.datasets = []
                else:
                    # Load just the first page worth of datasets
                    st.session_state.datasets = load_dataset_sample(all_dataset_files, max_samples=initial_display)
                    if st.session_state.datasets:
                        save_cached_datasets(st.session_state.datasets)
                        st.success(f"Loaded {len(st.session_state.datasets)} datasets (Page 1 of {math.ceil(len(all_dataset_files)/datasets_per_page)})")
                        
                        # Fetch publication titles for any PMIDs
                        with st.spinner("Fetching publication titles..."):
                            st.session_state.datasets = fetch_all_pubmed_titles(st.session_state.datasets)
                            # Update cache with fetched titles
                            save_cached_datasets(st.session_state.datasets)

    # Dataset search and filtering
    st.header("Browse Datasets")
    
    if not st.session_state.datasets:
        st.error("No datasets available. Please enable demo mode or upload dataset files.")
        return
        
    search_term = st.text_input("Search datasets:", "")
    filtered_datasets = filter_datasets(st.session_state.datasets, search_term)
    
    st.write(f"Found {len(filtered_datasets)} matching datasets")
    
    # Implement pagination for datasets
    total_pages = max(1, (len(filtered_datasets) + datasets_per_page - 1) // datasets_per_page)

    # Show full pagination if we have all dataset files
    if hasattr(st.session_state, 'all_dataset_files') and st.session_state.all_dataset_files and show_all:
        st.write(f"Showing page {st.session_state.current_dataset_page + 1} of {(st.session_state.total_datasets + datasets_per_page - 1) // datasets_per_page}")
        
        # Create better pagination controls
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First"):
                st.session_state.current_dataset_page = 0
                # Load the new page of datasets
                new_page_datasets = load_datasets_paged(
                    st.session_state.all_dataset_files,
                    st.session_state.current_dataset_page,
                    datasets_per_page
                )
                if new_page_datasets:
                    st.session_state.datasets = fetch_all_pubmed_titles(new_page_datasets)
                st.rerun()
                
        with col2:
            if st.button("◀️ Prev") and st.session_state.current_dataset_page > 0:
                st.session_state.current_dataset_page -= 1
                # Load the new page of datasets
                new_page_datasets = load_datasets_paged(
                    st.session_state.all_dataset_files,
                    st.session_state.current_dataset_page,
                    datasets_per_page
                )
                if new_page_datasets:
                    st.session_state.datasets = fetch_all_pubmed_titles(new_page_datasets)
                st.rerun()
        
        with col3:
            total_pages = (st.session_state.total_datasets + datasets_per_page - 1) // datasets_per_page
            page_number = st.number_input("Go to page", 
                                        min_value=1, 
                                        max_value=total_pages,
                                        value=st.session_state.current_dataset_page + 1)
            if page_number != st.session_state.current_dataset_page + 1:
                st.session_state.current_dataset_page = page_number - 1
                # Load the new page of datasets
                new_page_datasets = load_datasets_paged(
                    st.session_state.all_dataset_files,
                    st.session_state.current_dataset_page,
                    datasets_per_page
                )
                if new_page_datasets:
                    st.session_state.datasets = fetch_all_pubmed_titles(new_page_datasets)
                st.rerun()
        
        with col4:
            if st.button("Next ▶️") and st.session_state.current_dataset_page < total_pages - 1:
                st.session_state.current_dataset_page += 1
                # Load the new page of datasets
                new_page_datasets = load_datasets_paged(
                    st.session_state.all_dataset_files,
                    st.session_state.current_dataset_page,
                    datasets_per_page
                )
                if new_page_datasets:
                    st.session_state.datasets = fetch_all_pubmed_titles(new_page_datasets)
                st.rerun()
                
        with col5:
            if st.button("Last ⏭️"):
                st.session_state.current_dataset_page = total_pages - 1
                # Load the new page of datasets
                new_page_datasets = load_datasets_paged(
                    st.session_state.all_dataset_files,
                    st.session_state.current_dataset_page,
                    datasets_per_page
                )
                if new_page_datasets:
                    st.session_state.datasets = fetch_all_pubmed_titles(new_page_datasets)
                st.rerun()

    # Calculate page start and end
    start_idx = st.session_state.current_page * datasets_per_page
    end_idx = min(start_idx + datasets_per_page, len(filtered_datasets))

    # Display only the current page of datasets
    with st.expander("Available Datasets", expanded=True):
        if filtered_datasets:
            for i in range(start_idx, end_idx):
                dataset = filtered_datasets[i]
                
                # Create a container for each dataset
                with st.container():
                    # Add a subtle separator
                    if i > start_idx:
                        st.markdown("---")
                        
                    # Create columns for layout
                    col1, col2, col3 = st.columns([5, 4, 1])
                    
                    with col1:
                        # Title with appropriate icon
                        title = dataset.get('title') or f"Dataset {dataset.get('accession', '(unknown)')}"
                        
                        # Add icon based on type
                        if dataset.get('is_sample'):
                            icon = "📊"
                            title_suffix = "(Demo)"
                        elif dataset.get('is_uploaded'):
                            icon = "📤"
                            title_suffix = "(Uploaded)"
                        elif dataset.get('is_schema'):
                            icon = "📜"
                            title_suffix = "(Schema)"
                        elif dataset.get('is_bulk'):
                            icon = "📚"
                            title_suffix = "(Bulk Data)"
                        else:
                            icon = "🔬"
                            title_suffix = ""
                            
                        st.markdown(f"#### {icon} {title} {title_suffix}")
                        
                        # Show description with proper truncation
                        if dataset.get('description'):
                            desc = dataset['description']
                            if len(desc) > 100:
                                desc = desc[:100] + "..."
                            st.markdown(f"*{desc}*")
                    
                    with col2:
                        # Metadata with emoji icons
                        if dataset.get('organism') and dataset['organism'] != 'N/A':
                            st.markdown(f"🧬 **Organism:** {dataset['organism']}")
                        if dataset.get('study_type') and dataset['study_type'] != 'N/A' and dataset['study_type'] != 'Unknown':
                            st.markdown(f"🔬 **Study type:** {dataset['study_type']}")
                        
                        # Show accession ID
                        st.markdown(f"📋 **ID:** `{dataset.get('accession', 'Unknown')}`")
                        
                        # Show PMID if available
                        if dataset.get('pmid') and dataset['pmid'] not in ['schema', 'bulk', 'unknown']:
                            st.markdown(f"📝 **PMID:** `{dataset['pmid']}`")
                    
                    with col3:
                        # Selection checkbox
                        dataset_key = f"select_dataset_{i}"
                        if dataset_key not in st.session_state:
                            st.session_state[dataset_key] = False
                        st.session_state[dataset_key] = st.checkbox("Select", key=f"cb_{i}", value=st.session_state.get(dataset_key, False))
        else:
            st.info("No datasets match your search criteria.")
    
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
                # Add icon based on type
                if ds.get('is_sample'):
                    icon = "📊"
                elif ds.get('is_uploaded'):
                    icon = "📤"
                elif ds.get('is_schema'):
                    icon = "📜"
                elif ds.get('is_bulk'):
                    icon = "📚"
                else:
                    icon = "🔬"
                    
                title = ds.get('title') or f"Dataset {ds.get('accession', '(unknown)')}"
                if len(title) > 30:
                    title = title[:30] + "..."
                st.markdown(f"- {icon} **{title}**")
        
        # Center the button with columns
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("Ask Questions About Selected Datasets", use_container_width=True):
                # Navigate to Consumer Q&A page
                st.session_state.show_page = "Consumer Q&A"
                st.rerun()
    else:
        st.info("Please select at least one dataset to continue")

    # Add a debug section for advanced users
    with st.expander("Debug Information", expanded=False):
        st.write(f"Total datasets available: {len(st.session_state.datasets)}")
        
        # Show data directory locations
        st.subheader("Data directories")
        st.code("\n".join([
            os.path.join(project_root, "data"),
            "/mnt/data/upcast/data/arxpr",
            project_root
        ]))
        
        # Show cache info
        cache_file = os.path.join(project_root, 'cached_datasets.json')
        if os.path.exists(cache_file):
            st.write(f"Cache file exists: {cache_file} ({os.path.getsize(cache_file)} bytes)")
        else:
            st.write(f"No cache file found at: {cache_file}")