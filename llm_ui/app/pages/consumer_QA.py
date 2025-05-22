import streamlit as st
import json
import sys
import os
import glob
import random
from datetime import datetime
import requests
from xml.etree import ElementTree as ET
import time 
from typing import List
import re

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

Please provide a comprehensive, accurate answer based on the dataset information provided above and make sure to always give a PMID, pubmed URL or state your source in a apa7th style. 
"""

def _fetch_pubmed_context(pmid: str) -> List[str]:
    """Fetches abstract, publication details, authors, MeSH terms from PubMed."""
    if not pmid or not pmid.isdigit():
        return ["- PubMed Info: Invalid or missing PMID."]
    
    metadata = []
    try:
        pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
        response = requests.get(pubmed_url, timeout=10)
        response.raise_for_status() # Raise error for bad status codes
        
        root = ET.fromstring(response.text)
        article = root.find(".//PubmedArticle/MedlineCitation/Article")
        
        if article is None:
            return ["- PubMed Info: Could not find article details in XML."]

        metadata.append(f"\n### Publication Information (from PubMed):")

        # Extract article title (can be used to verify/complement dataset title)
        article_title = article.find("./ArticleTitle")
        if article_title is not None and article_title.text:
            metadata.append(f"Article Title: {article_title.text}")

        # Extract abstract text
        abstract_texts = root.findall(".//AbstractText")
        if abstract_texts:
            metadata.append(f"Abstract:")
            abstract_content = []
            for abstract_part in abstract_texts:
                label = abstract_part.get("Label")
                part_text = abstract_part.text or ""
                if label:
                    abstract_content.append(f"**{label}**: {part_text}")
                else:
                    abstract_content.append(part_text)
            metadata.append("\n".join(filter(None, abstract_content))) # Join non-empty parts

        # Extract publication date
        pub_date = root.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.find("Year")
            month = pub_date.find("Month")
            day = pub_date.find("Day")
            pub_date_str = ""
            if year is not None and year.text: pub_date_str += year.text
            if month is not None and month.text: pub_date_str += f"-{month.text}"
            if day is not None and day.text: pub_date_str += f"-{day.text}"
            if pub_date_str: metadata.append(f"Publication Date: {pub_date_str}")

        # Extract authors
        authors = root.findall(".//Author")
        if authors:
            author_names = []
            for author in authors[:5]: # Limit authors
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None and last_name.text:
                    author_name = last_name.text
                    if first_name is not None and first_name.text: author_name = f"{first_name.text} {author_name}"
                    author_names.append(author_name)
            if author_names:
                if len(authors) > 5: author_names.append("et al.")
                metadata.append(f"Authors: {', '.join(author_names)}")

        # Extract journal info
        journal = root.find(".//Journal/Title")
        if journal is not None and journal.text: metadata.append(f"Journal: {journal.text}")

        # Extract MeSH terms
        mesh_headings = root.findall(".//MeshHeadingList/MeshHeading") # Corrected path
        if mesh_headings:
            mesh_terms = []
            for heading in mesh_headings[:10]: # Limit terms
                descriptor = heading.find("./DescriptorName")
                if descriptor is not None and descriptor.text: mesh_terms.append(descriptor.text)
            if mesh_terms: metadata.append(f"MeSH Terms: {', '.join(mesh_terms)}")

    except requests.exceptions.RequestException as e:
    # specific about which URL might have failed if possible
        metadata.append(f"- ArrayExpress Info: Network/Request Error - {e}")
    except json.JSONDecodeError as e:
        metadata.append(f"- ArrayExpress Info: Error decoding API response (JSON) - {e}")
    except Exception as e:
        metadata.append(f"- ArrayExpress Info: Unexpected error during fetch - {type(e).__name__}: {e}")

    # Add a small delay
    time.sleep(0.5)
    return metadata




def _fetch_arrayexpress_context(accession: str) -> List[str]:
    """Fetches details from BioStudies/ArrayExpress APIs, targeting specific fields."""
    if not accession:
        return ["- ArrayExpress Info: Missing Accession."]

    metadata = []
    found_details = set() # Keep track of details found to avoid redundancy

    # Try BioStudies API 
    try:
        ae_url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}"
        print(f"DEBUG: Requesting BioStudies API: {ae_url}")
        response = requests.get(ae_url, timeout=15)
        print(f"DEBUG: BioStudies Response Status: {response.status_code}")
        response.raise_for_status()
        ae_data = response.json()
        metadata.append(f"\n### ArrayExpress Dataset Information (BioStudies API):")

        section = ae_data.get('section', {})
        attributes = section.get('attributes', [])
        if attributes:
            for attr in attributes:
                if isinstance(attr, dict) and 'name' in attr and 'value' in attr:
                    name = attr['name'].lower() # Use lower case for matching
                    value = str(attr['value'])
                    name_cap = attr['name'].capitalize()

                    # Explicitly look for desired fields (adjust 'name' based on API output)
                    if 'organism part' in name and 'organism_part' not in found_details:
                        metadata.append(f"Organism Part: {value}")
                        found_details.add('organism_part')
                    elif 'experimental design' in name and 'design' not in found_details:
                        metadata.append(f"Experimental Design: {value}")
                        found_details.add('design')
                    elif 'assay' in name or 'measurement' in name and 'assay' not in found_details:
                         metadata.append(f"Assay Type: {value}")
                         found_details.add('assay')
                    #  other less critical but potentially useful attributes
                    elif name not in ['title', 'description', 'name', 'release_date', 'accno', 'organism']: # Avoid redundancy
                         if len(value) > 100: value = value[:100] + "..."
                         metadata.append(f"- {name_cap}: {value}") # other attributes with filtering

        else:
             metadata.append("- No attributes found in BioStudies study section.")

    # Catch specific exceptions
    except requests.exceptions.Timeout: metadata.append(f"- AE/BioStudies Info: Timeout ({ae_url})")
    except requests.exceptions.HTTPError as e: metadata.append(f"- AE/BioStudies Info: HTTP Error ({ae_url}): {e}")
    except requests.exceptions.RequestException as e: metadata.append(f"- AE/BioStudies Info: Request Error ({ae_url}): {e}")
    except json.JSONDecodeError: metadata.append(f"- AE/BioStudies Info: Error decoding JSON (Status: {response.status_code if 'response' in locals() else 'N/A'})")
    except Exception as e: metadata.append(f"- AE/BioStudies Info: Unexpected error - {type(e).__name__}: {e}")


    # Try AE Legacy API 
    try:
        legacy_url = f"https://www.ebi.ac.uk/arrayexpress/json/v3/experiments/{accession}"
        print(f"DEBUG: Requesting AE Legacy API: {legacy_url}")
        response_legacy = requests.get(legacy_url, timeout=15)
        print(f"DEBUG: AE Legacy Response Status: {response_legacy.status_code}")
        response_legacy.raise_for_status()
        legacy_data = response_legacy.json()

        if 'experiments' in legacy_data and 'experiment' in legacy_data['experiments'] and legacy_data['experiments']['experiment']:
            exp = legacy_data['experiments']['experiment'][0]
            metadata.append(f"\n### ArrayExpress Detailed Information (Legacy API):")

            # Explicitly look for desired fields if not already found
            if 'experimenttype' in exp and 'study_type' not in found_details:
                 metadata.append(f"Experiment Type: {exp['experimenttype']}")
                 found_details.add('study_type')
            if 'organism' in exp and 'organism' not in found_details:
                 metadata.append(f"Organism: {exp['organism']}")
                 found_details.add('organism')
            # Hardware might be in 'performer' or within protocols
            if 'performer' in exp and 'hardware' not in found_details:
                 metadata.append(f"Performer/Hardware: {exp['performer']}")
                 found_details.add('hardware')

            # Look inside protocols for hardware/platform info
            if 'protocol' in exp and exp['protocol']:
                 protocol_hardware = []
                 for p in exp['protocol']:
                     # Check common keys within protocols
                     hw = p.get('hardware') or p.get('instrument') or p.get('platform')
                     if hw: protocol_hardware.append(f"{p.get('type','Protocol')} uses {hw}")
                 if protocol_hardware and 'hardware' not in found_details:
                      metadata.append(f"Protocol Hardware: {'; '.join(protocol_hardware)}")
                      found_details.add('hardware') # Mark as found if extracted here

            # Look in sample attributes for organism part
            if 'sampleattribute' in exp and exp['sampleattribute'] and 'organism_part' not in found_details:
                 parts = []
                 for sa in exp['sampleattribute']:
                     cat = sa.get('category','').lower()
                     if 'organism part' in cat or 'tissue' in cat:
                          parts.append(sa.get('value'))
                 unique_parts = list(set(filter(None, parts))) # Get unique, non-empty parts
                 if unique_parts:
                      metadata.append(f"Sample Organism Part(s): {', '.join(unique_parts)}")
                      found_details.add('organism_part')

            # Look for Experimental Factors
            if 'experimentalfactor' in exp and exp['experimentalfactor'] and 'design' not in found_details:
                 factors = [f.get('name') for f in exp['experimentalfactor'] if f.get('name')]
                 if factors:
                      metadata.append(f"Experimental Factors: {', '.join(factors)}")
                      found_details.add('design') # Mark design as found if factors exist

    # Catch specific exceptions
    except requests.exceptions.Timeout: metadata.append(f"- AE/Legacy Info: Timeout ({legacy_url})")
    except requests.exceptions.HTTPError as http_err:
        status_code_legacy = response_legacy.status_code if 'response_legacy' in locals() else 'N/A'
        if status_code_legacy == 404: metadata.append(f"- AE/Legacy Info: Dataset not found ({legacy_url}).")
        else: metadata.append(f"- AE/Legacy Info: HTTP Error ({legacy_url}, Status: {status_code_legacy}): {http_err}")
    except requests.exceptions.RequestException as req_err: metadata.append(f"- AE/Legacy Info: Request Error ({legacy_url}): {req_err}")
    except json.JSONDecodeError:
        # Silently ignore JSON parsing errors for the legacy API
        pass
    except Exception as e:
        # Silently ignore other unexpected errors during legacy API processing
        # print(f"[WARN] Unexpected error processing legacy AE data for {accession}: {e}") # Keep if debugging needed
        pass


    # Try to fetch IDF/SDRF snippets 
    base_file_url = f"https://www.ebi.ac.uk/arrayexpress/files/{accession}"

    # Fetch IDF snippet
    try:
        idf_url = f"{base_file_url}/{accession}.idf.txt"
        idf_response = requests.get(idf_url, timeout=10)
        if idf_response.status_code == 200:
            idf_content = idf_response.text
            metadata.append(f"\n### IDF Snippet (Experiment Design):")
            lines = idf_content.strip().split('\n')[:30] # First 30 lines
            metadata.extend(lines)
            if len(idf_content.strip().split('\n')) > 30:
                 metadata.append("...") # Indicate truncation
    except Exception as e:
        # print(f"[DEBUG] Failed to fetch/process IDF for {accession}: {e}") # Optional debug
        pass # Silently ignore errors fetching IDF

    # Fetch SDRF snippet
    try:
        sdrf_url = f"{base_file_url}/{accession}.sdrf.txt"
        sdrf_response = requests.get(sdrf_url, timeout=10)
        if sdrf_response.status_code == 200:
            sdrf_content = sdrf_response.text
            metadata.append(f"\n### SDRF Snippet (Sample Details):")
            lines = sdrf_content.strip().split('\n')[:15] # First 15 lines
            # Truncate long lines within the snippet
            truncated_lines = [(line[:200] + '...' if len(line) > 200 else line) for line in lines]
            metadata.extend(truncated_lines)
            if len(sdrf_content.strip().split('\n')) > 10:
                 metadata.append("...") # Indicate truncation
    except Exception as e:
        # print(f"[DEBUG] Failed to fetch/process SDRF for {accession}: {e}") # Optional debug
        pass # Silently ignore errors fetching SDRF

    time.sleep(0.5) # Keep existing delay
    if not metadata or all("Info:" in s or s.startswith("###") for s in metadata):
        # Check if ONLY info/error messages were added
        meaningful_content = any(not s.startswith(("- ", "###", "\n###")) for s in metadata)
        if not meaningful_content:
             return ["- ArrayExpress Info: Could not retrieve significant details from EBI APIs or files."]
    return metadata


def _fetch_europepmc_context(pmid: str) -> List[str]:
    """Fetches additional details like PubTypes, Grants, Chemicals from EuropePMC."""
    if not pmid or not pmid.isdigit():
        return ["- EuropePMC Info: Invalid or missing PMID."]

    metadata = []
    try:
        europepmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&resultType=core&format=json"
        response = requests.get(europepmc_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'resultList' in data and 'result' in data['resultList'] and data['resultList']['result']:
            article = data['resultList']['result'][0]
            metadata.append(f"\n### Additional Publication Details (EuropePMC):")

            # Get publication type
            if 'pubTypeList' in article and 'pubType' in article['pubTypeList']:
                pub_types = ", ".join(article['pubTypeList']['pubType'])
                metadata.append(f"Publication Type: {pub_types}")

            # Get grants (limit)
            if 'grantsList' in article and 'grant' in article['grantsList']:
                grants = [f"{g.get('agency', 'N/A')} ({g.get('grantId', 'N/A')})"
                          for g in article['grantsList']['grant'][:3] if g] # Limit grants
                if grants: metadata.append(f"Funding (Examples): {', '.join(grants)}")

            # Get chemicals (limit)
            if 'chemicalList' in article and 'chemical' in article['chemicalList']:
                chemicals = [c.get('name', 'N/A') for c in article['chemicalList']['chemical'][:5] if c] # Limit chemicals
                if chemicals: metadata.append(f"Chemicals/Substances: {', '.join(chemicals)}")

    except requests.exceptions.RequestException as e:
    # specific about which URL might have failed if possible
        metadata.append(f"- ArrayExpress Info: Network/Request Error - {e}")
    except json.JSONDecodeError as e:
        metadata.append(f"- ArrayExpress Info: Error decoding API response (JSON) - {e}")
    except Exception as e:
        metadata.append(f"- ArrayExpress Info: Unexpected error during fetch - {type(e).__name__}: {e}")

    # Adding a small delay
    time.sleep(0.5)
    return metadata

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
    """Extract and format metadata. TEST VERSION: Simplified context for non-user."""

    metadata_groups = {'overview': [], 'publication': [], 'local_details': [], 'api_details': [], 'user_specific': []}
    source = dataset.get('source', 'unknown')
    pmid = dataset.get('pmid')
    accession = dataset.get('accession')

    #  Group 1: Overview 
    metadata_groups['overview'].append(f"### Dataset Overview: {dataset.get('title', 'N/A')} (Accession: {accession if accession else 'N/A'})")
    metadata_groups['overview'].append(f"- Source Type: {source}")
    metadata_groups['overview'].append(f"- Provided PMID: {pmid if pmid else 'N/A'}")
    # Add Organism and Study Type here as key overview points from local DB
    if dataset.get('organism'): metadata_groups['overview'].append(f"- Organism: {dataset.get('organism')}")
    if dataset.get('study_type'): metadata_groups['overview'].append(f"- Study Type: {dataset.get('study_type')}")


    #  Group 2: Combined Details (Local DB + Publication via PMID) 
    details_list = [] # Temporary list for this group

    # Add Local DB fields first 
    local_fields_to_add = [
        'organism_part', 'experimental_designs', 'assay_by_molecule',
        'hardware', 'technology'
    ]
    found_local = False
    for field_key in local_fields_to_add:
         value = dataset.get(field_key)
         if value:
             details_list.append(f"- {field_key.replace('_', ' ').capitalize()}: {value}")
             found_local = True
    if not found_local and source == 'user_provider': # Only show for user if nothing else found
        details_list.append("- No additional specific local metadata details found.")
    # Assign collected details to the group 
    metadata_groups['local_details'] = details_list


    # Fetch and add Publication details if valid PMID exists
    valid_pmid = pmid and pmid != 'NO_PMID' and str(pmid).strip().isdigit()
    #print(f"DEBUG consumer_QA: Processing Accession: {accession}, Source: {source}") 
    #print(f"DEBUG consumer_QA: Raw PMID value from dataset dict: '{pmid}' (Type: {type(pmid)})")
    print(f"DEBUG consumer_QA: PMID '{pmid}' validity check result: {valid_pmid}")

    if valid_pmid:
        clean_pmid = str(pmid).strip()
        with st.spinner(f"Fetching publication data for PMID: {clean_pmid}..."):
            pubmed_context = _fetch_pubmed_context(clean_pmid)
            europepmc_context = _fetch_europepmc_context(clean_pmid)

            pub_details_added = False
            if len(pubmed_context) > 1 or (len(pubmed_context) == 1 and "Error" not in pubmed_context[0]):
                 
                 details_list.append("\n**Publication Info (from PubMed):**") # Optional header
                 for item in pubmed_context:
                      
                      if item.startswith("Abstract:"):
                           # a more distinct label for the abstract
                           abstract_text = item.replace("Abstract:", "").strip()
                           details_list.append(f"**Fetched Publication Abstract:** {abstract_text}") # New Label
                      elif item.startswith("MeSH Terms:") or item.startswith("Article Title:") or item.startswith("Journal:"):
                          # other key PubMed info, removing helper's header
                          details_list.append(item.replace('\n### Publication Information (from PubMed):','').strip())
                      
                 pub_details_added = True

            if not pub_details_added:
                 details_list.append("- No detailed publication information could be retrieved via PMID.")

            if not pub_details_added:
                 metadata_groups['publication'].append("- No detailed publication information could be retrieved via PMID.")
    else:
         metadata_groups['publication'].append("- Publication Info: No valid PMID provided.")


    # Group 3: User Specific Data (Profiler JSON, Full Text) 
    # This block only runs for user_provider source
    if source == 'user_provider':
        #  Read JSON and extract context snippets 
        user_file_path = dataset.get('file_path')
        if user_file_path and os.path.exists(user_file_path):
            try:
                with open(user_file_path, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)

                # Locate the context dictionary (assuming same structure as before)
                context_data = None
                if isinstance(user_data, dict):
                    if "0" in user_data and isinstance(user_data["0"], dict) and "context" in user_data["0"]:
                        context_data = user_data["0"].get("context")
                    elif "context" in user_data and isinstance(user_data["context"], dict):
                        context_data = user_data.get("context")

                if isinstance(context_data, dict):
                    metadata_groups['user_specific'].append("\n### Supporting Context Snippets:")
                    added_snippets = 0
                    for context_key, snippet in context_data.items():
                         if isinstance(snippet, str) and snippet.strip():
                              # Clean up key name (remove suffix) for display
                              base_key_display = re.sub(r'_\d+$', '', context_key).replace('_', ' ').capitalize()
                              # Limit snippet length for prompt
                              display_snippet = snippet[:300] + '...' if len(snippet) > 300 else snippet
                              metadata_groups['user_specific'].append(f"**{base_key_display}:** {display_snippet}")
                              added_snippets += 1
                    if added_snippets == 0:
                         metadata_groups['user_specific'].append("- No context snippets found in the saved JSON file.")
                else:
                    metadata_groups['user_specific'].append("- Could not locate context snippets in the saved JSON file.")

            except json.JSONDecodeError:
                metadata_groups['user_specific'].append(f"- Error: Could not parse the JSON file: {os.path.basename(user_file_path)}")
            except Exception as e:
                metadata_groups['user_specific'].append(f"- Error reading user data file {os.path.basename(user_file_path)}: {e}")
        else:
            metadata_groups['user_specific'].append("- Associated user JSON file not found or path missing.")
        


    #  Group 4: ArrayExpress API Details (if applicable) 
    # (Will be skipped for non-user in TEST below)
    is_arrayexpress_source = source in ['arrayexpress', 'bulk_processed']
    looks_like_ae_acc = accession and re.match(r"E-\w{4}-\d+", accession)
    if accession and (is_arrayexpress_source or looks_like_ae_acc):
         with st.spinner(f"Fetching ArrayExpress API data for Accession: {accession}..."):
            ae_context = _fetch_arrayexpress_context(accession) # Uses the refined helper
            if len(ae_context) > 1 or (len(ae_context) == 1 and "Error" not in ae_context[0] and "Info:" not in ae_context[0]):
                 metadata_groups['api_details'].extend([s for s in ae_context if not s.startswith('\n###')])
            else:
                 metadata_groups['api_details'].append("- Could not retrieve significant details from ArrayExpress APIs.")


    # Assemble Final Context String 
    final_metadata = []
    # Always add overview
    if metadata_groups['overview']:
        final_metadata.extend(metadata_groups['overview'])

        # Assemble Final Context String (Original Assembly Logic) 
    final_metadata = []
    if metadata_groups['overview']:
        final_metadata.extend(metadata_groups['overview'])
    if metadata_groups['publication']: # Combined local + publication
        final_metadata.append("\n### Publication & Local Details") # Combine headers
        final_metadata.extend(metadata_groups['publication'])
    # Add local details only if not already covered by publication group
    if metadata_groups['local_details']:
         final_metadata.append("\n### Locally Stored Details")
         final_metadata.extend(metadata_groups['local_details'])
    if metadata_groups['api_details']:
        final_metadata.append("\n### ArrayExpress API Specific Details")
        final_metadata.extend(metadata_groups['api_details'])
    if metadata_groups['user_specific']: # Append user specific section last
         final_metadata.append("\n### User Upload Specific Information")
         final_metadata.extend(metadata_groups['user_specific'])
    


    # DEBUG 
    print(f"\nDEBUG consumer_QA: FINAL metadata list for Accession {accession} BEFORE JOIN (REVISED LOGIC ACTIVE):")
    # Print each group that will be included for debugging
    if metadata_groups['overview']: print("  - Overview Included")
    if source == 'user_provider':
        if metadata_groups['publication']: print("  - Publication Included (User)")
        if metadata_groups['local_details']: print("  - Local Details Included (User)")
        if metadata_groups['user_specific']: print("  - User Specific Included (User)")
        # if metadata_groups['api_details']: print("  - API Details Included (User - Optional)")
    else:
        if metadata_groups['publication']: print("  - Publication Included (Non-User)")
        if metadata_groups['local_details']: print("  - Local Details Included (Non-User)")
        if metadata_groups['api_details']: print("  - API Details Included (Non-User)")
    


    return "\n".join(final_metadata)

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

    # Chat Display Area 
    st.markdown("<h3>Chat</h3>", unsafe_allow_html=True)
    # container for the chat messages
    chat_display_container = st.container()
    # chat-container style to this container specifically

    chat_display_container.markdown("<div class='chat-container' id='chat-container-div'>", unsafe_allow_html=True) # Start the styled div

    with chat_display_container:
        if not st.session_state.chat_history:
            st.markdown("<div style='text-align: center; color: #888; padding: 20px;'>Ask a question about the selected datasets to get started</div>", unsafe_allow_html=True)
        else:
            for i, (timestamp, question_text, answer_text) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class='message-container' style='align-items: flex-end; margin-bottom: 15px;'>
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
        st.rerun()
    
    # Clear entire chat history if requested
    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.chat_history = []
        st.rerun()
    
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
            prompt = f"""You are an AI assistant specializing in biomedical research datasets. Your task is to answer the question based *only* on the detailed context provided below.

CONTEXT START ===
### Overall Request
TITLE: {title}

QUESTION: {question}

### Detailed Dataset Information Provided
{formatted_dataset_text}
=== CONTEXT END

INSTRUCTIONS:
1.  Assume the role of an expert AI assistant interpreting this data for a PhD-level scientist.
2.  Answer the QUESTION by synthesizing information from ALL provided sections (Overview, Publication, Local DB, API). its okay to list some data points but not always. Pay close attention to experimental details like factors, organism part, and assay type.
3.  Based *mostly* on the provided metadata (especially experimental design, assay type, organism, and factors), briefly explain the likely scientific question or goal addressed by this study.
4.  Do not invent information.
5.  Provide a comprehensive, accurate, and insightful answer suitable for an expert audience. Try to always include a PMID, URL or use the APA7th style for references or citation. 

Answer:
"""


            
            # Show debug information
            with st.expander("Debug: Prompt being sent to LLM", expanded=False):
                st.text(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
                st.text(f"Total prompt length: {len(prompt)} characters")

            # Get the response using the LLM instance, which now loads its own settings.
            # Pass None for max_tokens and temperature to use the settings loaded by the LLM class.
            output = st.session_state.llm.ask(
                prompt,
                max_tokens=None, # Use settings loaded by LLM class
                temperature=None # Use settings loaded by LLM class
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
