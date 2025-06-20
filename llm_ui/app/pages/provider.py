import streamlit as st
import os
import sys
import json
import requests
import re
import tempfile
import pydantic
from pydantic import create_model, Field
from typing import Optional, Any, List, Dict, Union
from urllib.parse import urlparse
import shutil # For cleaning up temp dirs
import xml.etree.ElementTree as ET # For parsing XML

# Add project root and profiler directory to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)
profiler_dir = os.path.join(project_root, 'profiler')
sys.path.append(profiler_dir)
data_dir = os.path.join(project_root, 'data')
sys.path.append(data_dir)

# Attempt Imports with Error Handling 
# This function will import dependencies and handle errors
def import_dependencies():
    dependencies = {
        "call_inference": None, "save_xml": None, "init_db": None,
        "upsert_dataset": None # Removed "partition": None and "arxpr2_schema": None
    }
    import_errors = []

    try:
        from profiler.run_inference import call_inference
        dependencies["call_inference"] = call_inference
    except ImportError as e: import_errors.append(f"call_inference: {e}")
    try:
        from data.xml_fetcher import save_xml
        dependencies["save_xml"] = save_xml
    except ImportError as e: import_errors.append(f"save_xml: {e}")
    try:
        from llm_ui.app.db_utils import init_db, upsert_dataset
        dependencies["init_db"] = init_db
        dependencies["upsert_dataset"] = upsert_dataset
    except ImportError as e: import_errors.append(f"db_utils: {e}")
    try:
        # Try importing partition here, but don't store globally
        from unstructured.partition.auto import partition
        dependencies["partition"] = partition # Store function ref if successful
    except ImportError:
        import_errors.append("unstructured (required for PDF processing)")
        dependencies["partition"] = None # Explicitly set to None if import fails

    return dependencies, import_errors

# --- Helper Functions ---


def _try_extract_pmid(file_path: str) -> Optional[str]:
    """Attempts to extract a PMID from XML/HTML or text content."""
    if not file_path or not os.path.exists(file_path):
        print(f"[_try_extract_pmid] File path invalid or file does not exist: {file_path}")
        return None

    pmid = None
    content = ""
    file_processed_for_content = False

    # 1. Try parsing as XML
    if file_path.lower().endswith(".xml"):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            # Look for <article-id pub-id-type="pmid">...</article-id>
            # More robust search using XPath, ignoring namespaces
            pmid_elem = root.find(".//{*}article-id[@pub-id-type='pmid']")
            if pmid_elem is not None and pmid_elem.text and pmid_elem.text.strip().isdigit():
                pmid = pmid_elem.text.strip()
                print(f"[_try_extract_pmid] Found PMID {pmid} via XML tag <article-id>.")
                return pmid # Found via specific XML tag, return immediately

            # Look for <PMID>...</PMID> (simpler structure)
            # More robust search using XPath, ignoring namespaces
            pmid_elem = root.find(".//{*}PMID")
            if pmid_elem is not None and pmid_elem.text and pmid_elem.text.strip().isdigit():
                pmid = pmid_elem.text.strip()
                print(f"[_try_extract_pmid] Found PMID {pmid} via XML tag <PMID>.")
                return pmid # Found via specific XML tag, return immediately

            # Look for <infon key="article-id_pmid">...</infon>
            # More robust search using XPath, ignoring namespaces
            pmid_elem = root.find(".//{*}infon[@key='article-id_pmid']")
            if pmid_elem is not None and pmid_elem.text and pmid_elem.text.strip().isdigit():
                pmid = pmid_elem.text.strip()
                print(f"[_try_extract_pmid] Found PMID {pmid} via XML tag <infon key='article-id_pmid'>.")
                return pmid # Found via specific XML tag, return immediately

            # If specific tags not found, prepare to read content for regex
            print(f"[_try_extract_pmid] XML parsed but no specific PMID tag found (incl. infon) in {os.path.basename(file_path)}. Will check content via regex.")

        except ET.ParseError:
            print(f"[_try_extract_pmid] File {os.path.basename(file_path)} is not valid XML. Will check content via regex.")
            # Proceed to read content below
        except Exception as xml_err:
            print(f"[_try_extract_pmid] Error processing XML {os.path.basename(file_path)}: {xml_err}. Will check content via regex.")
            

    # 2. Read content (if not found via XML tags or if not XML) for regex check
    # Avoid reading again if XML parsing failed but reading succeeded below
    if not pmid:
        try:
            # Read beginning of file, increase size slightly
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(15000)
            file_processed_for_content = True
        except Exception as read_err:
            print(f"[_try_extract_pmid] Error reading file {os.path.basename(file_path)} for regex: {read_err}")
            # Can't proceed if reading fails
            return None

    # 3. Try regex on content (if read successfully)
    regex_found_pmid = False
    if file_processed_for_content and content:
        # Regex for PMID: optional space/colon/hyphen, 7-9 digits, word boundaries, case-insensitive
        # Added flexibility for variations like PubMed ID
        match = re.search(r'\b(?:PMID|PubMed\s*ID)\s*[:\-]?\s*(\d{7,9})\b', content, re.IGNORECASE)
        if match and match.group(1):
            pmid = match.group(1)
            print(f"[_try_extract_pmid] Found PMID {pmid} via regex.")
            regex_found_pmid = True
            return pmid # Found via regex

    # 4. Fallback: Try parsing content as XML/HTML if regex failed and it wasn't an XML file initially
    if file_processed_for_content and content and not regex_found_pmid and not file_path.lower().endswith(".xml"):
        print(f"[_try_extract_pmid] Regex failed for non-XML file {os.path.basename(file_path)}. Attempting content parsing...")
        try:
            # Attempt to parse the string content directly
            # Ensure content is treated as a string, not bytes
            if isinstance(content, bytes):
                 content_str = content.decode('utf-8', errors='ignore')
            else:
                 content_str = str(content) # Ensure it's a string

            # Basic check for XML/HTML structure before attempting full parse
            if '<' in content_str and '>' in content_str:
                 root = ET.fromstring(content_str) # Parse from string

                 # Reuse XML search logic
                 pmid_elem = root.find(".//{*}article-id[@pub-id-type='pmid']")
                 if pmid_elem is not None and pmid_elem.text and pmid_elem.text.strip().isdigit():
                     pmid = pmid_elem.text.strip()
                     print(f"[_try_extract_pmid] Found PMID {pmid} via content parsing <article-id>.")
                     return pmid

                 pmid_elem = root.find(".//{*}PMID")
                 if pmid_elem is not None and pmid_elem.text and pmid_elem.text.strip().isdigit():
                     pmid = pmid_elem.text.strip()
                     print(f"[_try_extract_pmid] Found PMID {pmid} via content parsing <PMID>.")
                     return pmid

                 pmid_elem = root.find(".//{*}infon[@key='article-id_pmid']")
                 if pmid_elem is not None and pmid_elem.text and pmid_elem.text.strip().isdigit():
                     pmid = pmid_elem.text.strip()
                     print(f"[_try_extract_pmid] Found PMID {pmid} via content parsing <infon key='article-id_pmid'>.")
                     return pmid
                 print(f"[_try_extract_pmid] Content parsing attempted but no specific PMID tag found.")
            else:
                 print(f"[_try_extract_pmid] Content does not appear to contain XML/HTML tags. Skipping content parsing.")

        except ET.ParseError:
            print(f"[_try_extract_pmid] Content parsing failed (not valid XML/HTML fragment).")
        except Exception as content_parse_err:
            print(f"[_try_extract_pmid] Error during content parsing: {content_parse_err}")

    # 5. If no PMID found after all checks
    print(f"[_try_extract_pmid] No PMID found in {os.path.basename(file_path)} after all checks.")
    return None

def is_pubmed_id(input_string):
    """Check if the input string looks like a PubMed ID (numeric)."""
    return input_string.strip().isdigit()

def fetch_xml_from_pubmed(pmid, temp_dir, save_xml_func):
    """Fetches XML for a PubMed ID and saves it to a temporary file."""
    if save_xml_func is None:
        st.error("XML fetching function (save_xml) is not available due to import error.")
        return None
    try:
        st.info(f"Attempting to fetch PubMed ID {pmid} using save_xml into {temp_dir}...")
        success = save_xml_func(pmid, folder=temp_dir)
        if success:
             possible_encodings = ["ascii", "utf-8"]
             possible_sources = ["pmcoa", "pubmed"]
             saved_path = None
             # Check specific known patterns first
             for enc in possible_encodings:
                 for src in possible_sources:
                     fname = f"{pmid}_{enc}_{src}.xml"
                     potential_path = os.path.join(temp_dir, fname)
                     if os.path.exists(potential_path):
                         saved_path = potential_path
                         st.info(f"Found saved PubMed XML: {saved_path}")
                         return saved_path
             # Fallback: check for any XML file starting with the PMID
             for item in os.listdir(temp_dir):
                 if item.startswith(str(pmid)) and item.lower().endswith(".xml"):
                     saved_path = os.path.join(temp_dir, item)
                     st.info(f"Found saved PubMed XML (fallback): {saved_path}")
                     return saved_path
             st.error(f"save_xml reported success, but couldn't find the saved file for {pmid} in {temp_dir}.")
             return None
        else:
            st.error(f"Failed to fetch XML for PubMed ID {pmid} using save_xml.")
            return None
    except Exception as e:
        st.error(f"Error fetching PubMed XML for ID {pmid}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def fetch_xml_from_url(url, temp_dir):
    """Fetches XML/HTML from a direct URL and saves it to a temporary file."""
    try:
        st.info(f"Attempting to download content from URL: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, timeout=30, headers=headers, allow_redirects=True) # Increased timeout, added headers, allow redirects
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        # Be more lenient with content type checking
        if not any(ct in content_type for ct in ['xml', 'html', 'text/plain']):
             st.warning(f"URL content type is '{content_type}'. Expected XML, HTML, or plain text. Proceeding anyway.")

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        # Improve filename generation
        if not filename or '.' not in filename[-5:]: # If no name or no common extension
             # Try to get name from content disposition header
             cd = response.headers.get('content-disposition')
             if cd:
                  fname = re.findall('filename="?(.+)"?', cd)
                  if fname:
                       filename = fname[0]
             if not filename or '.' not in filename[-5:]: # Fallback if still no good name
                  ext = ".xml" # Default assumption
                  if 'html' in content_type: ext = ".html"
                  elif 'text/plain' in content_type: ext = ".txt"
                  filename = f"downloaded_{hash(url)}{ext}"

        filename = re.sub(r'[^\w\-_\.]', '_', filename) # Sanitize
        temp_file_path = os.path.join(temp_dir, filename)

        with open(temp_file_path, 'wb') as f:
            f.write(response.content)
        st.info(f"Successfully downloaded content from URL to {temp_file_path}")
        return temp_file_path
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching from URL {url}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching from URL {url}: {e}")
        return None

def extract_text_from_file(input_path, temp_dir, partition_func):
    """Extracts text from PDF or other file types using unstructured and saves it."""
    if partition_func is None: # Check if partition function was successfully imported
        st.error("Text extraction is disabled because 'unstructured' library is not available or failed to import.")
        return None
    try:
        filename = os.path.basename(input_path)
        st.info(f"Extracting content from: {filename} using 'unstructured'...")
        # Using partition which handles various types including PDF, XML, HTML, TXT
        elements = partition_func(filename=input_path) # Use the passed function
        extracted_text = "\n\n".join([str(el.text) for el in elements]) # Use el.text for cleaner output

        if not extracted_text.strip():
             st.warning(f"No text content could be extracted from {filename}.")
             return None

        # Save extracted text to a file (useful for consistency or if pipeline needs text file)
        txt_filename = os.path.splitext(filename)[0] + "_extracted.txt"
        txt_path = os.path.join(temp_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        st.info(f"Successfully extracted text content and saved to {txt_path}")
        # Return the path to the extracted text file.
        return txt_path
    except Exception as e:
        st.error(f"Error extracting content from {os.path.basename(input_path)}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

@st.cache_resource # Cache the default schema model class (use resource for non-serializable types)
def get_default_schema_model():
     """Loads the default Pydantic schema model class."""
     try:
         # Import directly inside the cached function
         from profiler.metadata_schemas import arxpr2_schema
         # Check the imported module directly
         if hasattr(arxpr2_schema, 'Metadata_form'): # Corrected: Check the imported module
             return arxpr2_schema.Metadata_form
         else:
             st.error("Default schema module loaded, but 'Metadata_form' class is missing.")
             return None
     except ImportError as e:
         st.error(f"Default schema (profiler.metadata_schemas.arxpr2_schema) could not be imported: {e}")
         return None

@st.cache_resource # Cache model creation based on schema content hash (use resource for non-serializable types)
def create_pydantic_model_from_schema(schema_content: str, model_name: str = "CustomSchemaModel") -> Optional[type[pydantic.BaseModel]]:
    """Dynamically creates a Pydantic model from a JSON schema string."""
    fields = {}
    try:
        schema_dict = json.loads(schema_content)
        properties = schema_dict.get("properties", {})
        required = schema_dict.get("required", [])

        # More robust type mapping, including handling potential 'null' type
        def map_type(prop_schema):
            type_val = prop_schema.get("type")
            if isinstance(type_val, list): # Handle ["type", "null"]
                non_null_type = next((t for t in type_val if t != "null"), None)
                is_optional = "null" in type_val
                type_str = non_null_type
            else:
                type_str = type_val
                is_optional = False # Determined later by 'required' list

            base_type_mapping = {
                "string": str, "number": float, "integer": int,
                "boolean": bool, "array": List, "object": Dict, None: Any
            }
            base_type = base_type_mapping.get(type_str, Any)

            if type_str == "array":
                items_schema = prop_schema.get("items", {})
                item_type, _ = map_type(items_schema) # Recursive call for item type
                final_type = List[item_type]
            else:
                final_type = base_type

            return final_type, is_optional

        for name, prop_schema in properties.items():
            field_type, type_is_optional = map_type(prop_schema)
            is_required = name in required
            is_truly_optional = type_is_optional or not is_required

            # Use Field for descriptions, default values etc.
            field_args = {"description": prop_schema.get("description")}
            if "default" in prop_schema:
                 field_args["default"] = prop_schema["default"]
                 final_type = Optional[field_type] # Make optional if default exists
            elif is_truly_optional:
                 field_args["default"] = None
                 final_type = Optional[field_type]
            else: # Required field
                 field_args["default"] = ... # Ellipsis indicates required
                 final_type = field_type # Keep original type

            fields[name] = (final_type, Field(**field_args))

        if not fields:
             st.error("Cannot create model: No properties found in the schema.")
             return None

        DynamicModel = create_model(model_name, **fields)
        st.success(f"Successfully created dynamic Pydantic model '{model_name}' from schema.")
        return DynamicModel

    except json.JSONDecodeError:
         st.error("Invalid JSON content in schema file.")
         return None
    except Exception as e:
        st.error(f"Error creating Pydantic model from schema: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def fetch_pubmed_metadata(pmid: str) -> Optional[Dict[str, Any]]:
    """Fetches metadata (title, pub types, MeSH) from PubMed using E-utilities efetch."""
    if not pmid or not pmid.isdigit():
        return None

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml", # Fetch XML for detailed parsing
        "tool": "streamlit_app", # Identify the tool
        "email": "user@example.com" # Replace with a real email if possible
    }
    try:
        response = requests.get(base_url, params=params, timeout=20) # Increased timeout
        response.raise_for_status()
        xml_content = response.text

        # Parse the XML content
        root = ET.fromstring(xml_content)
        article = root.find(".//PubmedArticle/MedlineCitation/Article")
        mesh_list_elements = root.findall(".//PubmedArticle/MedlineCitation/MeshHeadingList/MeshHeading")
        pub_type_elements = root.findall(".//PubmedArticle/MedlineCitation/Article/PublicationTypeList/PublicationType")

        if article is None:
            st.warning(f"Could not find Article details in PubMed XML for PMID {pmid}.")
            return None

        # Extract Title
        title_element = article.find("./ArticleTitle")
        title = title_element.text if title_element is not None and title_element.text else None

        # Extract Publication Types (General)
        pub_types = [pt.text for pt in pub_type_elements if pt.text]
        pubtype_general = ", ".join(pub_types) if pub_types else None

        # Extract MeSH Terms and potential Organisms
        mesh_terms = []
        mesh_organisms = []
        # Define common organism MeSH terms (add more as needed, case-insensitive)
        common_organism_terms = {"humans", "mice", "rats", "animals"}

        for heading in mesh_list_elements:
            desc = heading.find("./DescriptorName")
            desc_text = desc.text.strip() if desc is not None and desc.text else None
            if desc_text:
                qualifiers = heading.findall("./QualifierName")
                qual_texts = [q.text.strip() for q in qualifiers if q.text]
                if qual_texts:
                    # Combine descriptor with qualifiers
                    mesh_terms.extend([f"{desc_text}/{q_text}" for q_text in qual_texts])
                else:
                    mesh_terms.append(desc_text)
                # Check if this descriptor is a potential organism
                if desc_text and desc_text.lower() in common_organism_terms:
                    if desc_text not in mesh_organisms: # Avoid duplicates
                         mesh_organisms.append(desc_text)

        metadata = {
            "title": title,
            "pubtype_general": pubtype_general,
            "mesh_list": mesh_terms if mesh_terms else None, # Store list or None
            "mesh_organisms": mesh_organisms if mesh_organisms else None # Store list or None
        }

        # Return None only if all fields are None
        if all(v is None for v in metadata.values()):
            st.warning(f"No useful metadata extracted from PubMed XML for PMID {pmid}.")
            return None

        st.info(f"Fetched PubMed metadata for {pmid}: Title='{metadata.get('title', 'N/A')}', PubTypes='{metadata.get('pubtype_general', 'N/A')}', MeSH found={bool(metadata.get('mesh_list'))}")
        return metadata

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching PubMed XML for PMID {pmid}: {e}")
        return None
    except ET.ParseError as e:
        st.error(f"Error parsing PubMed XML for PMID {pmid}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching/parsing PubMed metadata for PMID {pmid}: {e}")
        import traceback
        st.error(traceback.format_exc()) # More detailed error
        return None

def get_pmid_from_doi(doi: str) -> Optional[str]:
    """Attempts to find a PMID for a given DOI using NCBI ESearch."""
    # Clean up DOI string if needed (e.g., remove "doi:")
    doi = re.sub(r'^doi:\s*', '', doi, flags=re.IGNORECASE).strip()
    if not doi.startswith("10."): # Basic DOI format check
        st.error(f"'{doi}' does not look like a valid DOI.")
        return None

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{doi}[DOI]", # Search for the DOI in the DOI field
        "retmode": "json",
        "retmax": 1, # We only expect one PMID per DOI
        "tool": "streamlit_app",
        "email": "user@example.com" # Replace if possible
    }
    try:
        st.info(f"Searching PubMed for DOI: {doi}...")
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        esearchresult = data.get("esearchresult", {})
        idlist = esearchresult.get("idlist", [])

        if idlist and len(idlist) > 0:
            pmid = idlist[0]
            st.success(f"Found PMID {pmid} for DOI {doi}.")
            return pmid
        else:
            st.warning(f"No PMID found for DOI {doi} in PubMed.")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error searching PubMed for DOI {doi}: {e}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding PubMed search response for DOI {doi}.")
        return None
    except Exception as e:
        st.error(f"Unexpected error searching for DOI {doi}: {e}")
        return None

# Streamlit App 

def show():
    st.title("📄 Paper Processing and Analysis")

    # Load dependencies and display errors 
    deps, errors = import_dependencies()
    call_inference = deps["call_inference"]
    save_xml = deps["save_xml"]
    init_db = deps["init_db"]
    upsert_dataset = deps["upsert_dataset"]
    partition_func = deps.get("partition") # Get partition func, might be None

    if errors:
        # Display errors but allow the rest of the app to render if possible
        # Filter out specific errors handled elsewhere
        display_errors = [e for e in errors if "arxpr2_schema" not in e and "unstructured" not in e] # Also filter unstructured
        if display_errors:
            st.error(f"Import Errors: {', '.join(display_errors)}. Some functionality may be disabled.")

    # Initialize database
    if init_db:
        try:
            init_db()
        except Exception as db_init_e:
            st.error(f"Failed to initialize database: {db_init_e}")

    # Initialize session state
    default_session_state = {
        'processed_data': None, 'edited_json': None, 'source_info': None,
        'error_message': None, 'chat_context': None, 'temp_dir': None,
        'schema_choice': 'Default Schema', 'schema_model_to_use': None,
        'uploaded_schema_content': None
    }
    for key, default_value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Input Section 
    with st.container(border=True):
        st.subheader("1. Provide Input Paper")
        input_method = st.radio("Select Input Method:", ("Upload File", "Enter URL or PubMed ID"),
                                horizontal=True, key="input_method_radio", index=0)

        uploaded_file = None
        input_url_or_id = ""

        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload XML or PDF file", type=["xml", "pdf"], key="file_uploader")
            if uploaded_file:
                # Check if it's a new file upload
                is_new_file = (st.session_state.source_info is None or
                               st.session_state.source_info.get("type") != "file" or
                               st.session_state.source_info.get("name") != uploaded_file.name)

                if is_new_file:
                     st.session_state.source_info = {"type": "file", "name": uploaded_file.name, "pmid_candidate": None} # Initialize pmid_candidate
                     # Clear results on new input
                     st.session_state.processed_data = None; st.session_state.edited_json = None; st.session_state.error_message = None

                     #  Attempt to extract PMID from filename 
                     filename_match = re.match(r'^(\d{7,9})___', uploaded_file.name)
                     pmid_from_filename = filename_match.group(1) if filename_match else None
                     if pmid_from_filename:
                          st.session_state.source_info['pmid_candidate'] = pmid_from_filename
                          print(f"[Provider] Found potential PMID {pmid_from_filename} from filename.")
                          # st.info(f"Debug: PMID from filename: {pmid_from_filename}") # DEBUG
                     
        else:
            input_url_or_id_value = st.text_input("Enter XML URL or PubMed ID", key="url_input")
            if input_url_or_id_value:
                 # Check if it's a new URL/ID input
                 is_new_url_id = (st.session_state.source_info is None or
                                  st.session_state.source_info.get("type") != "url_or_id" or
                                  st.session_state.source_info.get("value") != input_url_or_id_value)

                 if is_new_url_id:
                      st.session_state.source_info = {"type": "url_or_id", "value": input_url_or_id_value, "pmid_candidate": None} # Initialize pmid_candidate
                      # Clear results on new input
                      st.session_state.processed_data = None; st.session_state.edited_json = None; st.session_state.error_message = None

                      #  Determine input type: PMID, DOI, or URL
                      input_value = input_url_or_id_value.strip()
                      if is_pubmed_id(input_value):
                           st.session_state.source_info['pmid_candidate'] = input_value
                           st.session_state.source_info['input_type'] = 'pmid'
                           print(f"[Provider] Input is PMID: {input_value}.")
                      elif input_value.lower().startswith("10.") or input_value.lower().startswith("doi:"):
                           st.session_state.source_info['input_type'] = 'doi'
                           # Attempt conversion immediately to provide feedback
                           converted_pmid = get_pmid_from_doi(input_value)
                           if converted_pmid:
                               st.session_state.source_info['pmid_candidate'] = converted_pmid
                               print(f"[Provider] Input is DOI: {input_value}, converted to PMID: {converted_pmid}.")
                           else:
                               # Keep DOI value but clear candidate, error shown by get_pmid_from_doi
                               st.session_state.source_info['pmid_candidate'] = None
                               print(f"[Provider] Input is DOI: {input_value}, but failed to convert to PMID.")
                      elif input_value.lower().startswith("http"):
                           st.session_state.source_info['input_type'] = 'url'
                           print(f"[Provider] Input is URL: {input_value}.")
                           # Attempt to extract DOI from URL and convert to PMID 
                           doi_match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', input_value, re.IGNORECASE)
                           if doi_match:
                                extracted_doi = doi_match.group(1)
                                print(f"[Provider] Extracted DOI {extracted_doi} from URL.")
                                st.info(f"Attempting to convert extracted DOI: {extracted_doi}") # ADDED LOGGING
                                converted_pmid_from_url_doi = get_pmid_from_doi(extracted_doi)
                                st.info(f"DOI to PMID conversion result: {converted_pmid_from_url_doi}") # ADDED LOGGING
                                if converted_pmid_from_url_doi:
                                     st.session_state.source_info['pmid_candidate'] = converted_pmid_from_url_doi
                                     print(f"[Provider] Stored PMID {converted_pmid_from_url_doi} from URL's DOI.")
                                else:
                                     print(f"[Provider] Failed to convert extracted DOI {extracted_doi} from URL.")
                           else:
                                print("[Provider] No DOI found in URL.")
                           # End DOI extraction from URL
                      else:
                           st.session_state.source_info['input_type'] = 'unknown'
                           st.error("Input not recognized as PMID, DOI, or URL.")
                           print(f"[Provider] Input not recognized: {input_value}.")

                 input_url_or_id = input_url_or_id_value # Keep original input for display/processing

    # Schema Selection 
    with st.container(border=True):
        st.subheader("2. Select Schema")
        st.session_state.schema_choice = st.radio(
            "Choose schema:",
            ("Default Schema", "Upload Custom Schema"),
            key="schema_choice_radio",
            index=0 if st.session_state.schema_choice == "Default Schema" else 1
        )

        uploaded_schema_file = None
        schema_load_error = False

        if st.session_state.schema_choice == "Upload Custom Schema":
            uploaded_schema_file = st.file_uploader("Upload JSON Schema file", type=["json"], key="schema_uploader")
            if uploaded_schema_file:
                # Read content only if it's a new file or content not stored
                if st.session_state.uploaded_schema_content is None or uploaded_schema_file.id != st.session_state.get('uploaded_schema_id'):
                    try:
                        st.session_state.uploaded_schema_content = uploaded_schema_file.read().decode("utf-8")
                        st.session_state.uploaded_schema_id = uploaded_schema_file.id # Store ID to detect new uploads
                        # Attempt to create model immediately
                        model = create_pydantic_model_from_schema(st.session_state.uploaded_schema_content, model_name=f"CustomSchema_{uploaded_schema_file.name}")
                        st.session_state.schema_model_to_use = model
                        if not model: schema_load_error = True
                    except Exception as e:
                        st.error(f"Error reading or parsing schema file: {e}")
                        st.session_state.uploaded_schema_content = None
                        st.session_state.schema_model_to_use = None
                        schema_load_error = True
                # If content already loaded, use the cached model
                elif st.session_state.schema_model_to_use is None and st.session_state.uploaded_schema_content:
                     # Re-create model if it failed previously but content exists
                     model = create_pydantic_model_from_schema(st.session_state.uploaded_schema_content, model_name="CustomSchema_Reloaded")
                     st.session_state.schema_model_to_use = model
                     if not model: schema_load_error = True

            elif st.session_state.uploaded_schema_content:
                 # File removed, clear stored content and model
                 st.session_state.uploaded_schema_content = None
                 st.session_state.schema_model_to_use = None
                 st.warning("Custom schema file removed.")
                 schema_load_error = True # Treat as error for disabling button

            # Display status of custom schema
            if st.session_state.schema_model_to_use and not schema_load_error:
                 st.caption(f"Using uploaded schema: {st.session_state.schema_model_to_use.__name__}")
            elif uploaded_schema_file and schema_load_error:
                 st.caption("Uploaded schema has errors or could not be processed.")
            elif not uploaded_schema_file:
                 st.warning("Select 'Upload Custom Schema' requires a file to be uploaded.")
                 schema_load_error = True # Disable processing if no file uploaded

        else: # Default Schema
            st.session_state.schema_model_to_use = get_default_schema_model()
            if st.session_state.schema_model_to_use:
                st.caption(f"Using default schema: {st.session_state.schema_model_to_use.__name__}")
            else:
                st.error("Default schema could not be loaded. Processing is disabled.")
                schema_load_error = True

    # Processing Section 
    with st.container(border=True):
        st.subheader("3. Process Paper")

        # Model Selection UI
        processing_model_choice = st.radio(
            "Select Processing Model:",
            ("Local Model", "OpenAI API"),
            key="processing_model_radio",
            horizontal=True,
            index=0 # Default to local
        )

        processing_disabled = (not uploaded_file and not input_url_or_id) or \
                              not st.session_state.schema_model_to_use or \
                              schema_load_error # Disable if any schema issue

        process_button = st.button("🚀 Process Input", use_container_width=True, disabled=processing_disabled)

        if process_button:
            st.session_state.processed_data = None; st.session_state.edited_json = None; st.session_state.error_message = None
            input_path_for_pipeline = None # Path to XML or original file
            extracted_text_content = None # Content of extracted text file
            schema_to_run = st.session_state.schema_model_to_use # Get the selected schema model

            # Manage temporary directory
            if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                 shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = tempfile.mkdtemp()
            temp_dir = st.session_state.temp_dir

            with st.spinner("Processing... This may take a while."):
                try:
                    # Handle Input (Save/Extract to Temp Dir) 
                    current_source_info = st.session_state.get('source_info', {})
                    temp_input_path = None # Path to the initial file in temp dir

                    if uploaded_file and current_source_info.get("type") == "file":
                        temp_input_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_input_path, "wb") as f: f.write(uploaded_file.getbuffer())
                        st.info(f"Using uploaded file: {uploaded_file.name}")
                        # Try extracting PMID from content (after saving) 
                        pmid_from_content = _try_extract_pmid(temp_input_path)
                        if pmid_from_content:
                              # Prioritize content over filename
                              st.session_state.source_info['pmid_candidate'] = pmid_from_content
                              print(f"[Provider] Found potential PMID {pmid_from_content} from file content.")
                              st.info(f"Debug: PMID from content (upload): {pmid_from_content}") # DEBUG
                         # 

                        if uploaded_file.name.lower().endswith(".pdf"):
                            extracted_text_path = extract_text_from_file(temp_input_path, temp_dir, partition_func) # Pass partition_func
                            if extracted_text_path and os.path.exists(extracted_text_path):
                                with open(extracted_text_path, "r", encoding="utf-8") as f:
                                    extracted_text_content = f.read()
                                input_path_for_pipeline = None # Don't pass path for extracted text
                                #  Try extracting PMID from extracted PDF text 
                                pmid_from_pdf_text = _try_extract_pmid(extracted_text_path)
                                if pmid_from_pdf_text:
                                     # Prioritize content over filename
                                     st.session_state.source_info['pmid_candidate'] = pmid_from_pdf_text
                                     print(f"[Provider] Found potential PMID {pmid_from_pdf_text} from extracted PDF text.")
                                     st.info(f"Debug: PMID from PDF text: {pmid_from_pdf_text}") # DEBUG
                                 
                            else:
                                 st.session_state.error_message = "Failed to extract text from PDF."
                        else: # Assume XML
                            input_path_for_pipeline = temp_input_path
                            extracted_text_content = None

                    elif input_url_or_id and current_source_info.get("type") == "url_or_id":
                        value = current_source_info.get("value", "") # The original user input
                        input_type = current_source_info.get("input_type", "unknown")
                        pmid_candidate = current_source_info.get("pmid_candidate") # Might be from direct input or DOI conversion

                        if input_type == 'pmid' and pmid_candidate:
                            temp_input_path = fetch_xml_from_pubmed(pmid_candidate, temp_dir, save_xml)
                        elif input_type == 'doi':
                            # Log original DOI value for debugging
                            st.info(f"Processing DOI: {value}")
                            
                            if pmid_candidate: # If DOI was successfully converted earlier
                                st.info(f"Using previously converted PMID: {pmid_candidate} for DOI: {value}")
                                temp_input_path = fetch_xml_from_pubmed(pmid_candidate, temp_dir, save_xml)
                                
                                # Check if XML was actually fetched
                                if temp_input_path and os.path.exists(temp_input_path):
                                    st.success(f"Successfully fetched XML file: {os.path.basename(temp_input_path)}")
                                    # Examine the XML content
                                    try:
                                        with open(temp_input_path, "r", encoding="utf-8", errors="ignore") as f:
                                            xml_content = f.read(1500)  # Read first 1000 chars
                                            st.info(f"XML content preview (first 200 chars): {xml_content[:200].replace(chr(10), ' ')}")
                                            if len(xml_content) < 50:
                                                st.warning(f"WARNING: XML content is very short ({len(xml_content)} chars)")
                                    except Exception as xml_read_error:
                                        st.error(f"Error reading XML content: {xml_read_error}")
                                else:
                                    st.error(f"Failed to fetch or save XML for PMID: {pmid_candidate}")
                            else:
                                # Attempt conversion again if it failed initially or state was lost
                                st.info(f"Re-attempting DOI to PMID conversion for DOI: {value}...")
                                pmid_candidate = get_pmid_from_doi(value)
                                if pmid_candidate:
                                    st.session_state.source_info['pmid_candidate'] = pmid_candidate # Update state
                                    st.success(f"Successfully converted DOI to PMID: {pmid_candidate}")
                                    
                                    # Inspect XML content before processing
                                    st.info(f"Fetching XML from PubMed using PMID: {pmid_candidate}")
                                    temp_input_path = fetch_xml_from_pubmed(pmid_candidate, temp_dir, save_xml)
                                    
                                    # Check if XML was actually fetched
                                    if temp_input_path and os.path.exists(temp_input_path):
                                        st.success(f"Successfully fetched XML file: {os.path.basename(temp_input_path)}")
                                        try:
                                            file_size = os.path.getsize(temp_input_path)
                                            st.info(f"XML file size: {file_size} bytes")
                                            
                                            if file_size < 1000:
                                                st.warning(f"XML file is very small ({file_size} bytes), checking if it contains an error message")
                                                # Verify file contents
                                                with open(temp_input_path, "r", encoding="utf-8", errors="ignore") as f:
                                                    xml_content = f.read()
                                                    if '[Error]' in xml_content or 'No result can be found' in xml_content:
                                                        st.error(f"XML file contains an error message: {xml_content[:200]}")
                                                        st.info("This paper might not be available in PubMed Central's open access collection.")
                                                        st.info("Attempting to use direct text entry instead...")
                                                        
                                                        # Create a fallback path by using the DOI to query a citation API
                                                        try:
                                                            doi_citation_url = f"https://api.crossref.org/works/{value}/transform/text/x-bibliography"
                                                            citation_response = requests.get(doi_citation_url, timeout=10)
                                                            if citation_response.status_code == 200:
                                                                citation_text = citation_response.text
                                                                st.info(f"Found citation: {citation_text}")
                                                                
                                                                # Create a text file with the citation and DOI
                                                                citation_file_path = os.path.join(temp_dir, f"{pmid_candidate}_citation.txt")
                                                                with open(citation_file_path, "w", encoding="utf-8") as cf:
                                                                    cf.write(f"Citation: {citation_text}\n\nDOI: {value}\nPMID: {pmid_candidate}\n\n")
                                                                    cf.write("NOTE: This paper could not be automatically retrieved. This is a placeholder with the citation information only.")
                                                                
                                                                # Use this text file as the input instead
                                                                extracted_text_content = f"Citation: {citation_text}\n\nDOI: {value}\nPMID: {pmid_candidate}\n\nNOTE: This paper could not be automatically retrieved. This is a placeholder with the citation information only."
                                                                input_path_for_pipeline = None
                                                                st.success("Created citation placeholder for processing")
                                                        except Exception as citation_error:
                                                            st.error(f"Error fetching citation: {citation_error}")
                                        except Exception as file_check_error:
                                            st.error(f"Error checking XML file: {file_check_error}")
                                    else:
                                        st.error(f"Failed to fetch or save XML for PMID: {pmid_candidate}")
                                        st.info("This paper might not be available in PubMed Central's open access collection.")
                                else:
                                    st.error(f"Could not find a PMID for DOI: {value}")
                                    st.session_state.error_message = f"Could not find a PMID for DOI: {value}"
                        elif input_type == 'url':
                            temp_input_path = fetch_xml_from_url(value, temp_dir)
                        else: # Handle case where input type was unknown or DOI conversion failed and wasn't handled
                             if not st.session_state.error_message: # Avoid overwriting specific DOI errors
                                  st.error("Invalid input type detected or required identifier missing.")
                                  st.session_state.error_message = "Invalid input or missing identifier."
                        if temp_input_path and os.path.exists(temp_input_path):
                            #  Try extracting PMID from fetched URL content
                            if not pmid_candidate: # Only extract if we don't already have one from DOI conversion etc.
                                 pmid_from_content = _try_extract_pmid(temp_input_path)
                                 if pmid_from_content:
                                      st.session_state.source_info['pmid_candidate'] = pmid_from_content
                                      print(f"[Provider] Found potential PMID {pmid_from_content} from URL content.")
                                      st.info(f"Debug: PMID from URL content: {pmid_from_content}") # DEBUG
                             # 

                            # Always attempt text extraction for URL downloads
                            st.info("Attempting to extract text content from downloaded URL file...")
                            extracted_text_path = extract_text_from_file(temp_input_path, temp_dir, partition_func)
                            if extracted_text_path and os.path.exists(extracted_text_path):
                                 with open(extracted_text_path, "r", encoding="utf-8") as f:
                                      extracted_text_content = f.read()
                                 input_path_for_pipeline = None # Use text content, not path
                                 st.info("Using extracted text content for pipeline.")
                                 # Optionally re-run PMID extraction on the cleaner text ONLY if not already found via URLs DOI
                                 if not st.session_state.source_info.get('pmid_candidate'):
                                      st.info("Attempting PMID extraction from extracted URL text...")
                                      pmid_from_extracted_text = _try_extract_pmid(extracted_text_path)
                                      if pmid_from_extracted_text:
                                           st.session_state.source_info['pmid_candidate'] = pmid_from_extracted_text
                                           st.info(f"Debug: PMID from extracted text: {pmid_from_extracted_html}")
                            else:
                                 st.warning("Failed to extract text from downloaded URL file, proceeding with original file path (might cause downstream errors).")
                                 input_path_for_pipeline = temp_input_path # Fallback to original path
                                 extracted_text_content = None
                            # ---------------------------------------------------------
                        elif not st.session_state.error_message: # If fetch didn't already set an error
                            st.session_state.error_message = "Failed to fetch or save input from URL."

                    else:
                         st.error("No valid input source detected.")
                         st.session_state.error_message = "No input provided."

                    # Run Pipeline 
                     # Check if we have either a valid path OR extracted text content AND no prior error
                    st.info("Checking conditions before running pipeline...") #  LOGGING
                    if not st.session_state.error_message and ((input_path_for_pipeline and os.path.exists(input_path_for_pipeline)) or extracted_text_content):
                         st.info(f"Schema to run: {schema_to_run.__name__ if schema_to_run else 'None'}") # LOGGING
                         if schema_to_run:
                             st.info(f"Debug: PMID candidate before pipeline: {st.session_state.get('source_info', {}).get('pmid_candidate')}") # DEBUG
                             if call_inference:
                                 # Determine model based on UI choice
                                if processing_model_choice == "OpenAI API":
                                    selected_ff_model = "4o" # Or "4om"
                                    st.info("Using OpenAI API for processing.")
                                else:
                                    selected_ff_model = "llama3.1I-8b-q4" # Default local
                                    st.info("Using Local Model for processing.")

                                pipeline_settings = { # TODO: Make similarity_k/field_info configurable via UI
                                    'similarity_k': 5,
                                    'field_info_to_compare': 'choices',
                                    'ff_model': selected_ff_model
                                }

                                # Prepare arguments for call_inference
                                inference_args = {
                                    "schema": schema_to_run,
                                    **pipeline_settings
                                }
                                if extracted_text_content:
                                    inference_args["parsed_paper_text"] = extracted_text_content
                                    st.info(f"Running pipeline on extracted text content with schema '{schema_to_run.__name__}'...")
                                elif input_path_for_pipeline:
                                    inference_args["paper_path"] = input_path_for_pipeline
                                    st.info(f"Running pipeline on file '{os.path.basename(input_path_for_pipeline)}' with schema '{schema_to_run.__name__}'...")
                                else: # Should not happen if the outer condition is met
                                     st.error("Internal error: No valid input path or text content for pipeline.")
                                     st.session_state.error_message = "Internal input error."

                                # Only run if no internal error
                                if "error_message" not in st.session_state or st.session_state.error_message is None:
                                    st.info("Calling inference pipeline...") # LOGGING
                                    output = call_inference(**inference_args)
                                    st.info("Inference pipeline finished.") # LOGGING

                                    # [Output handling remains similar]
                                    if output:
                                        first_key = next(iter(output), None)
                                        if first_key and isinstance(output[first_key], dict) and "filled_form" in output[first_key]:
                                            st.session_state.processed_data = {
                                                "form": output[first_key].get("filled_form", {}),
                                                "context": output[first_key].get("context", {})
                                            }
                                            serializable_form = st.session_state.processed_data["form"]
                                            if hasattr(serializable_form, 'dict'): serializable_form = serializable_form.dict()
                                            st.session_state.edited_json = json.dumps(serializable_form, indent=4, default=str)
                                            st.success("Pipeline completed successfully!")
                                        else:
                                            st.warning("Pipeline output structure unexpected. Using raw output.")
                                            st.session_state.processed_data = {"form": output, "context": {}}
                                            st.session_state.edited_json = json.dumps(output, indent=4, default=str)
                                            st.success("Pipeline completed (structure might differ).")
                                    else:
                                        st.error("Pipeline execution failed or returned no output.")
                                        st.session_state.error_message = "Pipeline execution failed."
                             else:
                                st.error("Processing function (call_inference) is not available.")
                                st.session_state.error_message = "Processing function unavailable."
                    elif not st.session_state.error_message: # If no specific error yet
                            if not schema_to_run: st.session_state.error_message = "Schema unavailable."
                             # The condition below is now covered by the outer if statement
                             # elif not input_path_for_pipeline or not os.path.exists(input_path_for_pipeline): st.session_state.error_message = "Input file path invalid or missing after processing."

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.session_state.error_message = str(e)

    # Output Display and Editing 
    with st.container(border=True):
        st.subheader("4. Review and Edit Results")
        # Display logic remains similar
        if st.session_state.processed_data and st.session_state.edited_json is not None:
            st.info("Review the extracted data below. You can edit the JSON directly before saving or chatting.")
            edited_json_state = st.text_area(
                "Extracted Data (Editable JSON):", value=st.session_state.edited_json, height=400, key="json_edit_area"
            )
            if edited_json_state != st.session_state.edited_json:
                st.session_state.edited_json = edited_json_state
                st.caption("✏️ Changes detected in JSON. Remember to save if needed.")
        elif st.session_state.error_message:
            st.error(f"Processing failed: {st.session_state.error_message}")
        else:
            st.info("Process a paper using the button above to see results here.")

    # Actions Section
    with st.container(border=True):
        st.subheader("5. Actions")
        # Actions logic remains similar
        if st.session_state.processed_data and st.session_state.edited_json is not None:
            # Add input for custom dataset name
            custom_dataset_name = st.text_input(
                "Enter a name for this dataset (optional):",
                placeholder=f"e.g., {st.session_state.source_info.get('name', 'My Processed Paper')}",
                key="custom_dataset_name_input"
            )

            col1, col2 = st.columns(2)
            with col1:
                save_db_button = st.button("💾 Save to Database", key="save_db", use_container_width=True)
                if save_db_button:
                    if upsert_dataset:
                        try:
                            # Get the full structure potentially edited from the text area
                            edited_data_full = json.loads(st.session_state.edited_json)

                            #  Access the NESTED filled_form dictionary 
                            # Assuming the outer key is usually 0, or fallback to the first key's value
                            first_key = next(iter(edited_data_full), None)
                            if first_key:
                                inner_data = edited_data_full.get(first_key, {})
                                form_fields = inner_data.get("filled_form", {}) # Get the 'filled_form' dict
                            else:
                                form_fields = {} # Handle case where JSON is empty or malformed

                            if not form_fields:
                                st.error("Could not find 'filled_form' data within the processed JSON. Cannot save.")
                                raise ValueError("Missing 'filled_form' in processed data")
                            

                            source_info = st.session_state.get('source_info', {})

                            # Extract Key Identifiers 
                            pmid_candidate = source_info.get('pmid_candidate')
                            # Check if pmid is directly in form_fields (less likely but possible)
                            pmid_from_form = form_fields.get('pmid') or form_fields.get('PMID')
                            pmid = pmid_candidate or pmid_from_form
                            if not pmid or str(pmid).lower() == 'unknown': pmid = 'NO_PMID'

                            #  Fetch Metadata from PubMed (useful for Title fallback)
                            pubmed_metadata = None
                            if pmid != 'NO_PMID':
                                pubmed_metadata = fetch_pubmed_metadata(pmid)

                            # Accession: Generate a unique one if not in form_fields
                            accession_from_form = form_fields.get('accession') or form_fields.get('Accession')
                            if not accession_from_form or str(accession_from_form).lower() == 'unknown':
                                base_name = custom_dataset_name.strip() or source_info.get('name', 'user_dataset')
                                safe_base = re.sub(r'[^\w\-]+', '_', base_name).strip('_') if base_name else 'dataset'
                                accession = f"USER_{safe_base}_{hash(st.session_state.edited_json) % 10000}"
                            else:
                                accession = accession_from_form

                            # Title: Prioritize custom input, then PubMed metadata, then fallback
                            title = custom_dataset_name.strip() or \
                                    (pubmed_metadata.get('title') if pubmed_metadata else None) or \
                                    source_info.get('name', f"Dataset {accession}")

                            #  Prepare Data for Database - Prioritize Profiler Output 
                            db_data = {
                                'title': title,
                                'accession': accession,
                                'pmid': pmid,
                                'description': form_fields.get('description', 'User-provided dataset via Provider page.'),
                                'source': 'user_provider',
                                'original_file': source_info.get('value', source_info.get('name', 'Unknown')),
                                # Initialize ALL expected DB columns to None
                                'organism': None,
                                'study_type': None,
                                'hardware': None,
                                'organism_part': None, # <<< ENSURE THIS LINE EXISTS
                                'experimental_designs': None,
                                'assay_by_molecule': None,
                                'technology': None,
                                
                            }

                            #  Map profiler output (form_fields) to db_data keys 
                            # (The loop below this should now work correctly for organism_part)
                            for key, value in form_fields.items():
                                # Remove the trailing '_number' if it exists to get the base field name
                                base_key = re.sub(r'_\d+$', '', key)

                                # If this base_key matches a column name in db_data, update it
                                if base_key in db_data:
                                    # Prioritize the value from the profiler (form_fields)
                                    if value is not None and value != "":
                                        db_data[base_key] = value
                                else:
                                    # Optional: Log keys from profiler output that don't match DB columns
                                    print(f"[Provider Save] Warning: Profiler key '{key}' (base: '{base_key}') not found in db_data map.")

                            # --- Log final values before saving, mostly for debugging ---
                            # st.info(f"Debug (Save): Final Organism for DB: {db_data.get('organism')}")
                            # Add log for organism_part
                            # st.info(f"Debug (Save): Final Organism Part for DB: {db_data.get('organism_part')}")
                            # st.info(f"Debug (Save): Final Study Type for DB: {db_data.get('study_type')}")
                            # st.info(f"Debug (Save): Final Hardware for DB: {db_data.get('hardware')}")

                            # --- Save JSON and Update DB ---
                            user_datasets_dir = os.path.join(project_root, 'llm_ui', 'app', 'user_datasets')
                            os.makedirs(user_datasets_dir, exist_ok=True)
                            safe_filename = re.sub(r'[^\w\-_\.]', '_', f"user_dataset_{accession}.json")
                            save_path = os.path.join(user_datasets_dir, safe_filename)

                            # Save the original *full* profiler output structure
                        
                            with open(save_path, "w") as f:
                                json.dump(edited_data_full, f, indent=4) # Save the full structure
                            st.info(f"Saved full profiler output JSON to: {save_path}")

                            # Add the file_path (primary key for DB) and upsert using the mapped db_data
                            db_data['file_path'] = save_path
                            upsert_dataset(db_data)
                            st.success(f"Dataset '{db_data['title']}' saved to database!")

                        except json.JSONDecodeError: st.error("Cannot save to database: Edited JSON is invalid.")
                        except ValueError as ve: st.error(f"Cannot save to database: {ve}") # Catch specific error
                        except Exception as e: st.error(f"Error saving to database: {e}"); import traceback; st.error(traceback.format_exc())
                    else: st.error("Database function (upsert_dataset) is not available.")

            with col2:
                chat_button = st.button("💬 Chat with LLM", key="chat_llm", use_container_width=True)
                if chat_button:
                     #Chat preparation logic remains similar
                    try:
                        st.session_state.chat_context = {
                            "processed_data": json.loads(st.session_state.edited_json),
                            "context_info": st.session_state.processed_data.get("context", {}),
                            "source": st.session_state.get('source_info', {})
                        }
                        st.success("Data prepared for chat.")
                        st.info("Navigate to the 'Consumer QA' page from the sidebar to start chatting.")
                    except json.JSONDecodeError: st.error("Cannot prepare for chat: Edited JSON is invalid.")
                    except Exception as e: st.error(f"Error preparing for chat: {e}")
        else:
            st.info("Process a paper to enable actions.")

    #  Footer 
    st.markdown("---")
    st.caption("Provider Page - Alpha Version")

    # Optional: Add a button to clear the temporary directory if needed 
    if 'temp_dir' in st.session_state and st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
         if st.button("Clean Temp Files"):
             try:
                 shutil.rmtree(st.session_state.temp_dir)
                 st.session_state.temp_dir = None # Reset state variable
                 st.success("Temporary files cleaned.")
                 st.rerun() # Rerun to reflect the change
             except Exception as e:
                 st.error(f"Failed to clean temporary files: {e}")
