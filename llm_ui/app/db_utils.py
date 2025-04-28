import sqlite3
import os
import json
import glob
import re
from datetime import datetime
import sys
import requests
from xml.etree import ElementTree as ET
import time
from typing import List, Dict, Any, Optional

# Define project root relative to this file's location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DB_PATH = os.path.join(project_root, 'llm_ui', 'app', 'datasets.db')
USER_DATASETS_DIR = os.path.join(project_root, 'llm_ui', 'app', 'user_datasets')
DATA_DIR = os.path.join(project_root, 'data')

# --- DB Connection and Initialization ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, timeout=10) # Increased timeout
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the datasets table if it doesn't exist."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                file_path TEXT PRIMARY KEY,
                accession TEXT,
                pmid TEXT,
                title TEXT,
                organism TEXT,
                study_type TEXT,
                description TEXT,
                source TEXT,
                last_updated TIMESTAMP,
                hardware TEXT,
                organism_part TEXT,
                experimental_designs TEXT,
                assay_by_molecule TEXT,
                technology TEXT,
                sample_count TEXT,
                release_date TEXT,
                experimental_factors TEXT
            )
        ''')
        # Add indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_accession ON datasets (accession)",
            "CREATE INDEX IF NOT EXISTS idx_pmid ON datasets (pmid)",
            "CREATE INDEX IF NOT EXISTS idx_organism ON datasets (organism)",
            "CREATE INDEX IF NOT EXISTS idx_study_type ON datasets (study_type)",
            "CREATE INDEX IF NOT EXISTS idx_source ON datasets (source)",
            "CREATE INDEX IF NOT EXISTS idx_title ON datasets (title)" # Index title for searching
        ]
        for index_sql in indexes:
            cursor.execute(index_sql)
        conn.commit()
        print("[DB_UTILS] Database table 'datasets' ensured.")
    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error during initialization: {e}")
        raise
    except Exception as e:
        print(f"[ERROR][DB_UTILS] An unexpected error occurred during DB initialization: {e}")
        raise
    finally:
         if conn:
             conn.close()

# --- Data Upsert Functions ---
def _prepare_data_tuple(metadata, columns):
    """Prepares a tuple from metadata dictionary for DB insertion."""
    now = datetime.now()
    data_list = []
    for col in columns:
        if col == 'last_updated':
            data_list.append(now)
        else:
            value = metadata.get(col)
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            elif not isinstance(value, (str, int, float, bytes, type(None))):
                 value = str(value)
            data_list.append(value)
    return tuple(data_list)

def upsert_dataset(metadata):
    """Adds or updates a single dataset record using INSERT OR REPLACE."""
    conn = None
    if not metadata or 'file_path' not in metadata:
         print("[ERROR][DB_UTILS] upsert_dataset called with invalid metadata (missing file_path).")
         return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        columns = [
            'file_path', 'accession', 'pmid', 'title', 'organism', 'study_type',
            'description', 'source', 'last_updated', 'hardware', 'organism_part',
            'experimental_designs', 'assay_by_molecule', 'technology',
            'sample_count', 'release_date', 'experimental_factors'
        ]
        data_tuple = _prepare_data_tuple(metadata, columns)
        sql = f'''
            INSERT OR REPLACE INTO datasets ({', '.join(columns)})
            VALUES ({', '.join(['?'] * len(columns))})
        '''
        cursor.execute(sql, data_tuple)
        conn.commit()
        print(f"[DB_UTILS] Successfully upserted single record: {metadata.get('file_path')}") # Add success print
    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error during single upsert for {metadata.get('file_path')}: {e}")
        print(f"  SQLite Error Code: {e.sqlite_errorcode}")
        print(f"  SQLite Error Name: {e.sqlite_errorname}")
        print(f"  SQL: {sql}")
        print(f"  Data tuple ({len(data_tuple)} items): {data_tuple}") # Print data that caused error
    except Exception as e:
         print(f"[ERROR][DB_UTILS] Unexpected error during single upsert for {metadata.get('file_path')}: {e}")
         print(f"  Metadata: {metadata}") # Print metadata that caused error
         print(f"  Data tuple: {data_tuple}")
    finally:
        if conn:
            conn.close()

def batch_upsert_datasets(metadata_list):
    """Adds or updates a batch of dataset records using executemany."""
    if not metadata_list:
        return 0
    conn = None
    inserted_count = 0
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        columns = [
            'file_path', 'accession', 'pmid', 'title', 'organism', 'study_type',
            'description', 'source', 'last_updated', 'hardware', 'organism_part',
            'experimental_designs', 'assay_by_molecule', 'technology',
            'sample_count', 'release_date', 'experimental_factors'
        ]
        data_tuples = [_prepare_data_tuple(metadata, columns) for metadata in metadata_list]
        sql = f'''
            INSERT OR REPLACE INTO datasets ({', '.join(columns)})
            VALUES ({', '.join(['?'] * len(columns))})
        '''
        # Use a transaction for batch insert
        conn.execute("BEGIN TRANSACTION")
        cursor.executemany(sql, data_tuples)
        conn.commit()
        inserted_count = len(metadata_list)
        # print(f"[DB_UTILS] Batch upserted {inserted_count} records.")
    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error during batch upsert: {e}")
        if conn: conn.rollback() # Rollback transaction on error
        # Optionally try individual upserts as fallback here
        print(f"  Attempting individual upserts for failed batch...")
        failed_count = 0
        for metadata in metadata_list:
             try:
                 upsert_dataset(metadata) # Fallback to individual insert
             except Exception as individual_e:
                 failed_count += 1
                 print(f"[ERROR][DB_UTILS] Individual upsert failed for {metadata.get('file_path')} after batch failure: {individual_e}")
        print(f"  Individual upsert fallback completed with {failed_count} failures.")

    except Exception as e:
         print(f"[ERROR][DB_UTILS] Unexpected error during batch upsert: {e}")
         if conn: conn.rollback()
    finally:
        if conn:
            conn.close()
    return inserted_count

# --- Metadata Extraction Helpers ---
def _fetch_pubmed_title(pmid):
    """Fetches the title for a given PMID from PubMed."""
    if not pmid or pmid == 'unknown' or not str(pmid).isdigit():
        return None
    try:
        pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
        response = requests.get(pubmed_url, timeout=10)
        time.sleep(0.4) # Be nice to NCBI
        response.raise_for_status()
        root = ET.fromstring(response.text)
        article_title = root.find(".//ArticleTitle")
        if article_title is not None and article_title.text:
            return article_title.text
    except requests.exceptions.RequestException as e:
        print(f"[WARN][DB_UTILS] Network error fetching title for PMID {pmid}: {e}")
    except ET.ParseError as e:
        print(f"[WARN][DB_UTILS] XML parsing error for PMID {pmid}: {e}")
    except Exception as e:
        print(f"[WARN][DB_UTILS] Unexpected error fetching title for PMID {pmid}: {e}")
    return None

def _extract_metadata_for_db(file_path):
    """Helper to extract metadata from a JSON file for DB insertion. Returns metadata dict or None."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    except Exception as e:
        print(f"[ERROR][DB_UTILS] Reading/parsing JSON file {file_path}: {e}")
        return None

    filename = os.path.basename(file_path)
    accession_match = re.search(r'(E-\w+-\d+)', filename)
    pmid_match = re.search(r'^(\d+)___', filename)
    user_accession_match = re.search(r'user_dataset_(.+)\.json', filename)

    pmid = pmid_match.group(1) if pmid_match else data.get('pmid', 'unknown')
    accession = accession_match.group(1) if accession_match else (user_accession_match.group(1) if user_accession_match else data.get('accession', 'Unknown'))

    metadata = {'file_path': file_path, 'accession': accession, 'pmid': pmid}
    metadata.update({k: None for k in [
        'title', 'organism', 'study_type', 'description', 'source', 'hardware',
        'organism_part', 'experimental_designs', 'assay_by_molecule', 'technology',
        'sample_count', 'release_date', 'experimental_factors']})

    is_user_saved_file = 'user_datasets' in file_path.replace(os.sep, '/')
    is_user_provider_source = isinstance(data, dict) and data.get('source') == 'user_provider'

    if is_user_saved_file or is_user_provider_source:
        metadata['source'] = 'user_provider'
        metadata.update({k: data.get(k) for k in metadata if k in data})
        if not metadata.get('title'): metadata['title'] = f"User Dataset {accession}"
    elif any(pattern in filename for pattern in ["arxpr_simplified.json", "arxpr_metadataset", "arxpr2_25"]):
         return None # Skip bulk files in this function
    else:
        metadata['source'] = 'arrayexpress'
        if isinstance(data, dict):
            data_lower = {k.lower(): v for k, v in data.items() if isinstance(k, str)}
            metadata['title'] = data_lower.get('title', metadata['title'])
            metadata['description'] = data_lower.get('description', metadata['description'])
            metadata['organism'] = data_lower.get('organism', metadata['organism'])
            metadata['study_type'] = data_lower.get('study type', data_lower.get('study_type', metadata['study_type']))

            if 'section' in data:
                 sections = data['section'] if isinstance(data['section'], list) else [data['section']]
                 for section in sections:
                     if not isinstance(section, dict): continue
                     section_type = section.get('type', '').lower()
                     attributes = section.get('attributes', [])
                     if isinstance(attributes, dict): attributes = [attributes]

                     if section_type == 'study title' and not metadata['title']:
                         for attr in attributes:
                             if isinstance(attr, dict) and attr.get('name') == 'Title': metadata['title'] = attr.get('value'); break
                     elif section_type == 'study description' and not metadata['description']:
                         for attr in attributes:
                             if isinstance(attr, dict) and attr.get('name') == 'Description': metadata['description'] = attr.get('value'); break
                     elif section_type == 'study':
                          for attr in attributes:
                              if isinstance(attr, dict) and 'name' in attr and 'value' in attr:
                                  attr_name = attr['name'].lower()
                                  if attr_name == 'organism' and not metadata['organism']: metadata['organism'] = attr['value']
                                  elif attr_name == 'study type' and not metadata['study_type']: metadata['study_type'] = attr['value']
                                  elif attr_name == 'release date' and not metadata['release_date']: metadata['release_date'] = attr['value']

                     subsections = section.get('subsections', [])
                     if isinstance(subsections, dict): subsections = [subsections]
                     if isinstance(subsections, list):
                          for sub in subsections:
                              if not isinstance(sub, dict): continue
                              sub_type = sub.get('type', '').lower()
                              sub_attributes = sub.get('attributes', [])
                              if isinstance(sub_attributes, dict): sub_attributes = [sub_attributes]
                              for attr in sub_attributes:
                                   if isinstance(attr, dict) and 'name' in attr and 'value' in attr:
                                       attr_name = attr['name'].lower()
                                       if attr_name == 'sample count' and not metadata['sample_count']: metadata['sample_count'] = attr['value']
                                       elif attr_name == 'experimental factors' and not metadata['experimental_factors']: metadata['experimental_factors'] = attr['value']
                                       elif attr_name == 'hardware' and not metadata['hardware']: metadata['hardware'] = attr['value']
                                       elif attr_name == 'technology' and not metadata['technology']: metadata['technology'] = attr['value']
                                       elif attr_name == 'organism part' and not metadata['organism_part']: metadata['organism_part'] = attr['value']
                                       elif attr_name == 'organism' and not metadata['organism']: metadata['organism'] = attr['value']
                                       elif attr_name == 'assay by molecule' and not metadata['assay_by_molecule']: metadata['assay_by_molecule'] = attr['value']

    # Set default title (will be updated later if possible)
    if not metadata.get('title'): metadata['title'] = f"Dataset {accession}"
    if not metadata.get('organism'): metadata['organism'] = "Unknown"
    if not metadata.get('study_type'): metadata['study_type'] = "Unknown"

    # Convert all values to string for consistency, default missing crucial text fields to None
    for key in metadata:
        if metadata[key] is not None:
            metadata[key] = str(metadata[key])
        elif key in ['organism', 'study_type', 'title', 'accession', 'pmid', 'source']:
             metadata[key] = "Unknown" # Keep Unknown for these required fields
        # else: description and other non-essential fields remain None if not found

    return metadata

def _extract_metadata_from_bulk_entry(entry: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    """Extracts and maps metadata from a single entry within a bulk JSON file."""
    metadata = { # Initialize with DB column names
        'title': None,
        'description': None,
        'accession': None,
        'pmid': None,
        'organism': None,
        'study_type': None,
        'hardware': None,
        'organism_part': None,
        'experimental_designs': None,
        'assay_by_molecule': None,
        'technology': None,
        'source': 'bulk_processed', # Mark source
         # Keep file_path for reference, but accession will be primary key if available
        'file_path': None # Will be set based on accession later if possible
    }

    try:
        # --- Direct Mapping (Adjust keys based on actual bulk JSON structure) ---
        metadata['accession'] = entry.get('accession')
        metadata['title'] = entry.get('title')
        metadata['description'] = entry.get('description')
        metadata['organism'] = entry.get('organism') # Assumes 'organism' key exists
        metadata['pmid'] = entry.get('pmid')

        # --- Mapping based on common ArrayExpress / SDRF concepts ---
        # Map experimenttype to study_type
        metadata['study_type'] = entry.get('experimenttype') or entry.get('study_type') # Prioritize specific key if available

        # Map performer/protocol info to hardware/technology
        # This might need refinement based on how hardware is listed (e.g., in 'performer' or 'protocol' fields)
        metadata['hardware'] = entry.get('performer') or entry.get('hardware') # Example: use 'performer' if 'hardware' key missing
        metadata['technology'] = entry.get('technology') # If a specific technology field exists

        # --- Attempt to extract other fields (Requires knowing the bulk JSON structure) ---
        # These are examples - **adjust the entry.get('key_name') based on your actual bulk JSON keys**
        metadata['organism_part'] = entry.get('organism_part') or entry.get('sample_characteristic_organism_part') # Example keys
        metadata['experimental_designs'] = entry.get('experimental_design') or entry.get('experiment_design') # Example keys
        metadata['assay_by_molecule'] = entry.get('assay_name') or entry.get('assay_type') # Example keys


        # --- Set file_path based on accession for uniqueness if needed ---
        # The primary key in the DB is file_path, so we need a unique value.
        # Using the accession (if available) combined with the source file ensures uniqueness.
        acc = metadata['accession']
        if acc:
            metadata['file_path'] = f"{file_path}::{acc}" # Combine source file and accession
        else:
            # Fallback if no accession - this might cause issues if multiple entries lack accession
            metadata['file_path'] = f"{file_path}::entry_{hash(json.dumps(entry, sort_keys=True))}"


        # Simple cleanup
        cleaned_metadata = {k: v.strip() if isinstance(v, str) else v for k, v in metadata.items() if v is not None and v != ""} # Remove None and empty strings
        # Ensure essential keys for DB are present, even if None from above
        cleaned_metadata.setdefault('file_path', metadata['file_path'])
        cleaned_metadata.setdefault('accession', metadata['accession'])


        #print(f"[DB Utils Bulk Extract] Extracted for {metadata['accession']}: { {k:v for k,v in cleaned_metadata.items() if k!='file_path'} }") # Log extracted data
        return cleaned_metadata

    except Exception as e:
        acc = entry.get('accession', 'UNKNOWN_ACC')
        print(f"[DB Utils Bulk Extract] Error extracting metadata for entry {acc} in {file_path}: {e}")
        # Return minimal info on error
        return {'file_path': f"{file_path}::error_{acc}", 'accession': acc, 'title': f"Error Extracting {acc}"}


# --- PubMed Title Fetching ---
def _fetch_pubmed_titles_batch(pmids):
    """Fetches titles for a batch of PMIDs."""
    titles = {}
    valid_pmids = [p for p in pmids if p and str(p).isdigit()]
    if not valid_pmids: return titles
    pmid_list = ",".join(valid_pmids)

    print(f"[DB_UTILS] Fetching titles for {len(valid_pmids)} PMIDs...")
    try:
        pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid_list}&retmode=xml"
        response = requests.get(pubmed_url, timeout=20)
        time.sleep(0.4) # Be nice to NCBI
        response.raise_for_status()
        root = ET.fromstring(response.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is not None and pmid_elem.text:
                pmid = pmid_elem.text
                article_title = article.find(".//ArticleTitle")
                if article_title is not None and article_title.text:
                    titles[pmid] = article_title.text
    except requests.exceptions.RequestException as e:
        print(f"[WARN][DB_UTILS] Network error fetching PubMed batch: {e}")
    except ET.ParseError as e:
        print(f"[WARN][DB_UTILS] XML parsing error for PubMed batch: {e}")
    except Exception as e:
        print(f"[WARN][DB_UTILS] Unexpected error fetching PubMed batch: {e}")
    print(f"[DB_UTILS] Fetched {len(titles)} titles for batch.")
    return titles

def update_titles_from_pubmed(batch_size=100):
    """Queries DB for entries needing titles and updates them in batches."""
    conn = None
    updated_count = 0
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql_select = """
            SELECT file_path, pmid FROM datasets
            WHERE pmid IS NOT NULL AND pmid != 'unknown' AND pmid != ''
            AND (title IS NULL OR title = 'Unknown' OR title LIKE 'Dataset %' OR title LIKE 'PMID:%' OR title LIKE 'User Dataset %')
        """
        cursor.execute(sql_select)
        records_to_update = cursor.fetchall()
        total_to_fetch = len(records_to_update)
        print(f"[DB_UTILS] Found {total_to_fetch} records potentially needing title update from PubMed.")
        if total_to_fetch == 0:
            return

        pmid_map = {row['pmid']: row['file_path'] for row in records_to_update if str(row['pmid']).isdigit()}
        pmids_to_fetch = list(pmid_map.keys())

        for i in range(0, len(pmids_to_fetch), batch_size):
            batch_pmids = pmids_to_fetch[i:i+batch_size]
            fetched_titles = _fetch_pubmed_titles_batch(batch_pmids)

            updates = []
            for pmid, title in fetched_titles.items():
                if pmid in pmid_map:
                    updates.append((title, pmid_map[pmid]))

            if updates:
                sql_update = "UPDATE datasets SET title = ?, last_updated = ? WHERE file_path = ?"
                now = datetime.now()
                updates_with_ts = [(title, now, file_path) for title, file_path in updates]
                cursor.executemany(sql_update, updates_with_ts)
                conn.commit()
                updated_count += len(updates)
                print(f"[DB_UTILS] Updated {len(updates)} titles in batch {i//batch_size + 1}. Total updated: {updated_count}")

    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error during title update: {e}")
    except Exception as e:
        print(f"[ERROR][DB_UTILS] Unexpected error during title update: {e}")
    finally:
        if conn:
            conn.close()
    print(f"[DB_UTILS] Finished title update process. Total titles updated: {updated_count}")


# --- Main Scan Function ---
def scan_and_update_db(directories):
    """Scans directories for JSON files, extracts metadata, and updates the database in batches."""
    print(f"[DB_UTILS] Starting scan_and_update_db for directories: {directories}")
    all_files_found = []
    processed_count = 0
    bulk_processed_count = 0
    batch_size = 500
    metadata_batch = []

    for directory in directories:
        abs_dir = os.path.abspath(directory)
        if os.path.isdir(abs_dir):
            print(f"[DB_UTILS] Scanning directory: {abs_dir}")
            try:
                json_files = glob.glob(os.path.join(abs_dir, "**", "*.json"), recursive=True)
                print(f"[DB_UTILS] Found {len(json_files)} potential JSON files in {abs_dir}.")
                all_files_found.extend(json_files)
            except Exception as e:
                print(f"[ERROR][DB_UTILS] Error scanning directory {abs_dir}: {e}")
        else:
            print(f"[WARN][DB_UTILS] Directory not found or is not a directory: {abs_dir}")

    config_files = ['cached_datasets.json', 'settings.json', 'previous_datasets.json', 'default_schema.json', 'output.json']
    all_files_found = [
        f for f in all_files_found if os.path.basename(f) not in config_files and
        not os.path.basename(f).startswith('full_output_') and
        not os.path.basename(f).startswith('updated_output_')
    ]
    print(f"[DB_UTILS] Found {len(all_files_found)} files after filtering config/output files.")

    bulk_file_patterns = [
        "arxpr_simplified.json", "arxpr_metadataset_train.json",
        "arxpr_metadataset_holdout.json", "arxpr2_25_metadataset_train.json",
        "arxpr2_25_metadataset_holdout.json"
    ]
    files_to_process_individually = []
    files_to_process_as_bulk = []
    for f in all_files_found:
        if any(pattern in os.path.basename(f) for pattern in bulk_file_patterns):
            files_to_process_as_bulk.append(f)
        else:
            files_to_process_individually.append(f)

    print(f"[DB_UTILS] Processing {len(files_to_process_individually)} individual JSON files...")
    for file_path in files_to_process_individually:
        metadata = _extract_metadata_for_db(file_path)
        if metadata:
            metadata_batch.append(metadata)
            processed_count += 1
            if len(metadata_batch) >= batch_size:
                batch_upsert_datasets(metadata_batch)
                print(f"[DB_UTILS] Processed batch of {len(metadata_batch)} individual files (Total: {processed_count})...")
                metadata_batch = []
    if metadata_batch:
        batch_upsert_datasets(metadata_batch)
        print(f"[DB_UTILS] Processed final batch of {len(metadata_batch)} individual files (Total: {processed_count}).")
        metadata_batch = []

    print(f"[DB_UTILS] Processing {len(files_to_process_as_bulk)} bulk JSON files...")
    for file_path in files_to_process_as_bulk:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: bulk_data = json.load(f)
            if isinstance(bulk_data, dict):
                print(f"[DB_UTILS] Processing entries in bulk file: {os.path.basename(file_path)}")
                for pmid, entry_data in bulk_data.items():
                    metadata = _extract_metadata_from_bulk_entry(entry_data, pmid, os.path.basename(file_path))
                    if metadata:
                        metadata_batch.append(metadata)
                        bulk_processed_count += 1
                        if len(metadata_batch) >= batch_size:
                            batch_upsert_datasets(metadata_batch)
                            print(f"[DB_UTILS] Processed batch of {len(metadata_batch)} bulk entries (Total: {bulk_processed_count})...")
                            metadata_batch = []
            else:
                print(f"[WARN][DB_UTILS] Bulk file {file_path} is not a dictionary. Skipping.")
        except Exception as e:
            print(f"[ERROR][DB_UTILS] Error processing bulk file {file_path}: {e}")
    if metadata_batch:
        batch_upsert_datasets(metadata_batch)
        print(f"[DB_UTILS] Processed final batch of {len(metadata_batch)} bulk entries (Total: {bulk_processed_count}).")

    print("[DB_UTILS] Starting PubMed title update process...")
    update_titles_from_pubmed()

    print(f"[DB_UTILS] Finished scanning and updating. Processed {processed_count} individual files and {bulk_processed_count} entries from bulk files.")


# --- Data Retrieval Functions ---
def get_dataset_count(search_term: Optional[str] = None) -> int:
    """Gets the total number of datasets, optionally filtered by a search term across multiple fields."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "SELECT COUNT(*) FROM datasets"
        params = []

        if search_term and search_term.strip(): # Ensure search term is not empty
            search_term = search_term.strip()
            # Define columns to search
            search_columns = [
                'title', 'description', 'accession', 'pmid', 'organism',
                'study_type', 'hardware', 'organism_part', 'experimental_designs',
                'assay_by_molecule', 'technology', 'source'
            ]
            # Build WHERE clause dynamically
            where_clauses = [f"{col} LIKE ?" for col in search_columns if col] # Filter out potential None/empty strings if list was dynamic
            if where_clauses: # Check if there are clauses to add
                sql += " WHERE " + " OR ".join(where_clauses)
                # Add parameter for each clause
                params = [f'%{search_term}%'] * len(where_clauses)

        cursor.execute(sql, params)
        count = cursor.fetchone()[0]
        return count if count is not None else 0
    except sqlite3.Error as e:
        print(f"Database error in get_dataset_count: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_datasets_page(page_num: int, page_size: int, search_term: Optional[str] = None) -> List[sqlite3.Row]:
    """Gets a single page of datasets, optionally filtered by a search term across multiple fields."""
    conn = None
    offset = page_num * page_size
    try:
        conn = get_db_connection()
        # Use Row factory for dictionary-like access
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = "SELECT * FROM datasets"
        params = []

        if search_term and search_term.strip(): # Ensure search term is not empty
            search_term = search_term.strip()
             # Define columns to search (should match get_dataset_count)
            search_columns = [
                'title', 'description', 'accession', 'pmid', 'organism',
                'study_type', 'hardware', 'organism_part', 'experimental_designs',
                'assay_by_molecule', 'technology', 'source'
            ]
            # Build WHERE clause dynamically
            where_clauses = [f"{col} LIKE ?" for col in search_columns if col]
            if where_clauses: # Check if there are clauses to add
                sql += " WHERE " + " OR ".join(where_clauses)
                 # Add parameter for each clause
                params = [f'%{search_term}%'] * len(where_clauses)

        # Add ORDER BY, LIMIT and OFFSET - these are safe with f-strings as they are integers
        sql += f" ORDER BY last_updated DESC LIMIT {int(page_size)} OFFSET {int(offset)}"

        #print(f"[DB UTILS get_datasets_page] SQL: {sql}") # Uncomment for debugging SQL
        #print(f"[DB UTILS get_datasets_page] PARAMS: {params}") # Uncomment for debugging params

        cursor.execute(sql, params)
        datasets = cursor.fetchall()
        return datasets if datasets else []
    except sqlite3.Error as e:
        print(f"Database error in get_datasets_page: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_datasets_by_ids(ids):
    """Gets full dataset details for a list of IDs (accession or file_path)."""
    if not ids: return []
    conn = None
    datasets = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        placeholders = ', '.join('?' for _ in ids)
        sql = f"SELECT * FROM datasets WHERE accession IN ({placeholders})"
        print(f"[DB_UTILS] Fetching datasets by ID (Accession): {ids}")

        cursor.execute(sql, list(ids))
        datasets_by_accession = {dict(row)['accession']: dict(row) for row in cursor.fetchall()}

        found_accessions = set(datasets_by_accession.keys())
        missing_ids = [id_ for id_ in ids if id_ not in found_accessions]

        if missing_ids:
             print(f"[DB_UTILS] Accessions not found, trying file_path for: {missing_ids}")
             placeholders_path = ', '.join('?' for _ in missing_ids)
             sql_path = f"SELECT * FROM datasets WHERE file_path IN ({placeholders_path})"
             cursor.execute(sql_path, missing_ids)
             for row in cursor.fetchall():
                 ds_dict = dict(row)
                 dict_key = ds_dict.get('accession', ds_dict.get('file_path'))
                 if dict_key not in datasets_by_accession:
                      datasets_by_accession[dict_key] = ds_dict

    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error fetching by IDs: {e}")
        print(f"  SQL used: {sql}")
        print(f"  IDs: {list(ids)}")
    except Exception as e:
         print(f"[ERROR][DB_UTILS] Unexpected error fetching by IDs: {e}")
    finally:
        if conn: conn.close()

    print(f"[DB_UTILS] Returning {len(datasets_by_accession)} datasets for IDs: {ids}")
    return list(datasets_by_accession.values())


if __name__ == '__main__':
    print(f"Database path: {DB_PATH}")
    init_db()
    print("[DB_UTILS] Running example scan...")
    scan_directories = [
        os.path.join(project_root, 'data'), # Scan data dir for bulk files
        os.path.join(project_root, 'data', 'arxpr'), # Scan for individual ArrayExpress files
        "/mnt/data/upcast/data/arxpr", # Scan mounted data path
        USER_DATASETS_DIR # Scan user saved dir
    ]
    valid_scan_dirs = [d for d in scan_directories if os.path.isdir(d)]
    if valid_scan_dirs:
        scan_and_update_db(valid_scan_dirs)
    else:
        print("[DB_UTILS] No valid directories found for example scan.")
    print("[DB_UTILS] DB Utils script finished.")
