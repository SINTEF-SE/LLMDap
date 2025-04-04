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
    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error during single upsert for {metadata.get('file_path')}: {e}")
    except Exception as e:
         print(f"[ERROR][DB_UTILS] Unexpected error during single upsert for {metadata.get('file_path')}: {e}")
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
        cursor.executemany(sql, data_tuples)
        conn.commit()
        inserted_count = len(metadata_list)
        # print(f"[DB_UTILS] Batch upserted {inserted_count} records.")
    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error during batch upsert: {e}")
        # Optionally try individual upserts as fallback here
    except Exception as e:
         print(f"[ERROR][DB_UTILS] Unexpected error during batch upsert: {e}")
    finally:
        if conn:
            conn.close()
    return inserted_count

# --- Metadata Extraction Helpers ---
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
    metadata.update({k: None for k in [ # Initialize all other DB fields to None
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
            # Simplified extraction from potential ArrayExpress structure
            metadata['title'] = data.get('title', metadata['title'])
            metadata['description'] = data.get('description', metadata['description'])
            metadata['organism'] = data.get('organism', metadata['organism'])
            metadata['study_type'] = data.get('study_type', data.get('study type', metadata['study_type']))

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
                     # Simplified extraction for other fields - add more as needed
                     subsections = section.get('subsections', [])
                     if isinstance(subsections, dict): subsections = [subsections]
                     if isinstance(subsections, list):
                          for sub in subsections:
                              if not isinstance(sub, dict): continue
                              sub_attributes = sub.get('attributes', [])
                              if isinstance(sub_attributes, dict): sub_attributes = [sub_attributes]
                              for attr in sub_attributes:
                                   if isinstance(attr, dict) and 'name' in attr and 'value' in attr:
                                       attr_name = attr['name'].lower()
                                       if attr_name == 'organism part' and not metadata['organism_part']: metadata['organism_part'] = attr['value']
                                       # Add other fields like hardware, technology, etc.

    # Set default title if still None (will be updated later by PubMed fetch)
    if not metadata.get('title'): metadata['title'] = f"Dataset {accession}"
    if not metadata.get('organism'): metadata['organism'] = "Unknown"
    if not metadata.get('study_type'): metadata['study_type'] = "Unknown"

    # Convert all values to string for consistency before returning (but keep None as None)
    for key in metadata:
        if metadata[key] is not None:
            metadata[key] = str(metadata[key])
        elif key in ['organism', 'study_type', 'title', 'description', 'accession', 'pmid', 'source']:
             metadata[key] = "Unknown"

    return metadata

def _extract_metadata_from_bulk_entry(entry_data, pmid, file_basename):
    """Helper to extract metadata from a single entry within a bulk JSON file."""
    metadata = {
        'pmid': pmid,
        'title': f"PMID: {pmid} from {file_basename}",
        'accession': f'BULK-{pmid}',
        'organism': "Unknown", 'study_type': "Unknown",
        'description': f"Entry for PMID {pmid} from bulk file: {file_basename}",
        'source': 'bulk_processed',
        'file_path': f"{file_basename}#PMID{pmid}", # Pseudo-path
    }
    metadata.update({k: None for k in [ # Initialize other fields
        'hardware', 'organism_part', 'experimental_designs', 'assay_by_molecule',
        'technology', 'sample_count', 'release_date', 'experimental_factors']})

    if isinstance(entry_data, dict):
        field_mappings = { # Map canonical names to potential indexed keys
            'organism': ['organism_16', 'organism_17'], 'study_type': ['study_type_18'],
            'hardware': ['hardware_4'], 'organism_part': ['organism_part_5'],
            'experimental_designs': ['experimental_designs_10'], 'assay_by_molecule': ['assay_by_molecule_14'],
            'technology': ['technology_15'], 'sample_count': ['sample_count_13'],
            'release_date': ['releasedate_12'], 'experimental_factors': ['experimental_factors_20'],
            'title': ['title_11']
        }
        for canonical_field, indexed_keys in field_mappings.items():
            found_value = None
            for key in indexed_keys:
                if key in entry_data and entry_data[key]:
                    value = entry_data[key]
                    if isinstance(value, list):
                        if len(value) > 0: found_value = str(value[0])
                    else:
                        found_value = str(value)
                    if found_value:
                        metadata[canonical_field] = found_value
                        break
            if not found_value and canonical_field in ['organism', 'study_type']:
                 metadata[canonical_field] = "Unknown"

    # Set default title if still None (will be updated later)
    if not metadata.get('title') or metadata['title'].startswith("PMID:"):
        metadata['title'] = f"PMID: {pmid} from {file_basename}"
    if not metadata.get('organism'): metadata['organism'] = "Unknown"
    if not metadata.get('study_type'): metadata['study_type'] = "Unknown"

    # Convert all values to string
    for key in metadata:
        if metadata[key] is not None: metadata[key] = str(metadata[key])
        elif key in ['organism', 'study_type', 'title', 'description', 'accession', 'pmid', 'source']:
             metadata[key] = "Unknown"
    return metadata

# --- PubMed Title Fetching ---
def _fetch_pubmed_titles_batch(pmids):
    """Fetches titles for a batch of PMIDs."""
    titles = {}
    if not pmids: return titles
    pmid_list = ",".join(filter(None, pmids))
    if not pmid_list: return titles

    print(f"[DB_UTILS] Fetching titles for {len(pmids)} PMIDs...")
    try:
        pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid_list}&retmode=xml"
        response = requests.get(pubmed_url, timeout=20) # Longer timeout for batch
        time.sleep(0.5) # Be nice to NCBI
        if response.status_code == 200:
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
    print(f"[DB_UTILS] Fetched {len(titles)} titles.")
    return titles

def update_titles_from_pubmed(batch_size=100):
    """Queries DB for entries needing titles and updates them in batches."""
    conn = None
    updated_count = 0
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Find entries with placeholder titles and valid PMIDs
        sql_select = """
            SELECT file_path, pmid FROM datasets
            WHERE pmid IS NOT NULL AND pmid != 'unknown' AND pmid != ''
            AND (title IS NULL OR title = 'Unknown' OR title LIKE 'Dataset %' OR title LIKE 'PMID:%')
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
                    updates.append((title, pmid_map[pmid])) # Prepare (title, file_path) for update

            if updates:
                sql_update = "UPDATE datasets SET title = ? WHERE file_path = ?"
                cursor.executemany(sql_update, updates)
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
    batch_size = 500 # Process DB inserts in batches
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
                metadata_batch = [] # Clear batch
    # Process any remaining individual files
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
    # Process any remaining bulk entries
    if metadata_batch:
        batch_upsert_datasets(metadata_batch)
        print(f"[DB_UTILS] Processed final batch of {len(metadata_batch)} bulk entries (Total: {bulk_processed_count}).")

    # --- Update Titles from PubMed ---
    print("[DB_UTILS] Starting PubMed title update process...")
    update_titles_from_pubmed()

    print(f"[DB_UTILS] Finished scanning and updating. Processed {processed_count} individual files and {bulk_processed_count} entries from bulk files.")


# --- Data Retrieval Functions ---
def get_dataset_count(search_term=None):
    """Gets the total count of datasets, optionally filtered by search term."""
    conn = None
    count = 0
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "SELECT COUNT(*) FROM datasets"
        params = []
        if search_term:
            search_pattern = f"%{search_term}%"
            # Ensure all columns used in WHERE clause exist in the table
            # Added organism_part, experimental_factors, technology to the search
            sql += """
                WHERE (accession LIKE ? OR pmid LIKE ? OR title LIKE ? OR
                      organism LIKE ? OR study_type LIKE ? OR description LIKE ? OR
                      organism_part LIKE ? OR technology LIKE ? OR experimental_factors LIKE ?)
            """
            params = [search_pattern] * 9 # Increased count to match WHERE clauses

        cursor.execute(sql, params)
        result = cursor.fetchone()
        if result:
             count = result[0]
        # print(f"[DB_UTILS] get_dataset_count (term: {search_term}): {count}") # Keep this less verbose
    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error counting datasets: {e}")
    finally:
        if conn:
            conn.close()
    return count

def get_datasets_page(page_number, page_size, search_term=None):
    """Gets a specific page of datasets, optionally filtered."""
    conn = None
    datasets = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        offset = page_number * page_size

        # Select columns needed for display and identification
        # Added organism_part, technology etc. for potential display/use and search verification
        select_cols = [
            'file_path', 'accession', 'pmid', 'title', 'organism', 'study_type',
            'description', 'source', 'organism_part', 'technology', 'sample_count',
            'release_date', 'experimental_factors', 'hardware', 'experimental_designs',
            'assay_by_molecule'
        ]
        sql = f"SELECT {', '.join(select_cols)} FROM datasets"
        params = []

        if search_term:
            search_pattern = f"%{search_term}%"
            # Ensure all columns used in WHERE clause exist
            sql += """
                WHERE (accession LIKE ? OR pmid LIKE ? OR title LIKE ? OR
                      organism LIKE ? OR study_type LIKE ? OR description LIKE ? OR
                      organism_part LIKE ? OR technology LIKE ? OR experimental_factors LIKE ?)
            """
            params = [search_pattern] * 9 # Increased count

        sql += " ORDER BY accession LIMIT ? OFFSET ?" # Order consistently
        params.extend([page_size, offset])

        # print(f"[DB_UTILS] Executing SQL: {sql}") # Debug SQL
        # print(f"[DB_UTILS] With Params: {params}") # Debug Params
        cursor.execute(sql, params)
        datasets = [dict(row) for row in cursor.fetchall()]
        # print(f"[DB_UTILS] Fetched {len(datasets)} datasets for page {page_number}") # Keep less verbose
    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error fetching page: {e}")
        print(f"  SQL: {sql}")
        print(f"  Params: {params}")
    finally:
        if conn:
            conn.close()
    return datasets

def get_datasets_by_ids(ids):
    """Gets full dataset details for a list of IDs (accession or file_path)."""
    if not ids:
        return []
    conn = None
    datasets = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        placeholders = ', '.join('?' for _ in ids)
        # Query primarily by accession as it's more likely to be the stable ID used in selection
        sql = f"SELECT * FROM datasets WHERE accession IN ({placeholders})"
        print(f"[DB_UTILS] Fetching datasets by ID (Accession): {ids}") # Debug fetch

        cursor.execute(sql, list(ids)) # Ensure ids is a list
        datasets_by_accession = {dict(row)['accession']: dict(row) for row in cursor.fetchall()}

        # Find IDs that weren't matched by accession
        found_accessions = set(datasets_by_accession.keys())
        missing_ids = [id_ for id_ in ids if id_ not in found_accessions]

        # Try matching missing IDs by file_path
        if missing_ids:
             print(f"[DB_UTILS] Accessions not found, trying file_path for: {missing_ids}")
             placeholders_path = ', '.join('?' for _ in missing_ids)
             sql_path = f"SELECT * FROM datasets WHERE file_path IN ({placeholders_path})"
             cursor.execute(sql_path, missing_ids)
             for row in cursor.fetchall():
                 # Add only if not already found via accession (avoids duplicates if ID matches both)
                 ds_dict = dict(row)
                 # Use file_path as key if accession was missing/duplicate from file_path search
                 dict_key = ds_dict.get('accession', ds_dict.get('file_path'))
                 if dict_key not in datasets_by_accession:
                      datasets_by_accession[dict_key] = ds_dict

    except sqlite3.Error as e:
        print(f"[ERROR][DB_UTILS] Database error fetching by IDs: {e}")
        print(f"  SQL used: {sql}") # Print SQL that might have failed
        print(f"  IDs: {list(ids)}")
    except Exception as e:
         print(f"[ERROR][DB_UTILS] Unexpected error fetching by IDs: {e}")
    finally:
        if conn:
            conn.close()

    # Return the found datasets as a list of dictionaries
    print(f"[DB_UTILS] Returning {len(datasets_by_accession)} datasets for IDs: {ids}") # Debug return
    return list(datasets_by_accession.values())


if __name__ == '__main__':
    # Example usage: Initialize DB when script is run directly
    print(f"Database path: {DB_PATH}")
    init_db()
    # Example scan (adjust paths as needed for your environment)
    print("[DB_UTILS] Running example scan...")
    scan_directories = [
        # os.path.join(project_root, 'data'), # Scan original data dir - uncomment if needed
        # os.path.join(project_root, 'data', 'arxpr'), # Scan original data dir - uncomment if needed
        USER_DATASETS_DIR # Scan user saved dir
    ]
    # scan_and_update_db(scan_directories) # Commented out by default
    print("[DB_UTILS] DB Utils script finished.")
