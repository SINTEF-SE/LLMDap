import streamlit as st
import os
import sys
import math
from typing import List, Dict, Any

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'llm_ui', 'app')) # Add app dir for db_utils

# Import database utility functions
try:
    import db_utils
except ModuleNotFoundError:
    st.error("Could not import db_utils.py. Make sure it's in the llm_ui/app directory.")
    st.stop()
except Exception as e:
    st.error(f"Error importing db_utils: {e}")
    st.stop()

def show():
    st.title("Dataset Browser")
    st.write("Browse and select datasets for analysis using the database index.")

    # --- Initialization ---
    # Initialize database (creates table if needed)
    try:
        db_utils.init_db()
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        st.stop()

    # Initialize session state variables
    if 'current_dataset_page' not in st.session_state: st.session_state.current_dataset_page = 0
    # Use a dictionary to store selected items {id: dataset_dict}
    if 'globally_selected_items' not in st.session_state: st.session_state.globally_selected_items = {}
    # 'datasets' now holds the current page's data from DB, loaded below
    # 'selected_datasets' will hold the full data for selected items when navigating

    # --- Settings ---
    datasets_per_page = 10 # Default, can be made configurable
    with st.expander("Settings & Actions", expanded=True): # Expand by default
        datasets_per_page = st.slider("Datasets per page:", min_value=5, max_value=50, value=datasets_per_page, step=5, key="datasets_per_page_slider")
        if st.button("🔄 Rescan Directories & Update Index"):
            st.write("[DEBUG] Scan button clicked!") # Confirm button press
            with st.spinner("Scanning directories and updating database... This may take time."):
                # Define directories to scan - ensure USER_DATASETS_DIR exists
                os.makedirs(db_utils.USER_DATASETS_DIR, exist_ok=True)
                # --- IMPORTANT: Add paths to your actual data here ---
                scan_directories = [
                    # Common paths based on project structure - ADJUST AS NEEDED
                    os.path.join(project_root, 'data'), # For bulk files like arxpr_simplified.json
                    os.path.join(project_root, 'data', 'arxpr'), # If individual jsons are here
                    "/mnt/data/upcast/data/arxpr", # Path mentioned in data scripts
                    db_utils.USER_DATASETS_DIR # Always scan user datasets
                ]
                # Filter out non-existent directories before scanning
                valid_scan_dirs = [d for d in scan_directories if os.path.isdir(d)]
                print(f"[DATASETS_PAGE] Valid directories to scan: {valid_scan_dirs}") # Debug print
                if not valid_scan_dirs:
                     st.warning("No valid directories found to scan.")
                else:
                     db_utils.scan_and_update_db(valid_scan_dirs)
                     st.success("Dataset index updated!")
                     # Clear current page data to force reload from DB
                     st.session_state.datasets = []
                     # Reset page to 0 after scan to avoid index errors if total pages decreased
                     st.session_state.current_dataset_page = 0
                     st.rerun() # Reload data after scan

    # --- Search ---
    st.header("Browse Datasets")
    search_term = st.text_input("Search datasets (by title, description, organism, etc.):", key="dataset_search")

    # --- Data Loading & Pagination ---
    try:
        total_datasets = db_utils.get_dataset_count(search_term if search_term else None)
        print(f"[DATASETS_PAGE] Total matching datasets from DB: {total_datasets}") # Print count
        total_pages = max(1, math.ceil(total_datasets / datasets_per_page))

        # Ensure current page is valid after search or page size change
        if st.session_state.current_dataset_page >= total_pages:
            st.session_state.current_dataset_page = max(0, total_pages - 1)

        st.write(f"Found {total_datasets} matching datasets in the database.")

        datasets_on_page = []
        if total_datasets > 0:
            with st.spinner(f"Loading page {st.session_state.current_dataset_page + 1}..."):
                datasets_on_page = db_utils.get_datasets_page(
                    st.session_state.current_dataset_page,
                    datasets_per_page,
                    search_term if search_term else None
                )
            print(f"[DATASETS_PAGE] Fetched {len(datasets_on_page)} datasets for page {st.session_state.current_dataset_page + 1}") # Print fetch result
        # Store current page's data in session state
        st.session_state.datasets = datasets_on_page

    except Exception as e:
        st.error(f"Error interacting with database: {e}")
        st.error("Please ensure the database file exists and the schema is correct. Try running the 'Scan/Update' button.")
        return # Stop execution if DB fails

    # --- Pagination Controls ---
    if total_datasets > datasets_per_page:
        st.write(f"Page {st.session_state.current_dataset_page + 1} of {total_pages}")
        cols = st.columns([1, 1, 2, 1, 1])
        disable_prev = st.session_state.current_dataset_page == 0
        disable_next = st.session_state.current_dataset_page >= total_pages - 1

        if cols[0].button("⏮️ First", key="pg_first", disabled=disable_prev):
            st.session_state.current_dataset_page = 0
            st.rerun()
        if cols[1].button("◀️ Prev", key="pg_prev", disabled=disable_prev):
            st.session_state.current_dataset_page -= 1
            st.rerun()

        # Page number input - ensure value updates trigger rerun
        page_number_input = cols[2].number_input(
            "Go to page", min_value=1, max_value=total_pages,
            value=st.session_state.current_dataset_page + 1, key="pg_num_input"
        )
        # Check if the input value changed and update state + rerun
        if page_number_input != st.session_state.current_dataset_page + 1:
             st.session_state.current_dataset_page = page_number_input - 1
             # Clear current datasets to force reload for the new page
             st.session_state.datasets = []
             st.rerun()

        if cols[3].button("Next ▶️", key="pg_next", disabled=disable_next):
            st.session_state.current_dataset_page += 1
            st.rerun()
        if cols[4].button("Last ⏭️", key="pg_last", disabled=disable_next):
            st.session_state.current_dataset_page = total_pages - 1
            st.rerun()

    # --- NEW: Select/Deselect All Buttons for Current Page ---
    if datasets_on_page: # Only show if there are datasets on the page
        select_cols = st.columns(2)
        with select_cols[0]:
            if st.button("Select All on This Page", key="select_all_page"):
                for dataset_dict in st.session_state.datasets:
                    dataset = dict(dataset_dict) # Ensure it's a dict
                    dataset_id_for_key = dataset.get('file_path')
                    dataset_id_for_selection = dataset.get('accession', dataset_id_for_key)
                    if dataset_id_for_selection: # Ensure we have an ID
                        st.session_state.globally_selected_items[dataset_id_for_selection] = dataset
                st.rerun()
        with select_cols[1]:
            if st.button("Deselect All on This Page", key="deselect_all_page"):
                for dataset_dict in st.session_state.datasets:
                    dataset = dict(dataset_dict) # Ensure it's a dict
                    dataset_id_for_key = dataset.get('file_path')
                    dataset_id_for_selection = dataset.get('accession', dataset_id_for_key)
                    if dataset_id_for_selection in st.session_state.globally_selected_items:
                        try:
                            del st.session_state.globally_selected_items[dataset_id_for_selection]
                        except KeyError:
                            pass # Item already removed
                st.rerun()

    # --- Display Datasets ---
    with st.expander("Available Datasets", expanded=True):
        if datasets_on_page:
            for i, dataset_dict in enumerate(datasets_on_page):
                dataset = dict(dataset_dict) # Convert from Row object
                with st.container():
                    if i > 0: st.markdown("---")
                    col1, col2, col3 = st.columns([5, 4, 1])

                    with col1:
                        title = dataset.get('title', 'Unknown Title')
                        source = dataset.get('source', 'unknown') # Get source from DB data
                        print(f"[DATASETS_PAGE] Displaying dataset: Accession={dataset.get('accession')}, Source='{source}'") # Debug source value
                        # Determine icon and suffix based on source
                        if source == 'user_provider': icon = "💾"; title_suffix = "(User Saved)"
                        elif source == 'arrayexpress': icon = "🔬"; title_suffix = "" # No suffix for default source
                        elif source == 'bulk_processed': icon = "📚"; title_suffix = "(Bulk Processed)"
                        # Add more source types if needed (e.g., 'uploaded')
                        else: icon = "❓"; title_suffix = f"({source or 'Unknown Source'})" # Fallback

                        st.markdown(f"#### {icon} {title} {title_suffix}")
                        desc = dataset.get('description', '')
                        if desc and len(desc) > 100: desc = desc[:100] + "..."
                        if desc: st.markdown(f"*{desc}*")

                    with col2:
                        # --- Existing Fields ---
                        organism = dataset.get('organism') # Removed default 'Unknown' to check existence
                        if organism and organism != 'N/A': # Check if value exists and is not 'N/A'
                            st.markdown(f"🧬 **Organism:** {organism}")

                        study_type = dataset.get('study_type')
                        if study_type and study_type != 'N/A':
                            st.markdown(f"🔬 **Study type:** {study_type}")

                        # --- New Fields ---
                        organism_part = dataset.get('organism_part')
                        if organism_part and organism_part != 'N/A':
                            st.markdown(f"🔬 **Organism Part:** {organism_part}") # Using same icon as study type, adjust if needed

                        exp_design = dataset.get('experimental_designs')
                        if exp_design and exp_design != 'N/A':
                             # Shorten if too long for display
                            display_design = exp_design[:40] + '...' if len(exp_design) > 40 else exp_design
                            st.markdown(f"📊 **Design:** {display_design}")

                        assay = dataset.get('assay_by_molecule')
                        if assay and assay != 'N/A':
                            st.markdown(f"🧪 **Assay:** {assay}")

                        hardware = dataset.get('hardware')
                        if hardware and hardware != 'N/A':
                             display_hw = hardware[:40] + '...' if len(hardware) > 40 else hardware
                             st.markdown(f"💻 **Hardware:** {display_hw}")

                        # --- Existing ID/PMID ---
                        st.markdown(f"📋 **ID:** `{dataset.get('accession', 'Unknown')}`")
                        pmid = dataset.get('pmid')
                        pmid_str = str(pmid) if pmid is not None else ''
                        # Improved check for valid-looking PMIDs
                        if pmid_str and pmid_str.isdigit() and len(pmid_str) > 5:
                             st.markdown(f"📝 **PMID:** `{pmid_str}`")
                        # Optional: Log ignored PMIDs for debugging
                        # elif pmid and pmid_str not in ['schema', 'bulk', 'unknown', 'uploaded'] and not pmid_str.startswith('provider_'):
                        #    print(f"[Dataset Display] Ignored invalid PMID value: {pmid_str} for Accession: {dataset.get('accession')}")


                    with col3:
                        # Use file_path (PRIMARY KEY) for the most stable unique ID for the key
                        dataset_id_for_key = dataset.get('file_path')
                        # Use accession for the selection dictionary key (more user-friendly if available)
                        dataset_id_for_selection = dataset.get('accession', dataset_id_for_key)

                        if not dataset_id_for_key: continue # Skip if no file_path (shouldn't happen)

                        # Use file_path for the checkbox key to ensure uniqueness
                        checkbox_key = f"select_{dataset_id_for_key}"
                        is_selected = dataset_id_for_selection in st.session_state.globally_selected_items

                        # Update selection dictionary on change using a callback
                        def update_selection(selection_id, ds_dict, key):
                            # Check the state of the specific checkbox that changed
                            if st.session_state[key]:
                                st.session_state.globally_selected_items[selection_id] = ds_dict
                            elif selection_id in st.session_state.globally_selected_items:
                                # Use try-except for safety when deleting
                                try:
                                    del st.session_state.globally_selected_items[selection_id]
                                except KeyError:
                                    pass # Item already removed, do nothing

                        st.checkbox("Select", value=is_selected, key=checkbox_key, # Key uses file_path
                                    on_change=update_selection, args=(dataset_id_for_selection, dataset, checkbox_key)) # Args use accession/fallback ID

        elif search_term:
            st.info("No datasets match your search criteria.")
        else:
            st.info("No datasets found in the database. Try the 'Scan/Update Dataset Index' button.")

    # --- Selected Datasets Section ---
    st.header("Selected Datasets", divider="gray")
    num_selected = len(st.session_state.globally_selected_items)

    if num_selected > 0:
        st.write(f"You have selected {num_selected} datasets:")

        # Display selected datasets with Remove buttons
        selected_items_list = list(st.session_state.globally_selected_items.items()) # Get (id, dict) pairs

        items_to_remove = [] # Store IDs to remove after iteration

        # Use columns for better layout if many items are selected
        num_cols = 3
        cols = st.columns(num_cols)

        for i, (item_id, ds) in enumerate(selected_items_list):
            col_idx = i % num_cols
            with cols[col_idx]:
                 # Determine icon based on source
                source = ds.get('source', 'unknown')
                if source == 'user_provider': icon = "💾"
                elif source == 'arrayexpress': icon = "🔬"
                elif source == 'bulk_processed': icon = "📚"
                else: icon = "❓"

                title = ds.get('title', 'Unknown Title')
                if len(title) > 25: title = title[:25] + "..." # Shorter truncation for columns

                # Display Title, Accession ID, and Remove button in one line
                button_key = f"remove_{item_id}"
                # Use a more compact display with the button integrated
                remove_clicked = st.button(f"❌ {icon} {title} (`{ds.get('accession', 'N/A')}`)", key=button_key, help=f"Remove {title}")
                if remove_clicked:
                     items_to_remove.append(item_id) # Mark for removal

        # Remove items outside the loop to avoid modifying dict during iteration
        if items_to_remove:
            for item_id in items_to_remove:
                 if item_id in st.session_state.globally_selected_items:
                     del st.session_state.globally_selected_items[item_id]
            st.rerun() # Rerun to update the selected list display and checkbox states

        # Only show the "Ask Questions" button if items remain selected
        if st.session_state.globally_selected_items:
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                # The button to proceed to Consumer Q&A
                if st.button("Ask Questions About Selected Datasets", use_container_width=True):
                    # Prepare the list of selected dataset dictionaries for the next page
                    # Fetch full details from DB using the stored IDs (accessions or file_paths used as keys)
                    selected_ids = list(st.session_state.globally_selected_items.keys())
                    if selected_ids:
                         with st.spinner("Loading selected dataset details..."):
                              # Fetch full details using the IDs (which are accessions or file_paths)
                              # get_datasets_by_ids should handle searching by accession primarily
                              full_selected_datasets = db_utils.get_datasets_by_ids(selected_ids)
                              # Store the list of full dictionaries
                              st.session_state.selected_datasets = full_selected_datasets
                         st.session_state.show_page = "Consumer Q&A" # Assuming this triggers navigation in app.py
                         st.rerun()
                    else:
                         st.warning("No datasets currently selected.") # Should not happen if button is visible

    else:
        st.info("Please select at least one dataset to continue")
