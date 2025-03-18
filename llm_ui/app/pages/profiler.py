import os
import tempfile
import streamlit as st
import json
import requests
import sys

# Add the project root and profiler directories to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
profiler_dir = os.path.join(project_root, 'profiler')
sys.path.append(profiler_dir)

# Print paths for debugging
print(f"Project root: {project_root}")
print(f"Profiler directory: {profiler_dir}")

# Try to import call_inference at the module level
try:
    from profiler.run_inference import call_inference
except ImportError as e:
    print(f"Error importing run_inference: {str(e)}")
    call_inference = None

# Move these functions outside the show() function
def handle_input(uploaded_file, xml_url):
    """Handle uploaded file or URL input"""
    if uploaded_file is not None:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write uploaded file to the temp file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        return temp_path
    
    elif xml_url:
        try:
            # Download the file from URL
            response = requests.get(xml_url)
            response.raise_for_status()
            
            # Get filename from URL or use a default
            filename = xml_url.split('/')[-1] if '/' in xml_url else 'downloaded.xml'
            
            # Create a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, filename)
            
            # Write downloaded content to the temp file
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            return temp_path
        
        except Exception as e:
            st.error(f"Error downloading file from URL: {str(e)}")
            return None
    
    return None

def handle_schema(schema_file):
    """Handle schema file upload"""
    if schema_file is not None:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, schema_file.name)
        
        # Write uploaded file to the temp file
        with open(temp_path, "wb") as f:
            f.write(schema_file.getvalue())
        
        return temp_path
    
    # If no schema provided, use default
    try:
        from profiler.metadata_schemas.arxpr2_schema import Metadata_form
        return Metadata_form
    except ImportError as e:
        st.error(f"Could not import default schema: {str(e)}")
        return None
    
def run_pipeline(xml_path, schema_path, ff_model="llama3.1I-8b-q4"):
    """Run the profiling pipeline on the input file"""
    if call_inference is None:
        st.error("call_inference function could not be imported")
        return None
        
    try:
        if isinstance(schema_path, str):
            # Load schema from file
            with open(schema_path, 'r') as f:
                schema_json = json.load(f)
                # Would need to convert this JSON to a pydantic model
                st.error("Custom JSON schema support not yet implemented")
                return None
        else:
            # Use the provided schema class
            schema = schema_path
        
        # Run the inference with the specified model
        st.info(f"Using {ff_model} for paper analysis")
        
        # Call the inference function with the exact model name
        output = call_inference(
            schema=schema,
            paper_path=xml_path,
            similarity_k=5,
            field_info_to_compare="choices",
            ff_model=ff_model  # Pass the model name directly
        )
            
        # Get the first paper's data
        if output and len(output) > 0:
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
    st.title("Profiler")
    st.write("This is a standalone profiler page for testing purposes.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload an XML file", type=["xml"])
    
    if uploaded_file is not None:
        if st.button("Process File"):
            with st.spinner("Processing..."):
                xml_path = handle_input(uploaded_file, None)
                schema_path = handle_schema(None)
                result = run_pipeline(xml_path, schema_path)
                
                if result:
                    st.json(result)

