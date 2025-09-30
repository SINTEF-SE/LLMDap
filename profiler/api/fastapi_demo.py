from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Path, Depends
from fastapi.responses import FileResponse, JSONResponse
from enum import Enum
import os
import logging
import uuid
import pandas as pd
from ydata_profiling import ProfileReport
from typing import Optional
import requests
import json
import shutil

import sys
from pathlib import Path


# Add project root to Python path
#sys.path.append(str(Path(__file__).resolve().parents[2]))

#from profiler.api.llmdap_options import LLMDapOptions

#import profiler.api.llmdap_options.LLMDapOptionsForm

# For LLMDap
#import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add root directory (llmdap/profiler)
import run_inference
from api.llmdap_options import LLMDapOptions
from api.llmdap_options import AppName


app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# A class to encapsulate the optional parameters for the post(generate_profile)
#class LLMDapOptions:
#    def __init__(
#        self,
#        path: Optional[str] = Form(None),
#        url: Optional[str] = Form(None),
#        similarity_k: Optional[int] = Form(None),
#        field_info_to_compare: Optional[str] = Form(None),
#        schema: Optional[UploadFile] = File(None)
#    ):
#        self.path = path
#        self.url = url
#        self.similarity_k = 5 # default
#        self.field_info_to_compare = "choices" # default 
#        from metadata_schemas.arxpr2_schema import Metadata_form as schema
#        self.schema = schema

# Enum to represent available apps (profilers)
#class AppName(str, Enum):
#    App1 = "Ydata"
#    App2 = "Abstat"
#    App3 = "LLMDap"


# Function to run selected app
#def run_App1(file: UploadFile):
def run_App1(file: Path, url: str, flag: str):
    # Display selected app name and file
    logger.info(f"Selected App - run_App1: Ydata")
    logger.info(f"flag: {flag}")

    # Generate profile using Ydata profiler
    if flag=="url":
        # download the file locally
        print("[get a url for profiling]")
        # Fetch file from URL
        print("[url:] ", file_url)
        try:
            if url.startswith(("http://", "https://")):
    
                response = requests.get(file_url)
                response.raise_for_status()

                filename = file_url.split("/")[-1] or "downloaded_file"
                file_path = os.path.join(UPLOAD_DIR, filename)
                print("[url:] ", file_path)

                # Save downloaded file
                with open(file_path, "wb") as f:
                    f.write(response.content)
        except requests.RequestException as e:
            return JSONResponse({"error": f"Failed to fetch file: {str(e)}"}, status_code=400)
    else:
        file_path = file
  
    df = pd.read_csv(file_path, sep=';', low_memory=False, on_bad_lines='warn')
    logger.info("Profiling using Ydata profiler")
    profile = ProfileReport(df, title="Ydata Profiling Report")
    logger.info("Profile report generated")
    # Generate the filename for the output file
    json_profile = f"{file_path.split('.')[0]}_profile.json"
    json_profile_name = os.path.basename(json_profile)
    output_file_path = f"all_results/{json_profile_name}"
    # Create output_files directory if it doesn't exist
    #os.makedirs("output_files", exist_ok=True)

    logger.info("the json profile name is: " + json_profile)
    #jsonfile = profile.to_file(json_profile)

    jsonfile = profile.to_file(output_file_path)
    output_filename = output_file_path
 
    return output_file_path, output_filename

# Function to run App2 -- TO BE UPDATED
def run_App2(file: Path, url: str, flag: str):
    # Display selected app name and file
    logger.info(f"Selected App: Abstat")
    logger.info(f"File: {file.filename}")
    # Implement logic to run App1 using the uploaded file
    # For demonstration purposes, let's assume the app simply writes to a new file
    output_filename = f"App2_{str(uuid.uuid4())}.txt"
    output_file_path = f"{output_filename}"
    with open(output_file_path, "wb") as output_file:
        output_file.write(file.file.read())
    return output_file_path, output_filename

# Function to run App3 - LLMDap
# Upload the .xml file (paper) and call LLMDap pipeline and return the JSON file
#def run_App3(file: Path, url: str, flag: str, schema, similarity_k, field_info_to_compare):
def run_App3(file: Path, url: str, flag: str, options: LLMDapOptions):
    # Display selected app name and file
    print(f"Selected App: LLMDap")
    print(f"File: {file}")

    # Run the script and capture errors
    #output_filename = f"{file.split('.')[0]}.json"
    if flag=="file":
        output_filename = os.path.basename(f"{file.split('.')[0]}.json")
    else:
        output_filename = url.split("/")[-1] or "downloaded_file"
        if not output_filename.endswith(".json"):
            output_filename += ".json"

    output_file_path = f"all_results/{output_filename}"

    #paper_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/12093373/ascii"
    #TODO: user optionally upload a schema 
    from metadata_schemas.arxpr2_schema import Metadata_form as default_schema
    # parse the optional parameters
    similarity_k = options.similarity_k if options.similarity_k else 5
    print(f"similarity_k: {similarity_k}")
    field_info_to_compare = options.field_info_to_compare if options.field_info_to_compare else "choices"
    print(f"field_info_to_compare: {field_info_to_compare}")
    schema = options.schema.filename if options.schema else default_schema
    result = run_inference.call_inference(schema,
            # choose one to try out
            #parsed_paper_text = parsed_xml_paper_text,
            #raw_xml_paper_text = raw_xml_paper_text,
            paper_path = file,
            paper_url = url,
            similarity_k = similarity_k, #= 5,
            field_info_to_compare = field_info_to_compare # = "choices",
            )
    print("[fastapi - app3] the result is: ")
    #import pprint
    #pprint.pprint(result)

    # extract the result and save it into a JSOn file
    if result:
        first_key = next(iter(result), None)
        if first_key and isinstance(result[first_key], dict) and "filled_form" in result[first_key]:
            processed_data = {
                "form": result[first_key].get("filled_form", {})
                #"context": result[first_key].get("context", {})
            }
            serializable_form = processed_data["form"]
            if hasattr(serializable_form, 'dict'): serializable_form = serializable_form.dict()
            edited_json = json.dumps(serializable_form, indent=4, default=str)
            print("Pipeline completed successfully!")
        else:
            print("Pipeline output structure unexpected. Using raw output.")
            processed_data = {"form": result, "context": {}}
            edited_json = json.dumps(result, indent=4, default=str)
            print("Pipeline completed (structure might differ).")
    else:
        print("Pipeline execution failed or returned no output.")
        #error_message = "Pipeline execution failed."

    # Save result in a JSON file
    # Save to a JSON file
    #with open(output_file_path, "w") as json_file:
    #    json.dump(result, json_file, indent=4)  # `indent=4` for pretty formatting

    return output_file_path, output_filename

# Function to run selected app
#def run_app(selected_app: str, file: UploadFile):
#def run_app(selected_app: str, file: Path, url: str, flag: str, schema, similarity_k, field_info_to_compare):
def run_app(selected_app: str, file: Path, url: str, flag: str, options: LLMDapOptions):
    # Display selected app name and file
    logger.info(f"Selected App: {selected_app}")
    logger.info(f"File: {file}")

    # Call the appropriate function based on the selected app name
    if selected_app == AppName.App1.value:
        return run_App1(file, url, flag)
    elif selected_app == AppName.App2.value:
        return run_App2(file, url, flag)
    elif selected_app == AppName.App3.value:
        #return run_App3(file, url, flag, schema, similarity_k, field_info_to_compare)
        return run_App3(file, url, flag, options)
    else:
        raise ValueError("Invalid app name")

# Function for returning the list of available profiles
def handle_get_profiles(url: str, selected_profiler: str):
    # for demo purpose, return a statis list of profiles
    profiles_ydata = ["sample_csv_profile.json", "EV_Population_Data_Sample_profile.json"]
    profile_list = ["sample_csv_profile.json", "EV_Population_Data_Sample_profile.json", "profile_other_profilers.json"]

    if selected_profiler is not None and selected_profiler == "Ydata":
        lists = profiles_ydata
    else:
        lists = profile_list
    

    return lists 

# Endpoint to get list of available apps (profilers)
@app.get("/profile/get_profilers")
async def get_available_profilers():
    for app_name in AppName:
        print(f"AppName: {app_name}")
    return [app_name.value for app_name in AppName]

# Endpoint to generate profile using the selected app
@app.post("/profile/generate_profile_with_file")
async def generate_profile_for_local_file_with_selected_profiler(selected_app: str = Form(...), file: UploadFile = File(...)):
    logger.debug(f"Request Payload: selected_app={selected_app}, file={file.filename}")

    if selected_app not in [app_name.value for app_name in AppName]:
        return JSONResponse(content={"error": "Invalid app name"}, status_code=400)
    
    output_file_path, output_filename = run_app(selected_app, file)
    
    # Read content of the output file
    with open(output_file_path, "r") as output_file:
        file_content = output_file.read()
    
    # Return the filename and content of the output file
    return JSONResponse(content={"filename": output_filename, "file_content": file_content})


@app.post("/profile/generate_profile")
#async def upload_file_or_url(selected_app: str = Form(...), file: UploadFile = None, file_url: str = Form(None), similarity_k: Optional[int] = Form(None), field_info_to_compare: Optional[str] = Form(None), schema: Optional[UploadFile] = File(None)):
async def upload_file_or_url(selected_app: str = Form(...), file: UploadFile = None, file_url: str = Form(None), options: LLMDapOptions = Depends()):

    logger.info(f"Request Payload: selected_app={selected_app}")
    logger.info(f"Request Payload: similarity_k={options.similarity_k}")
    logger.info(f"Request Payload: selected_app={selected_app}")


    if selected_app not in [app_name.value for app_name in AppName]:
        return JSONResponse(content={"error": "Invalid app name"}, status_code=400)
    
    # parse the optional parameters
    #similarity_k = options.similarity_k,
    #field_info_to_compare = options.field_info_to_compare,
    #schema = options.schema.filename if options.schema else None

    UPLOAD_DIR = "output_files"
    #output_file_path, output_filename = run_app(selected_app, file)
    if file:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        #output_file_path, output_filename = run_app(selected_app, file_path, file_url, flag="file", schema, similarity_k, field_info_to_compare)
        output_file_path, output_filename = run_app(selected_app, file_path, file_url, flag="file", options = options)

        # Read content of the output file
        with open(output_file_path, "r") as output_file:
            file_content = output_file.read()
        
        # Return the filename and content of the output file
        return JSONResponse(content={"filename": output_filename, "file_content": file_content, "message": "Profile generated successfully!"})

    elif file_url:
        try:
            # Fetch file from URL
            print("[url:] ", file_url)
            if file_url.startswith(("http://", "https://")):

                #output_file_path, output_filename = run_app(selected_app, file, file_url, flag="url", schema, similarity_k, field_info_to_compare)
                output_file_path, output_filename = run_app(selected_app, file, file_url, flag="url", options = options)

                # Read content of the output file
                with open(output_file_path, "r") as output_file:
                    file_content = output_file.read()
                
                # Return the filename and content of the output file
                return JSONResponse(content={"filename": output_filename, "file_content": file_content, "message": "Profile generated successfully!"})
        
        except requests.RequestException as e:
            return JSONResponse({"error": f"Failed to fetch file: {str(e)}"}, status_code=400)

    return JSONResponse({"error": "No file or URL provided"}, status_code=400)


# retrive profile of a resource specified by url, optionally with the profiler used
@app.get("/profile/get_profile")
async def get_profile_from_url(url: str, profiler_name: Optional[str] = None):

    try:
        logger.info(f"Received URL: {url}")
        if url.startswith(("http://", "https://")):
            logger.info(f"This is a remote url")
        
            # Fetch file content from the URL
            #response = requests.get(url)
            #response.raise_for_status()  # Raise exception for any HTTP error

            # call the profiler to generate the profile OR return the profile with specified profiler
            # TODO: this code need to be updated according to the local configuration. 
            # TODO: case 1) the profile exists: simply return the profile for the resource specified
            # TODO: case 2) the profile does not exist: call the specified profiler to generate the profile

            # Return the content of the file in the response
            #return response.content

            # Call the function that handle this request
            return JSONResponse(content={"profiles": handle_get_profiles(url, profiler_name)})

           
        else:
            # this is a local file, mostly for demo purpose
            logger.info(f"This is a local file: {url}")
            if os.path.exists(url):
                # TODO: # call the profiler to generate the profile OR return the profile with specified profiler
                #with open(url, "rb") as file:
                #    return file.read()
                # Call the function that handle this request
                return JSONResponse(content={"profiles": handle_get_profiles(url, profiler_name)})
            else:
                print(f"File not found at local path: {url}")
                return None

    except Exception as e:
        return {"error": f"Failed to fetch file from URL: {url}. Error: {str(e)}"}


# Endpoint to update a profile at the specified URL with content from a local file
@app.put("/profile/update_profile")
async def update_profile_specified_by_url_with_local_file(url: str, file_content: UploadFile = File(..., description="Local file to update with")
):
    try:
        
        if url.startswith(("http://", "https://")):
            # TODO: update the remote url with the uploaded file
            #return JSONResponse(content={"message": f"File at '{url}' is a remote file"})
            # Use PUT request to update remote file
            response = requests.put(url, data=await file_content.read())
            if response.status_code == 200:
                return {"message": f"Remote file at '{url}' updated successfully"}
            else:
                raise HTTPException(status_code=response.status_code, detail=f"Failed to update remote file. Response: {response.text}")


        else:
            # This is a local file. Check if the provided URL is valid
            #url_path = Path(url)
            if not os.path.exists(url):
                raise HTTPException(status_code=404, detail="File not found at the specified URL.")

            # Check if a file is provided as content
            if not file_content:
                raise HTTPException(status_code=400, detail="No file provided as content.")
            
            # Read the content of the uploaded file
            file_bytes = await file_content.read()

            # Write the content of the new file to the specified URL
            with open(url, "wb") as f:
                f.write(file_bytes)

            return JSONResponse(content={"message": f"File at '{url}' updated successfully"})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to update file. Error: {str(e)}"}, status_code=500)

# Endpoint to delete profile specified by URL
@app.delete("/profile/delete_profile")
async def delete_profile_specified_by_url(url: str):
    
    try:
        logger.info(f"Received URL: {url}")
        if url.startswith(("http://", "https://")):
            # This is a remote URL
            logger.info(f"handle remote URL, url={url}")
            print(f"handle remote URL, url={url}")
            response = requests.delete(url)
            if response.status_code == 200:
                return JSONResponse(content={"message": f"File at '{url}' deleted successfully"})
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        else:     
            # This is a local path. Check if the provided path exists
            if not os.path.exists(url):
                raise HTTPException(status_code=404, detail="File not found at the specified URL.")

            # Delete the file
            os.remove(url)

            return JSONResponse(content={"message": f"File at '{url}' deleted successfully"})
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return JSONResponse(content={"error": f"Failed to delete file. Error: {str(e)}"}, status_code=500)



# Generate OpenAPI schema
from fastapi.openapi.utils import get_openapi

openapi_schema = get_openapi(
    title="UPCAST Data Profiling API",
    version="0.3.0",
    description="OpenAPI specification for the UPCAST Data Profiling API",
    routes=app.routes,
)

# Create the api folder if it does not exist
api_folder = "openapi"
if not os.path.exists(api_folder):
    os.makedirs(api_folder)

# Save the OpenAPI schema to a JSON file under the api folder
openapi_file = os.path.join(api_folder, "openapi.json")
with open(openapi_file, "w") as file:
    import json
    json.dump(openapi_schema, file)

# Set the current path to the parent of api
print("current path: ", os.getcwd())
if os.getcwd().endswith("api"):
    os.chdir("..")
print("current path - new: ", os.getcwd())

TEMP_DIR = "all_results"  # Directory for storing temporary files
os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure directory exists