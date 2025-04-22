# Using BioC api, see https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/

import requests
import os

def save_xml(pmid, folder = "/mnt/data/upcast/data/all_xmls", encoding="ascii", source = "pmcoa"):
    """source (str) : pmcoa or pubmed
    """
    filename = f"{pmid}_{encoding}_{source}.xml"
    if filename in os.listdir(folder): # avoid requesting again
        print(f"[XML_FETCHER] File {filename} already exists in {folder}, skipping download")
        return 1
    
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/{source}.cgi/BioC_xml/{pmid}/{encoding}"
    print(f"[XML_FETCHER] Fetching from URL: {url}")
    
    try:
        page = requests.get(url)
        text = page.text
        
        # Print response status code and headers
        print(f"[XML_FETCHER] Response status: {page.status_code}")
        print(f"[XML_FETCHER] Response headers: {page.headers}")
        
        # Check for error conditions
        is_error = False
        content_type = page.headers.get('Content-Type', '').lower()
        
        # Check 1: Content type should be XML, not HTML
        if 'html' in content_type and 'xml' not in content_type:
            print(f"[XML_FETCHER] WARNING: Received HTML instead of XML content (Content-Type: {content_type})")
            is_error = True
            
        # Check 2: Look for error messages in content
        if '[Error]' in text or 'No result can be found' in text:
            print(f"[XML_FETCHER] ERROR: Response contains error message: {text[:200].replace(chr(10), ' ')}...")
            is_error = True
            
        # Check 3: Very short responses
        if len(text) < 50:
            print(f"[XML_FETCHER] ERROR: Response too short for PMID {pmid}: {text}")
            is_error = True
            
        # Try alternative source if error detected and we're using pmcoa
        if is_error and source == "pmcoa":
            print(f"[XML_FETCHER] Attempting to fetch from 'pubmed' source instead...")
            return save_xml(pmid, folder, encoding, "pubmed")
        elif is_error:
            print(f"[XML_FETCHER] ERROR: Both pmcoa and pubmed sources failed for PMID {pmid}")
            return 0
            
        print(f"[XML_FETCHER] Successfully fetched XML for PMID {pmid}, length: {len(text)} characters")
        
        # Print first 200 characters of response for debugging
        print(f"[XML_FETCHER] First 200 chars: {text[:200].replace(chr(10), ' ')}")
        
        output_path = f"{folder}/{pmid}_{encoding}_{source}.xml"
        with open(output_path, "w") as f:
            f.write(text)
        print(f"[XML_FETCHER] Saved XML to: {output_path}")
        return 1
    except Exception as e:
        print(pmid)
        print(e)
        return 0

if __name__ == "__main__":
    for pmid in [33495476, 35810190, 33789117, 35368039]: # nhrf examples
        save_xml(pmid)
