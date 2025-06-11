# Using BioC api, see https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/

import requests
import os

def save_xml(pmid, folder = "/mnt/data/upcast/data/all_xmls", encoding="ascii", source = "pmcoa"):
    """source (str) : pmcoa or pubmed
    """
    filename = f"{pmid}_{encoding}_{source}.xml"
    if filename in os.listdir(folder): # avoid requesting again
        return 1
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/{source}.cgi/BioC_xml/{pmid}/{encoding}"
    try:
        page = requests.get(url)
        text = page.text
        if len(text) < 50:
            print(pmid, text)
            return 0
        print(pmid, len(text))
        with open(f"{folder}/{pmid}_{encoding}_{source}.xml", "w") as f:
            f.write(text)
        return 1
    except Exception as e:
        print(pmid)
        print(e)
        return 0

if __name__ == "__main__":
    for pmid in [33495476, 35810190, 33789117, 35368039]: # nhrf examples
        save_xml(pmid)
