# This file handles fetching the arrrayexpress dataset.
# First, We query EuropePMC for all papers that are referred to by a dataset in arrayexpress 
#     (This happens in th EPCMIterator, by iterating through "pages" that only show some of the papers at a time)
#     This gives 23867 papers
# Then, in save_xml_and_metadata, for each paper, we try getting the xml (using xml_fethcer), and the arrayexpress data.
# This results in 14860 jsons.



import json
import requests
from pprint import pprint
import os

import xml_fetcher


class EPCMIterator:
    def __init__(self):
        self.ids = []
    def set_initial_url(self,page_size=25):
        query = "%28HAS_ARXPR%3Ay%29" # arrayexpress refer to these articles
        self.next_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&resultType=idlist&synonym=FALSE&cursorMark=*&pageSize={page_size}&format=json"
    def next(self,):
        page = get_(self.next_url)
        self.ids.extend([(result["id"], result["pmid"]) for result in page["resultList"]["result"]]) 
        try:
            self.next_url = page["nextPageUrl"]
        except KeyError:
            self.next_url = False

    def iterate(self, max_results = 100):
        while self.next_url:
            self.next()
            if len(self.ids) > max_results:
                break
            print(len(self.ids))


def save_xml_and_metadata(ids):
    for i, id_ in enumerate(ids):
        internal_id, pmid = id_
        if xml_fetcher.save_xml(pmid):
            try:
                db_ids = get_db_ids(internal_id)
                for db_id in db_ids:
                    save_metadata(pmid, db_id)
            except requests.exceptions.JSONDecodeError as e:
                print(e, pmid)
        if i%20 == 0:
            print(i, "/", len(ids))

def get_db_ids(internal_id):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{internal_id}/databaseLinks?database=ARXPR&format=json"
    page = get_(url)
    dbrefs = page["dbCrossReferenceList"]["dbCrossReference"]
    assert len(dbrefs)==1
    dbrefinfo = dbrefs[0]["dbCrossReferenceInfo"]
    db_ids = []
    for info in dbrefinfo:
        assert len(info) == 3
        assert info["info1"] == info["info2"]
        db_id = info["info1"]
        db_ids.append(db_id)
    return db_ids


def save_metadata(pmid, db_id, folder = "/mnt/data/upcast/data/arxpr"):
    filename = f"{pmid}___{db_id}.json"
    if filename in os.listdir(folder): # avoid requesting again
        return 1
    url = f"https://www.ebi.ac.uk/biostudies/files/{db_id}/{db_id}.json"
    try:
        text = requests.get(url).text
        if len(text) < 50:
            print(pmid, text)
            return 0
        with open(f"{folder}/{filename}", "w") as f:
            f.write(text)
        return 1
    except Exception as e:
        print(pmid, db_id)
        #print(e)
        raise(e)
        #return 0


def get_(url):
    response = requests.get(url)
    return response.json()

def main(page_size = 3, max_results = 7):
    it =  EPCMIterator()
    it.set_initial_url(page_size)
    it.iterate(max_results)
    
    save_xml_and_metadata(it.ids)

if __name__ == "__main__":
    main(1000, 25000)
