import os
import json
import pprint
import re
from ast import literal_eval



class FieldCounter:
    def __init__(self):
        self.counts = {}
    def analyse(self, json_obj):
        self.current_json_paths = []
        self.analyse_obj(json_obj, path = [])

    def analyse_obj(self,obj, path):
        if type(obj) == list:
            for element in obj:
                self.analyse_obj(element, path=[*path, 0])
        elif type(obj) == dict:
            if "name" in obj:
                self.add_one(path, obj)
                #path = [*path, obj["name"].lower()] 
                #value = None
                #if "value" in obj:
                #    value = obj["value"]
                #self.add_one(path, value)
                return # ignore deeper stuff. The only interesting thing here is ontology, which should be included in labelmaking, but not schemamaking
            for key in obj:
                if key == "attributes" and "type" in obj:
                    self.analyse_obj(obj[key], path=[*path, "attribute:"+obj["type"].lower()])
                else:
                    self.analyse_obj(obj[key], path=[*path, key])
        elif type(obj) == str:
            pass
        elif type(obj) == int:
            pass
        else:
            print(obj)
            print(type(obj))
            raise ValueError

    def add_one_old(self, path, value):
        path = tuple(path)
        if path in self.counts:
            self.counts[path]["count"] += 1
        else:
            self.counts[path] = {"count": 1, "ex": []}
        if len(self.counts[path]["ex"]) <=5 and value and len(value) < 30 and not value in self.counts[path]["ex"]:
            self.counts[path]["ex"].append(value)

    def add_one(self, path, obj):
        key = tuple([*path, obj["name"].lower()])

        # add key
        if not key in self.counts:
            self.counts[key] = {
                    "count": 0,
                    "ont_count":0,
                    "ex": [],
                    "ont_ex":[],
                    "unique_json_count" : 0,
                    "ont_term_ex":[]}

        # count
        self.counts[key]["count"] += 1

        if not key in self.current_json_paths:
            self.counts[key]["unique_json_count"] += 1
            self.current_json_paths.append(key)

        # add example
        if "value" in obj:
            value = obj["value"]
            if len(self.counts[key]["ex"]) <=10 and len(value) < 30 and not value in self.counts[key]["ex"]:
                self.counts[key]["ex"].append(value)

        # count ontology
        if "valqual" in obj:
            for element in obj["valqual"]:
                if "name" in element and element["name"].lower() == "ontology":
                    self.counts[key]["ont_count"] += 1
                    if "value" in element:
                        value = element["value"]
                        if len(self.counts[key]["ont_ex"]) <=10 and len(value) < 30 and not value in self.counts[key]["ont_ex"]:
                            self.counts[key]["ont_ex"].append(value)
                if "name" in element and element["name"].lower() == "termid":
                    if "value" in element:
                        value = element["value"]
                        if len(self.counts[key]["ont_term_ex"]) <=10 and len(value) < 30 and not value in self.counts[key]["ont_term_ex"]:
                            self.counts[key]["ont_term_ex"].append(value)

        

        

folder = "/mnt/data/upcast/data/arxpr/"
files = os.listdir(folder)

def count_fields():
    """iterate through all jsons metadata files, and find all the fields, and count the number of appearances of each fields.
    Print each field with some info.
    """
    field_counter = FieldCounter()
    
    i = 0
    #files = files[:100]
    for file in files:
        try:
            with open(folder+file, "r") as f:
                json_obj = json.load(f)
            field_counter.analyse(json_obj)
            i += 1
            if i%1000==0:
                print(i, "successes")
        except json.decoder.JSONDecodeError as e:
            print(file, e)
    # sort by values
    sorted_counts = sorted(field_counter.counts.items(), key=lambda item: item[1]["count"])
    
    with open("result.txt", "w") as f:
        def printwrite(*args):
            print(*args)
            string = ""
            for arg in args:
                string += str(arg)+" "
            string += "\n"
            f.write(string)
        for line in sorted_counts:
            if line[1]["unique_json_count"] >= len(files)/10: 
                printwrite(line[1]["count"], "\t", line[1]["unique_json_count"], "\t", line[1]["ont_count"], "\t", line[0],"\n\t\t\t\t", line[1]["ex"])
                if line[1]["ont_count"]:
                    printwrite("\t\t\t\t", line[1]["ont_ex"])
                    printwrite("\t\t\t\t", line[1]["ont_term_ex"])
        printwrite("Total line count: ", len(sorted_counts))


def find_field(field, obj):

    #print(field)
    #pprint.pprint(obj)
    #print("")

    key = field[0]
    if key == 0:           
        yield from find_field(field[1:], obj)
        return
    if type(obj) == list:
        for element in obj:
            yield from find_field(field, element)
    elif key.startswith("attribute:"):
        if obj["type"] == key[10:] and "attributes" in obj:
            yield from find_field(field[1:], obj["attributes"])
    else:
        if len(field) == 1:
            if "name" in obj:
                if obj["name"] == key:
                    yield obj
            return
        if key in obj:
            yield from find_field(field[1:], obj[key]) # dict key
            return
        #print("not found", type(obj))



def simplify_jsons():
    """
    For each json metadata file, find all values connected to the desired fields (in "interesting_fields.txt"),
    then merge this into one large json: "arxpr_simplified.josn".
    """
    with open("interesting_fields.txt", "r") as f:
        text = f.read()
    text = text.lower() # put everythiong in lowercase
    fields = re.findall(r"[0-9]+ \t (\('.+\)) ", text)
    fields = [literal_eval(f) for f in fields] # tuple from string
    #for f in fields:
        #print(type(f))

    all_jsons_combined = {}
    i = 0
    for file in files:
        try:

            # load json
            with open(folder+file, "r") as f:
                text = f.read()
            text = text.lower() # put everythiong in lowercase
            json_obj = json.loads(text)

            # parse fields
            parsed = {}
            for j, field in enumerate(fields):
                field_key = field[-1]+"_"+str(j)
                field_key = field_key.replace(" ", "_")
                parsed[field_key] = []
                fields_found = find_field(field, json_obj)
                for ff in fields_found:

                    # parse into dict
                    try:
                        value = ff["value"]
                    except KeyError:
                        continue
                    value = value.replace("_", " ") # "_" is used instead of " ", by some but not all it seems - especially in experimental design, we mergge many values by this.
                    ff_dict = {"value" : value,
                               "ontology" : None}

                    # find any ontology info
                    if "valqual" in ff:
                        valqual = {}
                        for element in ff["valqual"]:
                            if "value" in element:
                                assert not element["name"] in valqual, ff
                                valqual[element["name"]] = element["value"]
                        if "ontology" in valqual and "termid" in valqual:
                            ff_dict["ontology"] = (valqual["ontology"], valqual["termid"])

                    parsed[field_key].append(ff_dict)

            # add to all_jsons dict
            all_jsons_combined[file[:-5]] = parsed


            i += 1
            if i%1000==0:
                print(i, "successes")

        except json.decoder.JSONDecodeError as e:
            pass

    with open(folder+"../arxpr_simplified.json", "w") as f:
        json.dump(all_jsons_combined, f, indent=4)

def make_arxpr_metadataset():
    import sys
    import os
    sys.path.append(os.getcwd() + '/..')
    from profiler.metadata_schemas import arxpr_schema as form
    out_file = folder+"../arxpr_metadataset_true_values.json"
    make_metadataset(form, out_file)

def make_arxpr2_metadataset():
    import sys
    import os
    sys.path.append(os.getcwd() + '/..')
    from profiler.metadata_schemas import arxpr2_schemas as form
    form = form["25"]
    out_file = folder+"../arxpr2_25_metadataset_true_values.json"
    make_metadataset(form, out_file)

def make_metadataset(form, out_file):
    """
    Load simplified json and a metadata schema, and stores all fields that are in the schema
    """
    fields = list(form.__fields__.keys())

    with open(folder+"../arxpr_simplified.json", "r") as f:
        simplified_metadata = json.load(f)
    metadataset = {}

    for key in simplified_metadata:
        pmid = key.split("___")[0]

        # add pmid key (NOTE: for papers referring to several datasets, all of the values are collected. This is easily seen by length of sample_count_13 or releasedate_12. NOTE length of these could be a feature as well?)
        if not pmid in metadataset:
            metadataset[pmid] = {field : [] for field in fields}

        for field in fields:
            if field in simplified_metadata[key]:
                metadataset[pmid][field].extend([element["value"] for element in simplified_metadata[key][field]])


        #pprint.pprint(metadataset)

    with open(out_file, "w") as f:
        json.dump(metadataset, f, indent=4)

def restrict_metadataset():
    """
    Load data from make_metadataset, and a metadata schema, and restricts the values to the "allowed" ones (in the literal in the schema)
    """
    # load desired fields
    import sys
    import os
    sys.path.append(os.getcwd() + '/..')
    from profiler.metadata_schemas import arxpr_schema as form
    #form = arxpr_schema.Metadata_form
    #fields = list(form.__fields__.keys())

    with open(folder+"../arxpr_metadataset_true_values.json", "r") as f:
        metadataset = json.load(f)
    refined_metadataset = {}

    # restrict to literal values - use "other" for other values
    for key in metadataset:
        for fieldname in ["study_type_18", "type_21", "sex_2"]:
            field_values = metadataset[key][fieldname]

            refined_values = []
            for value in field_values:
                if value in form.__fields__[fieldname].annotation.__args__:
                    refined_values.append(value)
                else:   
                    refined_values.append("other" if fieldname=="study_type_18" else form.__fields__[fieldname].annotation.__args__[-1])
            metadataset[key][fieldname] = refined_values


        field_values = metadataset[key]["releasedate_12"]
        metadataset[key]["releasedate_12"] = [int(date[:4]) for date in field_values]



        #pprint.pprint(metadataset)

    with open(folder+"../arxpr_metadataset_restricted_values.json", "w") as f:
        json.dump(metadataset, f, indent=4)

def restrict_arxpr2_metadataset():
    """
    similar to restrict_metadataset, but with arxpr2_schemas["25"] ( the version were only 25 most common values are included)
    some other differnces:
    - duplicate values are merged (i.e. if one paper has two labels but they are the same)
    - if there are multiple different avlues, all of them are removed
    - values not within these 25 are removed ( as opposed to marked "other")
    the values are still stored in list form for backwards compatability even though they are never multiple
    """
    # load desired fields
    import sys
    import os
    sys.path.append(os.getcwd() + '/..')
    from profiler.metadata_schemas import arxpr2_schemas as form
    form = form["25"]
    fields = list(form.__fields__.keys())

    with open(folder+"../arxpr2_25_metadataset_true_values.json", "r") as f:
        metadataset = json.load(f)
    refined_metadataset = {}

    for pmid in metadataset:
        for fieldname in fields:
            field_values = metadataset[pmid][fieldname]

            # remove duplicates
            field_values = list(set(field_values))

            # drop if multiple
            if len(field_values)>1:
                field_values = []

            # drop uncommon ones
            if len(field_values) and (not field_values[0] in form.__fields__[fieldname].annotation.__args__):
                field_values = []

            metadataset[pmid][fieldname] = field_values


    with open(folder+"../arxpr2_25_metadataset_restricted_values.json", "w") as f:
        json.dump(metadataset, f, indent=4)

if __name__ == "__main__":
    # count_fields()
    #simplify_jsons()
    # make_arxpr_metadataset()
    make_arxpr2_metadataset()
    restrict_arxpr2_metadataset()
