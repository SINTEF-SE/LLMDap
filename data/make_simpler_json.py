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
            #if line[1]["unique_json_count"] >= len(files)/10: 
            printwrite(line[1]["count"], "\t", line[1]["unique_json_count"], "\t", line[1]["ont_count"], "\t", line[0],"\n\t\t\t\t", line[1]["ex"])
                #if line[1]["ont_count"]:
                #    printwrite("\t\t\t\t", line[1]["ont_ex"])
                #    printwrite("\t\t\t\t", line[1]["ont_term_ex"])
        printwrite("Total line count: ", len(sorted_counts))
    print(len(sorted_counts))


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


if __name__ == "__main__":
    count_fields()
    #simplify_jsons()
