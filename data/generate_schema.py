from genson import SchemaBuilder
import os
import json
import pprint
import re


def reduce_group(regex, s):
    x = 1
    while x:
        x = re.search(regex, s)
        if x:
            s = s.replace(x.group(), f'{{"{x.group(1)}":"{x.group(2)}"}}')
    return s

def remove_group(regex, s):
    x = 1
    while x:
        x = re.search(regex, s)
        if x:
            s = s.replace(x.group(), "")
    return s


def simplify_json(json_):
    s = json.dumps(json_) # the process (read file into string -> parse json -> dump to string) is needed to remove whitespace

    # from {"name":"n", "value":"v"} to {"n":"v"}
    regex = r'\{"name": "([^"]+)", "value": "([^"]+)"\}'
    s = reduce_group(regex, s)
    
    
    # from {"name":"n", "value":"v", "valqual":[...]} to {"n":"v"}
    # valqual is interesting later (for labels), but not for schema making.
    regex = r'\{"name": "([^"]+)", "value": "([^"]+)", "valqual": \[[^]]+\]\}'
    s = reduce_group(regex, s)
    
    # same but for "referece" instead of valqual
    regex = r'\{"name": "([^"]+)", "value": "([^"]+)", "reference": true\}'
    s = reduce_group(regex, s)
    
    # remove {"name":"..."} with "," before or after
    
    regex = r'\{"name": "([^"]+)"\},'
    s = remove_group(regex, s)
    regex = r', +\{"name": "([^"]+)"\}'
    s = remove_group(regex, s)

    if '"name"' in s:
        #print(s)
        m = re.findall(r"...name[^}]*\}",s)
        print(m)
        print("----failed")
        raise ValueError
    return json.loads(s)


builder = SchemaBuilder()
folder = "/mnt/data/upcast/data/arxpr/"
files = os.listdir(folder)
i = 0
for file in files[:1000]:
    try:
        with open(folder+file, "r") as f:
            json_file = json.load(f)
        json_file = simplify_json(json_file)
        builder.add_object(json_file)
        i += 1
        if i%20==0:
            print(i, "successes")
    except json.decoder.JSONDecodeError as e:
        print(file, e)
    except ValueError:
        pass
schema = builder.to_json(indent=2)
print(schema)
with open("schema.json", "w") as f:
    f.write(schema)

# then run
# datamodel-codegen --input schema.json --output-model-type pydantic_v2.BaseModel --output pydantic.py
