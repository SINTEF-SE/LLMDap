import json
import yaml

# Load JSON
with open("openapi.json", "r") as f:
    openapi_dict = json.load(f)

# Dump as YAML
with open("openapi.yaml", "w") as f:
    yaml.dump(openapi_dict, f, sort_keys=False)