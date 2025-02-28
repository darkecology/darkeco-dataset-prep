import os
import yaml
import json

yaml_file = 'schema-source.yml'

with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

for name in data['schemas']:
    with open(f'json/{name}.schema.json', 'w') as f:
        json.dump(data['schemas'][name], f, indent=4)
