# Schemas

The source file for all schemas is the file `schema-source.yml`. 
After editing that file, run

``` .bash
python make_json_schemas.py  # writes individual schemas in json output folder 
python validate.py # validates sample data files against schemas
```
