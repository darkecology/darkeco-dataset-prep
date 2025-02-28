# Dark Ecology Dataset Schemas

The source file for all schemas is the file `schema-source.yml`. 
After editing that file, run

``` .bash
python make_json_schemas.py     # writes schemas to json folder 
python validate.py              # validate sample data files against schemas
```
