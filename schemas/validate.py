from frictionless import validate, Resource, system
import os

root = '../data'

pairs = [
    ("profiles/2019/01/01/KABR", "profile.schema.json"),
    ("scans/2022", "scan.schema.json"),
    ("5min/2022", "5min.schema.json"),
    ("daily", "daily.schema.json")
    ]


with system.use_context(trusted=True):
    for path, schema in pairs:
        path = f"{root}/{path}"
        files = os.listdir(path)
        for file in files[1:10]:
            if file.endswith(".csv"):
                print(f"{file} ({schema}): ", end="")
                report = validate(f"{path}/{file}", schema=f"json/{schema}")
                print(report.valid)
