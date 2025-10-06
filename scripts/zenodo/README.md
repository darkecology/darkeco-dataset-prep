# zenodo scripts

This directory contains a few scripts used for creating and updating zenodo records for the Dark Ecology dataset. The scripts are useful because: (1) there are seven repositories that should have nearly identical metadata, and (2) some of the files are very large, so programmatic upload from the servers where they are stored is much faster than copying to a personal computer and uploading them manually.

This is the rough workflow:

1. Create the zenodo records manually and record their IDs.
2. Fill in the detailed metadata for the first record using the web UI.
3. Download the metadata as json and edit it locally as needed, save in `metadata.json`.
4. Use `copy_metadata.sh` to copy the metadata to the other records. Edit the recrods manually as needed using the web UI.
5. Use `upload.sh` to upload the data files to the records.
5. The `nexrad-stations.csv` was uploaded manually

A zenodo access token is required. Generate it in the zenodo UI and run
```
export ZENODO_TOKEN=<your token>
```

To upload new data files, edit `upload.sh` appropriately and rerun it.

If new records are needed, create them manually and consider using `copy_metadata.sh` to populate some of the metadata.

The `zenodo_upload.sh` script comes from [zenodo-upload](https://github.com/jhpoelen/zenodo-upload)  by Jorrit Poelen (distributed under an MIT license). There are additional scripts there for programmatic access to zenodo.
