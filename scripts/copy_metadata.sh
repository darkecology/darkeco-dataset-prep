#!/bin/bash

root=${1:-/scratch2/cajun_results/cajun_1_0}
cajun=${2:-~/cajun}
VERSION=${2:-1.0.0}

OUT=/scratch2/darkecodata/$VERSION

stationcsv="$cajun/python/nexrad/nexrad-stations.csv"

# Check if the file exists
if [ -f $stationcsv ]; then
    cp "$cajun/python/nexrad/nexrad-stations.csv" "$OUT"
else
    echo "Error: File '$stationcsv' does not exist."
    exit 1
fi
