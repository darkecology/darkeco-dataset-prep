# Cajun Data Scripts

The scripts in this directory process cajun output and create derived
data products.

By default, they use this folder structure:

~~~ text
root/
|-- profiles/        <-- Cajun output root directory
|   |-- 1995
|   |-- ...
|   |-- 2021
|   |-- errors
|   |-- stats
|
|-- file_lists/      <-- Beginning of derived data products
|-- scan_level/   
|-- 5min/
|-- daily/
|-- allstations/
    |-- 5min/
    |-- daily/
~~~~

# Summary

Run this sequence of commands to go from raw profiles to the prepared data set:

~~~ text

#export ROOT=/data/cajun_results/cajun-complete
export ROOT=/scratch2/cajun_results/cajun_1_0
export VERSION=1.0.0

python compile_file_list.py --root $ROOT --start 1995 --end 2023
python summarize.py --root $ROOT # takes ~24 hours
python combine_stations.py  --root $ROOT
./zip.sh $ROOT $VERSION
./zip_profiles.sh $ROOT $VERSION

~~~

Here are some more options for partial runs:


~~~
# Summarize selected stations/years
python summarize.py --root $ROOT --stations KBOX KENX --years 2017 2018

# Selected summarize steps (scan, resample, daily)
python summarize.py --root $ROOT --actions scan      # ~2 hours/year
python summarize.py --root $ROOT --actions resample  # ~35 min/year
python summarize.py --root $ROOT --actions daily     # ~20 min/year
~~~

# Step 1: Compile file lists

The first step is to compile lists of files organized by station, by
year, and by station-year.

~~~ text
python compile_file_list.py --root $ROOT
~~~

This populates the `file_lists` directory:

~~~ text

file_lists/
|-- station_lists/
|   |-- KBOX.txt
|   |-- KENX.txt
|   
|-- station_year_lists/
|   |-- KBOX-2016.txt
|   |-- KBOX-2017.txt
|   |-- KENX-2016.txt
|   |-- KENX-2017.txt
|
|-- year_lists
    |-- 2016.txt
    |-- 2017.txt

~~~

Each text file lists the relative paths corresponding profile csv
files, e.g.:

~~~ text
2016/02/01/KENX/KENX20160201_000448.csv
2016/02/01/KENX/KENX20160201_001430.csv
2016/02/01/KENX/KENX20160201_002412.csv
2016/02/01/KENX/KENX20160201_002911.csv
2016/02/01/KENX/KENX20160201_003409.csv
2016/02/01/KENX/KENX20160201_003907.csv
2016/02/01/KENX/KENX20160201_004406.csv
2016/02/01/KENX/KENX20160201_004902.csv
2016/02/01/KENX/KENX20160201_005402.csv
2016/02/01/KENX/KENX20160201_005900.csv
...
~~~

# Step 2: Summarize Profiles

~~~
python summarize.py --root $ROOT --stations KBOX KENX --years 2018
~~~

# Step 3: Aggregate by Station


# Expanded Folder Stucture

~~~ text

root/
|-- profiles/
|   |-- 2016
|   |-- 2017
|   |-- errors
|   |-- stats
|
|-- file_lists/
|   |-- station_lists/
|   |-- station_year_lists/
|   |-- year_lists/
|
|-- scans/
|   |-- KBOX-2016.csv
|   |-- KBOX-2017.csv
|   |-- KENX-2016.csv
|   |-- KENX-2017.csv
|   
|-- 5min/
|   |-- KBOX-2016-5min.csv
|   |-- KBOX-2017-5min.csv
|   |-- KENX-2016-5min.csv
|   |-- KENX-2017-5min.csv
|   
|-- daily/
|   |-- KBOX-2016-daily.csv
|   |-- KBOX-2017-daily.csv
|   |-- KENX-2016-daily.csv
|   |-- KENX-2017-daily.csv
|
|-- combined-5min/
|   |-- 2016-5min.csv
|   |-- 2017-5min.csv
|
|-- combined-daily/
|   |-- 2016-daily.csv
|   |-- 2017-daily.csv
| 
|-- setup.py
|-- README

~~~
