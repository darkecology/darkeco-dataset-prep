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

# Summary of Commands to Runs

~~~ text
python compile_file_list.py --root /data/cajun_results/cajun-complete
python summarize.py --root /data/cajun_results/cajun-complete --stations KBOX KENX --years 2018
~~~

# Step 1: Compile file lists

The first step is to compile lists of files organized by station, by
year, and by station-year.

~~~ text
python compile_file_list.py --root /data/cajun_results/cajun-complete
~~~

This populates the `file_lists` directory, which looks like this:

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
python summarize.py --root /data/cajun_results/cajun-complete --stations KBOX KENX --years 2018
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
|-- scan_level/
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
|-- allstations/
|   |-- 5min/
|   |   |-- 2016-5min.csv
|   |   |-- 2017-5min.csv
|   |
|   |-- daily/
|       |-- 2016-daily.csv
|       |-- 2017-daily.csv
| 
|-- allstations-daily/
|   |-- 5min
|-- setup.py
|-- README

~~~