import pandas as pd
import numpy as np
import os
import glob
import csv
from collections import defaultdict

def main():    

    root = '/data/cajun_results/cajun-complete/'  # on doppler
    
    meta_file_folder = f'{root}/file_lists/'

    station_lists = defaultdict(list)
    year_lists = defaultdict(list)
    station_year_lists = defaultdict(list)
    for year in range(1995, 2020):
        for month in range(1,13):
            for day in range(1,32):

                datestr = f'{year}/{month:02d}/{day:02d}'
                datestr_folder = f'{root}/{datestr}/'
                if os.path.exists(datestr_folder):  # make sure this month/day exists in this year
                    print(datestr)
                    
                    for datestr_station_folder in glob.glob(datestr_folder+'*/'):
                        station = datestr_station_folder.split('/')[-2]
                        print(f'\t{station} ({datestr})')
                        
                        for scan_file in sorted(glob.glob(f'{datestr_station_folder}/*.csv')):
                            station_lists[station].append(scan_file)
                            year_lists[year].append(scan_file)
                            station_year_lists[f'{station}-{year}'].append(scan_file)

    save_lists(station_lists, 'station_lists', meta_file_folder)
    save_lists(year_lists, 'year_lists', meta_file_folder)
    save_lists(station_year_lists, 'station_year_lists', meta_file_folder)


def save_lists(lists, folder_name, meta_file_folder):

    folder_path = f'{meta_file_folder}/{folder_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for key, val in lists.items():
        with open(f'{folder_path}/{key}.txt', 'w') as outfile:
            outfile.write('\n'.join(sorted(val)))


if __name__ == '__main__':
    main()
