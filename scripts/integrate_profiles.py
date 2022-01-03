import pandas as pd
import sys
import os
import glob
from nexrad import get_lat_lon
import pvlib
import time as t

from util import summarize_profile


def main():
    years = [2016, 2017, 2018]

    if len(sys.argv) < 2:
        raise ValueError('Must supply root directory for data)')

    root = sys.argv[1]

    if not os.path.exists(root):
        raise FileNotFoundError(f'Path {root} does not exist')

    os.chdir(root)

    scan_level_folder = f'{root}/scan_level'
    file_list_folder = f'{root}/file_lists/station_year_lists'

    if not os.path.exists(scan_level_folder):
        os.makedirs(scan_level_folder)

    lat_lon = get_lat_lon()

    # TODO: could potentially be sped up by loading many profiles at once and using pandas commands, including
    # TODO: group by to group by scan

    for year in years:

        file_list_paths = glob.glob(f'{file_list_folder}/*.txt')  # why doesn't f'{file_list_folder}/*-{year}.txt' work?
        file_list_paths = [f for f in file_list_paths if f'{year}' in f]

        for file_list_path in file_list_paths:

            with open(file_list_path, 'r') as infile:
                file_paths = infile.read().split('\n')

            station_year = file_list_path.split('/')[-1].split('.')[0]

            print(station_year)
            start = t.time()
            outfile = f'{scan_level_folder}/{station_year}.csv'

            column_names = ['station',                    # 0
                            'lat',                        # 1
                            'lon',                        # 2
                            'date',                       # 3
                            'density',                    # 4
                            'density_precip',             # 5
                            'traffic_rate',               # 6
                            'traffic_rate_precip',        # 7
                            'u',                          # 8
                            'v',                          # 9
                            'speed',                      # 10
                            'direction',                  # 11
                            'percent_rain']               # 12

            # Get rows from individual files
            rows = [summarize_profile(f, lat_lon) for f in file_paths]

            df = pd.DataFrame(rows, columns=column_names)

            do_solar_elevation = True
            if do_solar_elevation:
                # Add solar elevation (note: much faster to do in batch at end than row-by-row)
                solar_elev = pvlib.solarposition.spa_python(df['date'], df['lat'], df['lon'])
                df['solar_elevation'] = solar_elev['elevation'].values
                
                # Convert lat/lon to strings to preserve full precision --- other floats will be truncated
                df['lat'] = df['lat'].apply(str)
                df['lon'] = df['lon'].apply(str)
                
                # Add solar elevation to column names and write to file
                column_names.insert(4, 'solar_elevation')
                
            df.to_csv(outfile, columns=column_names, index=False, float_format='%.4f')

            n_scans = len(rows)

            elapsed = t.time()-start
            print(f'  time={elapsed:.2f}, scans={n_scans}, per scan={elapsed/n_scans:.4f}')


if __name__ == '__main__':
    main()
