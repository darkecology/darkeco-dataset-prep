import argparse
import pandas as pd
import sys
import os
import glob
from nexrad import get_lat_lon
import pvlib
import time as t
import warnings

import util

def main():

    parser = argparse.ArgumentParser(
        description='Create summary data products from cajun profiles'
    )    

    parser.add_argument('--root', help='data root directory (default: ../data)', default='../data')
    parser.add_argument('--stations',  nargs="+", help='stations to process')
    parser.add_argument('--years', nargs="+", type=int, help="years to process")
    parser.add_argument('--max_scans', type=int, default=None)    
    parser.add_argument('--actions', nargs="+", choices=['scan', 'resample', 'daily', 'all'], default='all')
    
    args = parser.parse_args()
    actions = args.actions

    resample_frequency = '5min'
    
    if 'scan' in actions or 'all' in actions: # True if actions is either 'all' or ['all']
        aggregate_station_years_by_scan(args.root, args.stations, args.years, args.max_scans)

    if 'resample' in actions or 'all' in actions:
        resample_station_years(args.root, args.stations, args.years, freq=resample_frequency)

    if 'daily' in actions or 'all' in actions:
        aggregate_station_years_to_daily(args.root, args.stations, args.years, freq=resample_frequency)


def aggregate_station_years_by_scan(root, stations, years, max_scans):

    if not os.path.exists(root):
        raise FileNotFoundError(f'Path {root} does not exist')

    os.chdir(root)

    file_list_folder = f'{root}/file_lists/station_year_lists'

    if not os.path.exists(f"{root}/scan_level"):
        os.makedirs(f"{root}/scan_level")

    lat_lon = get_lat_lon()

    # TODO: could potentially be sped up by loading many profiles at once and using pandas commands, including
    # TODO: group by to group by scan

    paths = glob.glob(f'{file_list_folder}/*.txt')
    filenames = [os.path.basename(p) for p in paths]
    all_stations = set([f[:4] for f in filenames])

    stations = stations or all_stations

    num_scans = 0    
    for station in stations:
        for year in years:

            station_year = f"{station}-{year}"
            print(station_year)
            
            path = f"{file_list_folder}/{station}-{year}.txt"            
            if path not in paths:
                warnings.warn(f"{path} not found")
                continue
                        
            with open(path, 'r') as infile:
                profile_paths = infile.read().split('\n')

            start = t.time()
            outfile = f'{root}/scan_level/{station_year}.csv'

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
            rows = []
            for f in profile_paths:
                if max_scans and num_scans >= max_scans:
                    break
                rows.append(util.aggregate_profile_to_scan_level(f, lat_lon))
                num_scans += 1

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
            if n_scans > 0:
                print(f'  time={elapsed:.2f}, scans={n_scans}, per scan={elapsed/n_scans:.4f}')


def resample_station_years(root, stations, years, freq="5min"):

    for station in stations:
        for year in years:
            resampled_df = util.load_and_resample_station_year(root, station, year)

            resampled_df.insert(3, 'date', resampled_df.index)
            
            outfile = f'{root}/resampled/{station}_{year:4d}.csv'

            resampled_df.to_csv(outfile, 
                                date_format='%Y-%m-%d %H:%M:%SZ',
                                float_format='%.4f',
                                index=False)


def aggregate_station_years_to_daily(root, stations, years, freq='5min'):

    daily_data_folder = f'{root}/daily'
    if not os.path.exists(daily_data_folder):
        os.makedirs(daily_data_folder)

    for station in stations:
        for year in years:
            df = util.aggregate_single_station_year_to_daily(root, station, year, freq=freq)
            outfile = f'{daily_data_folder}/{station}_{year}.csv'
            df.to_csv(outfile, index=False, float_format='%.6g')
    
                
if __name__ == "__main__":
    main()
