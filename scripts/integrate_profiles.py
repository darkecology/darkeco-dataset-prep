import pandas as pd
import numpy as np
import sys
import os
import glob
from nexrad import get_lat_lon
import pvlib
import time as t


def main():
    years = [2016, 2017, 2018]

    if len(sys.argv) < 2:
        raise ValueError('Must supply root directory for data)')

    root = sys.argv[1]

    if not os.path.exists(root):
        raise FileNotFoundError(f'Path {root} does not exist')

    os.chdir(root)

    meta_file_folder = f'{root}/summary_folder'
    file_list_folder = f'{root}/file_lists/station_year_lists'

    if not os.path.exists(meta_file_folder):
        os.makedirs(meta_file_folder)

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
            outfile = f'{meta_file_folder}/{station_year}.csv'

            column_names = ['station',                    # 0
                            'lat',                        # 1
                            'lon',                        # 2
                            'date',                       # 3
                            'vir',                        # 4
                            'vir_unfiltered',             # 5
                            'traffic_rate',               # 6
                            'traffic_rate_unfiltered',    # 7
                            'u',                          # 8
                            'v',                          # 9
                            'speed',                      # 10
                            'direction',                  # 11
                            'percent_rain']               # 12

            # Get rows from individual files
            rows = [extract_scan_meta_data(f, lat_lon) for f in file_paths]

            # Create data frame
            df = pd.DataFrame(rows, columns=column_names)

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


def extract_scan_meta_data(infile, lat_lon):
    scan = pd.read_csv(infile)

    infile = infile.split('/')[-1]

    station = infile[:4]
    year = infile[4:8]
    month = infile[8:10]
    day = infile[10:12]
    hour = infile[13:15]
    minute = infile[15:17]
    second = infile[17:19]

    date = f'{year}-{month}-{day} {hour}:{minute}:{second}Z'

    bin_widths = np.diff(scan['bin_lower'])
    if np.all(bin_widths == bin_widths[0]):
        bin_width = bin_widths[0]  # bin width in meters
    else:
        raise ValueError('Bin vertical widths are not consistent!')

    bin_width_km = bin_width / 1000  # bin width in km

    # Vertically integrated reflectivity (vir)
    #
    #  -- units in each elevation bin are reflectivity (cm^2/km^3)
    #
    #  -- multiply by height of each bin in km to get cm^2/km^2 
    #
    #     == total scattering area in that elevation bin per 1 sq km. 
    #        of area in x-y plane
    #
    #  -- add values together to get total scattering area in a column
    #     above a 1km patch on the ground (cm^2/km^2)
    #
    #  -- (NOTE: can multiply these values by 10^-8 to get units
    #      km^2/km^2, i.e., a unitless quantity. Interpretation: total
    #      fraction of a 1 square km patch on the ground that would be
    #      filled by the total scattering area of targets in the column
    #      above it. I.e., zap all the birds so they fall to the
    #      ground, and measure how much ground space is filled up.)

    linear_eta = np.array(scan['linear_eta'])

    # Compute vertically integrated reflectivity in cm2 / km2
    vir = np.nansum(bin_width_km * linear_eta)
    vir_unfiltered = np.nansum(bin_width_km * scan['linear_eta_unfiltered'])

    # Speed converted from m/s to km/h
    speed_km_h = scan['speed'] * (1/1000) * (3600/1)

    # Compute traffic rate in cm2 / km / h
    traffic_rate = np.nansum(bin_width_km * linear_eta * speed_km_h)
    traffic_rate_unfiltered = np.nansum(bin_width_km * scan['linear_eta_unfiltered'] * speed_km_h)

    # Average velocity and speed weighted by reflectivity
    u = wtd_mean(linear_eta, scan['u'])
    v = wtd_mean(linear_eta, scan['v'])
    speed = wtd_mean(linear_eta, scan['speed'])

    # Average track as compass bearing (degrees clockwise from north)
    direction = pol2cmp(np.arctan2(v, u))

    # TODO: double check calculation
    percent_rain = (scan['percent_rain'] * scan['nbins']).sum() / scan['nbins'].sum()

    lat = lat_lon[station]["lat"]
    lon = lat_lon[station]["lon"]

    row = [station,
           lat,
           lon,
           date,
           vir,
           vir_unfiltered,
           traffic_rate,
           traffic_rate_unfiltered,
           u,
           v,
           speed,
           direction,
           percent_rain]

    return row


def wtd_mean(w, x):
    """Compute weighted mean of two pandas Series
    """
    valid = ~(np.isnan(w) | np.isnan(x))
    tot = np.sum(w[valid])
    if tot == 0:
        return np.nan
    else:
        return np.sum(w[valid] * x[valid]) / tot


def pol2cmp(theta):
    """Convert from mathematical angle to compass bearing

    Parameters
    ----------
    theta: array-like
        angle in radians counter-clockwise from positive x-axis

    Returns
    -------
    bearing: array-like
        angle in degrees clockwise from north

    See Also
    --------
    cmp2pol
    """
    bearing = np.rad2deg(np.pi / 2 - theta)
    bearing = np.mod(bearing, 360)
    return bearing


if __name__ == '__main__':
    main()
