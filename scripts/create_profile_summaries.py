import pandas as pd
import numpy as np
import sys
import os
import glob
import csv
from collections import defaultdict
from nexrad import get_lat_lon

def main(): 

    years = [2017, 2018]   

    #root = '/data/cajun_results/cajun-complete'  # on doppler
    if len(sys.argv) < 2:
        raise ValueError('Must supply root directory for data)');

    root = sys.argv[1]
    
    if not os.path.exists(root):
        raise FileNotFoundError(f'Path {root} does not exist');

    os.chdir(root)
    
    meta_file_folder = f'{root}/summary_folder'
    file_list_folder = f'{root}/file_lists/station_year_lists'

    if not os.path.exists(meta_file_folder):
        os.makedirs(meta_file_folder)

    lat_lon = get_lat_lon()

    for year in years:

        file_list_paths = glob.glob(f'{file_list_folder}/*.txt')  # why doesn't f'{file_list_folder}/*-{year}.txt' work?
        file_list_paths = [f for f in file_list_paths if f'{year}' in f]

        for file_list_path in file_list_paths:

            with open(file_list_path, 'r') as infile:
                file_paths = infile.read().split('\n')
            
            station_year = file_list_path.split('/')[-1].split('.')[0]

            print(station_year)
            with open(f'{meta_file_folder}/{station_year}.csv', 'w', newline='') as outfile:
          
                outfile.write('station,lat,lon,date,time,total_reflectivity_bio,total_reflectivity_unfiltered,avg_u,avg_v,avg_speed,avg_track,total_percent_rain\n')
        
                for file_path in file_paths:
                    outfile.write(extract_scan_meta_data(file_path, lat_lon) + '\n')


def extract_scan_meta_data(infile, lat_lon):

    scan = pd.read_csv(infile)

    infile = infile.split('/')[-1]
    
    station = infile[:4]
    year    = infile[4:8]
    month   = infile[8:10]
    day     = infile[10:12]
    hour    = infile[13:15]
    minute  = infile[15:17]
    second  = infile[17:19]

    date = f'{year}-{month}-{day}'
    time = f'{hour}:{minute}:{second}'
    
    bin_widths = np.diff(scan['bin_lower'])
    if np.all(bin_widths == bin_widths[0]):
        bin_width = bin_widths[0] # bin width in meters
    else:
        raise ValueError('Bin vertical widths are not consistent!')

    bin_width_km = bin_width / 1000 # bin width in km
    
    # Total reflectivity (vertically integrated density)
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
    total_reflectivity_bio = sum(bin_width_km * linear_eta)
    total_reflectivity_unfiltered = sum(bin_width_km * scan['linear_eta_unfiltered'])

    # Average velocity and speed weighted by reflectivity
    avg_u = wtd_mean(linear_eta, scan['u'])
    avg_v = wtd_mean(linear_eta, scan['v'])
    avg_speed = wtd_mean(linear_eta, scan['speed'])

    # Average track as compass bearing (degrees clockwise from north)
    avg_track = pol2cmp(np.arctan2(avg_v, avg_u))

    # TODO: double check calculation
    total_percent_rain = sum(scan['percent_rain'] * scan['nbins']) / sum(scan['nbins'])

    row = f'{station},{lat_lon[station]["lat"]},{lat_lon[station]["lon"]},{date},{time},{total_reflectivity_bio:.4f},{total_reflectivity_unfiltered:.4f},{avg_u:.4f},{avg_v:.4f},{avg_speed:.4f},{avg_track:.4f},{total_percent_rain}'

    return row

def wtd_mean( w, x ):
    return sum(w * x) / sum(w)


def pol2cmp( theta ):
    '''Convert from mathematical angle to compass bearing

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
    '''
    bearing = np.rad2deg(np.pi/2 - theta)
    bearing = np.mod(bearing, 360)
    return bearing

    
if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print(f'Total time: {time.time() - start}')
