import numpy as np
import pandas as pd
import pvlib
import os
import nexrad
import functools


'''
Summarize vertical profiles across height dimension and aggregate to 
one row per scan
'''
def aggregate_profile_to_scan_level(infile, lat_lon):

    '''Produce scan-level summary for a vertical profile'''
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
    density = np.nansum(bin_width_km * linear_eta)
    density_unfiltered = np.nansum(bin_width_km * scan['linear_eta_unfiltered'])
    density_precip = density_unfiltered - density
    assert np.all(np.isnan(density_precip) | (density_precip >= 0))

    # Speed converted from m/s to km/h
    speed_km_h = scan['speed'] * (1/1000) * (3600/1)

    # Compute traffic rate in cm2 / km / h
    traffic_rate = np.nansum(bin_width_km * linear_eta * speed_km_h)
    traffic_rate_unfiltered = np.nansum(bin_width_km * scan['linear_eta_unfiltered'] * speed_km_h)
    traffic_rate_precip = traffic_rate_unfiltered - traffic_rate

    assert np.all(np.isnan(traffic_rate_precip) | (traffic_rate_precip >= 0))

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
           density,
           density_precip,
           traffic_rate,
           traffic_rate_precip,
           u,
           v,
           speed,
           direction,
           percent_rain]

    return row


'''
Load one station-year scan-level time series
'''
@functools.lru_cache(maxsize=32)
def load_station_year(root, station, year, resampled=False):
    '''Load scan-level data for given (station, year)'''
    
    if resampled:
        file = f'{root}/resampled/{station}_{year}.csv'
    else:
        file = f'{root}/scan_level/{station}-{year}.csv'

    print(f'Loading {file}')
    
    if os.path.exists(file):
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.set_index('date')
        return df
    
    else:
        return None


'''
Custom resampling method for pandas series
'''
def resample(data, index, limit=1, limit_direction='both'):
    '''Resample data frame with given index'''
    
    index_all = index.union(data.index)

    return data.reindex(index_all) \
               .interpolate(method='time', 
                            limit=limit,
                            limit_direction=limit_direction) \
               .reindex(index)


'''
Load and resample one station-year scan-level time series to a regular time interval
'''
#@functools.lru_cache(maxsize=32)
def load_and_resample_station_year(root, station, year,
                                   trim=True,
                                   freq='5Min',
                                   limit=12,
                                   limit_direction='both'):
    '''Load scan-level data for (station, year) and resample to fixed times'''

    data = load_station_year(root, station, year)
    
    if trim:
        # Trim to days that exist in data
        start = data.index.min().floor('D')
        end = data.index.max().ceil('D')
    else:
        # Use whole year
        start = pd.Timestamp(year=year, month=1, day=1)
        end =  pd.Timestamp(year=year+1, month=1, day=1)

    index = pd.date_range(start, end, freq=freq, tz='UTC')
    
    # First resample with limit=2 and record missing entries
    resampled_data = resample(data, 
                              index, 
                              limit=2, 
                              limit_direction=limit_direction)
    
    missing_before = np.isnan(resampled_data['lat'])
    
    # Now interpolate with the requested limit and record which values were filled
    resampled_data = resampled_data.interpolate(method='time', 
                                                limit=limit,
                                                limit_direction=limit_direction)
    
    missing_now = np.isnan(resampled_data['lat'])
    resampled_data['filled'] = (missing_before & ~missing_now).astype('int')

    resampled_data['station'] = station
    
    return resampled_data
    

'''
Aggregate resampled station-year time series to daily level
'''
def aggregate_single_station_year_to_daily(root, station, year, freq='5min'):

    time_step_in_hours = pd.Timedelta(freq) / pd.Timedelta('1h')
    
    integrate_fields = [
        # <input column name>, <output column name>
        ['density', 'density_hours'],
        ['density_precip', 'density_hours_precip'],
        ['traffic_rate', 'traffic'],
        ['traffic_rate_precip', 'traffic_precip']
    ]

    weighted_average_fields = [ 
        # <column name>, <weight column name>
        ['u', 'density'],
        ['v', 'density'],
        ['direction', 'density'],
        ['speed', 'density']
    ]

    average_fields = [ 
        # Row format: <column name>
        'percent_rain'
    ]

    copy_fields = [
        'date'
    ]

    last_df = None
    df = None
    next_df = None

    NEXRAD_LOCATIONS = nexrad.get_lat_lon()    
    lon = NEXRAD_LOCATIONS[station]['lon']
    lat = NEXRAD_LOCATIONS[station]['lat']

    # Load data frames as needed (methods are memoized to avoid duplicate work)        
    df = load_station_year(root, station, year, resampled=True)
    last_df = load_station_year(root, station, year-1, resampled=True)
    next_df = load_station_year(root, station, year+1, resampled=True)

    # Convenience variables with lists of all relevant dfs
    dfs = [last_df, df, next_df]

    # Initialize daily data        
    day_info = get_day_info(station, year)

    # Specify time periods of interest
    time_periods = [
        {
            'name': 'day',
            'start': 'sunrise', 
            'end': 'sunset'
        },
        {
            'name': 'night',
            'start': 'sunset',
            'end': 'next_sunrise'
        }
    ]

    write_dfs = []

    # Iterate over periods (night, day)
    for period in time_periods:

        print(f'Processing {period["name"]} data')

        write_df = pd.DataFrame(index=day_info.index)
        write_dfs.append(write_df)

        write_df['station'] = station
        write_df['lat'] = lat
        write_df['lon'] = lon
        
        for field in copy_fields:
            write_df[field] = day_info[field]

        write_df['period'] = period['name']

        # Iterate over days of year
        for day, row in day_info.iterrows():

            start = row[period['start']]
            end = row[period['end']]

            length = (end - start) / pd.Timedelta(1, 'h')
            write_df.loc[day, 'period_length'] = length

            rows = get_rows(dfs, start, end)

            # Skip if all missing (includes case when there are no rows at all)
            if rows['density'].isna().all():
                continue
            
            # Add diagnostics fields
            n_rows = len(rows)
            percent_missing = np.sum(np.isnan(rows['density'])) / n_rows
            percent_filled = np.sum(rows['filled']) / n_rows

            write_df.loc[day, 'percent_missing'] = percent_missing
            write_df.loc[day, 'percent_filled'] = percent_filled

            # Perform integration on specified fields
            for spec in integrate_fields:
                input_column, output_column = spec                    
                write_df.loc[day, output_column] = rows[input_column].sum() * time_step_in_hours

            # Perform weighted average on specified fields    
            for spec in weighted_average_fields:
                data_column, weight_column = spec
                weights = rows[weight_column]
                vals = rows[data_column]
                write_df.loc[day, data_column] = wtd_mean(weights, vals)

            # Perform simple average on specified fields
            for column in average_fields:
                write_df.loc[day, column] = rows[column].mean(skipna=True)

    out_df = pd.concat(write_dfs)
    return out_df


'''
Helper function for daily aggregation. Get a data frame describing each day of the year.
'''
def get_day_info(station, year):
    
    NEXRAD_LOCATIONS = nexrad.get_lat_lon()
    
    # location information
    lon = NEXRAD_LOCATIONS[station]['lon']
    lat = NEXRAD_LOCATIONS[station]['lat']
    loc = pvlib.location.Location(lat, lon)

    # Get series of days from Jan 1 this year to Jan 1 next year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year+1, month=1, day=1)
    days = pd.date_range(start, end, freq='D', tz='UTC')

    # Get sunrise, sunset, and transit times
    #   NOTE: methods other than 'spa' do not seem to correctly place
    #   sunset on the following date if it occurs after 00:00 UTC time
    day_info = loc.get_sun_rise_set_transit(days, method='spa')

    day_info['date'] = day_info.index.date
        
    # Add column for next sunrise, then drop last row, which corresponds to first day of following year
    day_info['next_sunrise'] = day_info['sunrise'].shift(-1)
    day_info = day_info.iloc[:-1]

    # Sanity checks
    assert(np.all(day_info['sunset'] > day_info['sunrise']))
    assert(np.all(day_info['next_sunrise'] > day_info['sunset']))

    # Compute lengths
    #day_info['day_length'] = (day_info['sunset'] - day_info['sunrise']) / pd.Timedelta(1, 'h')
    #day_info['night_length'] = (day_info['next_sunrise'] - day_info['sunset']) / pd.Timedelta(1, 'h')
    
    return day_info



'''
Utilities
'''

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


def get_rows(dfs, start, end):
    '''
    Get rows from df with index value between start and end from list
    of data frames.
    '''
    # If only one df is passed, turn it into a list
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        
    all_rows = [df[start:end] for df in dfs if df is not None]

    return pd.concat(all_rows)


def test_get_rows(root):
    station = 'KBOX'
    year = 2017
    
    last_df = load_station_year(root, station, year-1)
    df = load_station_year(root, station, year)
    next_df = load_station_year(root, station, year+1)
    
    # Get rows spanning last_df and df
    start = pd.Timestamp('2016-12-31 20:00:00Z')
    end = pd.Timestamp('2017-01-01 01:00:00Z')
    rows = get_rows([last_df, df, next_df], start, end)
    display(rows)
   
    # Get rows spanning df and next_df
    start = pd.Timestamp('2017-12-31 20:00:00Z')
    end = pd.Timestamp('2018-01-01 01:00:00Z')
    rows = get_rows([last_df, df, next_df], start, end)
    display(rows)
