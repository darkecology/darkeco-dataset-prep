import numpy  as np
import pandas as pd

# timezones and solar elevation
import pytz
import astral

# file i/o
import os
import glob

# for plotting
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

vpr_root = '.' # root directory where vprs are stored

# station info (hardcoded for KBGM)
# (from https://weather.gladstonefamily.net/site/KBGM)
station         = 'KBGM'
station_lat     =  42.2068
station_lon     = -75.9799
# station_elev  =  499 # meters (ultimately not needed for solar elevation)

station_tz_name = "US/Eastern"
station_tz      = pytz.timezone(station_tz_name)

DEFAULT_START = pd.Timestamp(year=2019, month=1, day=1,
                             tz = station_tz)
DEFAULT_END   = pd.Timestamp(year=2019, month=1, day=10,
                             tz = station_tz)

# for our data, this is always 30
n_height_bins = 30

# plotting parameters
x_tick_freq = 15
y_tick_freq = 5

fig_format = '.pdf' # vect. text in pdf is much better, but the pixels of the heatmap tend to have a small white grid
                    # on my Mac, the only tool that doesn't get a grid is Adobe Acrobat Reader
# fig_format = '.png' # text in png is bad, but there's no grid between heatmap pixels

# fig height is fixed, but width is a linear function of scan count (fig_width_offset + n_scans * fig_width_per_scan)
fig_height         = 5 # inches
fig_width_offset   = 2 # inches
fig_width_per_scan = 0.05 # inches

outlier_pctile = 99.99 # the percentile value to define the max of the colorbar (0-100)
highlight_over = True # whether to highlight outlier pixels in the heatmap

daylight_scale = 50  # increase to shrink the daylight bar
cbar_scale     = 0.1 # increase to shrink the heatmap colorbar

daylight_cmap = 'bone'
heatmap_cmap  = 'pink'

# a divergent colormap normalizer for solar elevation centered at civil twilight (-6deg below horizon)
class CivilTwilightNorm(colors.Normalize):
    def __init__(self, vmin=-90, vmax=90):
        self.midpoint = -6
        colors.Normalize.__init__(self, vmin, vmax)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y)

def vpr_heatmap(start_date = DEFAULT_START, end_date = DEFAULT_END):
	global station

	# construct "simplified" start/end dates for printing in titles later
	# this assumes you're plotting some number of whole days, doesn't factor in H/M/S
	if start_date.year == end_date.year:
	    if start_date.month == end_date.month and start_date.day == end_date.day:
	        date_range = ""
	        plot_title_suffix = start_date.strftime("%b %-d, %Y")
	    elif start_date.month == end_date.month:
	        start_date_print = start_date.strftime("%b %-d")
	        end_date_print = end_date.strftime("%-d")
	        date_range = f"{start_date_print}-{end_date_print}"
	        
	        plot_title_suffix = start_date.strftime(", %Y")
	    else: # months differ, years same
	        start_date_print = start_date.strftime("%b %-d")
	        end_date_print = end_date.strftime("%b %-d")
	        date_range = f"{start_date_print} to {end_date_print}"

	        plot_title_suffix = start_date.strftime(", %Y")
	else: # years differ
	    start_date_print = start_date.strftime("%-d/%-m/%Y")
	    end_date_print = end_date.strftime("%-d/%-m/%Y")
	    date_range = f"{start_date_print} to {end_date_print}"
	    
	    plot_title_suffix = ""
	    
	plot_title = f"Vertical profiles of refl @{station}, {date_range}{plot_title_suffix}"
	# print(plot_title)

	start_date_filename = start_date.strftime("%Y_%m_%d")
	end_date_filename = end_date.strftime("%Y_%m_%d")
	plot_filename = f"vpr_heatmap_{station}_{start_date_filename}-{end_date_filename}"
	# print(plot_filename)

	print(f"Building VPR heatmap for {station}, {date_range}{plot_title_suffix}")

	#####################
	##### LOAD DATA #####

	# the VPRs are sorted by day and in UTC time, so construct a list of all UTC days between start and end
	daylist = pd.date_range(start_date.astimezone(pytz.utc), end_date.astimezone(pytz.utc), normalize=True, freq='D')

	# list all VPR files that could be loaded
	scan_files = []
	for day in daylist:
	    # print(f"{root}/{day.year}/{day.month:02d}/{day.day:02d}/{station}/*.csv")
	    scan_files = scan_files + glob.glob(f"{vpr_root}/{day.year}/{day.month:02d}/{day.day:02d}/{station}/*.csv")
	    
	scan_files = sorted(scan_files)

	# preconstruct the dataframe
	height_bins = np.arange(0,n_height_bins) * 100
	all_scans = pd.DataFrame(columns=height_bins)

	# read in all the scans
	for s, scan_file in enumerate(scan_files):
	    scan_key = os.path.splitext(os.path.basename(scan_file))[0]
	    
	    # read in the profile data
	    scan_data = pd.read_csv(scan_file,
	                            usecols = ['bin_lower', 'linear_eta'],
	                            index_col = 'bin_lower').T
	    
	    scan_data.rename(index={'linear_eta': scan_key}, inplace=True)
	    
	    all_scans = all_scans.append(scan_data)
	    
	all_scans.rename_axis(columns='', inplace=True)
	
	# print(all_scans)

	n_scans_raw = len(all_scans.index)

	#######################
	##### FILTER DATA #####

	# parse the scan_key into a timestamp object
	UTC_time_indices = pd.Series()
	local_time_indices = pd.Series()
	for scan_key, row in all_scans.iterrows():
	    # parse the scan key (assumes well-formed filenames!)
	    station =     scan_key[0:4]
	    year    = int(scan_key[4:8])
	    month   = int(scan_key[8:10])
	    day     = int(scan_key[10:12])
	    hour    = int(scan_key[13:15])
	    minute  = int(scan_key[15:17])
	    second  = int(scan_key[17:19])
	    
	    t_UTC = pd.Timestamp(year=year, month=month, day=day, 
	                         hour=hour, minute=minute, second=second,
	                         tz=pytz.utc)
	    t_local = t_UTC.astimezone(station_tz)
	    
	    UTC_time_indices  [scan_key] = t_UTC
	    local_time_indices[scan_key] = t_local	    
	    
	# set the timestamp as the index (and cache the scan_key as a column)
	all_scans.insert(0, 'scan_key', all_scans.index)
	all_scans['t_UTC'] = UTC_time_indices
	all_scans['t_local'] = local_time_indices
	all_scans.set_index('t_UTC', inplace=True)

	# print(all_scans)

	# trim scans in all_scans outside [start_date, end_date] (in local time)
	mask = (all_scans['t_local'] >= start_date) & (all_scans['t_local'] <= end_date)
	all_scans = all_scans.loc[mask]

	# recompute the number of scans remaining after trimming
	n_scans = len(all_scans.index)

	print(f"{n_scans}/{n_scans_raw} loaded scans remaining.")

	###################################
	##### COMPUTE SOLAR ELEVATION #####

	# for interpretability, we'll be adding a "daylight bar" to the heatmap, 
	# which is computed using solar elevation angle (deg above horizon)
	a = astral.Astral()

	solar_elevs = pd.Series()
	for scan_dt, row in all_scans.iterrows():
	    solar_elevs[scan_dt] = a.solar_elevation(scan_dt, station_lat, station_lon)

	all_scans.loc[:,'solar_elev'] = solar_elevs

	# initialize the colormap normalizer for solar elev data (centered at civil twilight)
	all_scans_norm = CivilTwilightNorm(np.min(all_scans['solar_elev']), np.max(all_scans['solar_elev']))

	#####################
	##### PLOT DATA #####

	# compute the ticks and tick labels
	xticks = np.arange(0, n_scans - 1, x_tick_freq)
	xticklabels = all_scans.iloc[xticks]['t_local']
	# print(xticklabels)

	yticks = np.arange(0, n_height_bins, y_tick_freq)
	yticklabels = yticks * 100
	# print(yticklabels)

	fig  = plt.figure(figsize=[fig_width_offset + n_scans * fig_width_per_scan, fig_height])

	# the daylight bar is overlaid just above the heatmap
	# this grid aligns the daylight bar and heatmap, with the heatmap colorbar on the right
	# (the daylight bar is only one row, and the colorbar only one col)
	nrow = daylight_scale
	ncol = int(n_scans * cbar_scale)

	grid = plt.GridSpec(nrow,ncol,hspace = 0)
	ax_daylight = plt.subplot(grid[0,0:ncol-1])
	ax_heatmap = plt.subplot(grid[1:nrow-1,0:ncol-1])
	ax_cbar = plt.subplot(grid[:,ncol-1])

	# draw the daylight bar
	ax_daylight.imshow(np.array(all_scans['solar_elev']).reshape(1,-1),
	                   norm=all_scans_norm,
	                   cmap=daylight_cmap)
	ax_daylight.axis('off')

	# build the heatmap colormap
	heatmap_cm = cm.get_cmap(heatmap_cmap)
	if highlight_over:
	    heatmap_cm.set_over("red") # highlight values above the colormap max

	# define outliers
	all_refl = all_scans[height_bins].values.flatten()
	vmax = np.percentile(all_refl, outlier_pctile)

	# draw the heatmap
	sns.heatmap(all_scans[height_bins].T,
	            cmap = heatmap_cm, vmin = 0, vmax = vmax,
	            ax = ax_heatmap, cbar_ax=ax_cbar,
	            xticklabels=xticklabels, yticklabels=yticklabels)

	# configure axes and title
	ax_heatmap.invert_yaxis()
	ax_heatmap.set_xticks(xticks);
	ax_heatmap.set_yticks(yticks);

	ax_heatmap.set_ylabel("height above ground (m)");
	ax_heatmap.set_xlabel(f"local time ({station_tz_name})");

	ax_cbar.set_ylabel("total reflectivity (dbZ)");
	fig.suptitle(plot_title);

	# write figure to disk
	fig.savefig(f"{plot_filename}{fig_format}", bbox_inches='tight');

if __name__ == '__main__':
	freq  = '1D' # freq to compute start_dates
	dur   = pd.DateOffset(days=2, hours=23, minutes=59, seconds=59) # duration of each heatmap

	if freq is None:
		vpr_heatmap()
	else:
		start_date_range = pd.date_range(start=DEFAULT_START, end=DEFAULT_END, freq=freq)
		for start_date in start_date_range:
			end_date = start_date + dur
			vpr_heatmap(start_date, end_date)