import argparse
import os
import time as t
import warnings

import pandas as pd
import pvlib
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import util


def main():

    parser = argparse.ArgumentParser(
        description="Create summary data products from cajun profiles"
    )

    parser.add_argument(
        "--root", help="data root directory (default: ../data)", default="../data"
    )
    parser.add_argument(
        "--profile_dir",
        help="Profiles directory (default <root>/profiles)",
        default=None,
    )
    parser.add_argument("--stations", nargs="+", help="stations to process")
    parser.add_argument("--years", nargs="+", type=int, help="years to process")
    parser.add_argument("--max_scans", type=int, default=None)
    parser.add_argument("--scans_chunk_size", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=["scan", "resample", "daily", "all"],
        default="all",
    )

    args = parser.parse_args()
    actions = args.actions

    profile_dir = args.profile_dir or f"{args.root}/profiles"

    resample_frequency = "5min"

    if (
        "scan" in actions or "all" in actions
    ):  # True if actions is either 'all' or ['all']
        aggregate_station_years_by_scan(
            profile_dir,
            args.root,
            args.stations,
            args.years,
            args.max_scans,
            chunk_size=args.scans_chunk_size,
            max_workers=args.max_workers,
        )

    if "resample" in actions or "all" in actions:
        resample_station_years(
            args.root, args.stations, args.years, freq=resample_frequency
        )

    if "daily" in actions or "all" in actions:
        aggregate_station_years_to_daily(
            args.root, args.stations, args.years, freq=resample_frequency
        )


def aggregate_station_years_by_scan(
    profile_dir, root, stations, years, max_scans=None, chunk_size=100, max_workers=None
):

    if not os.path.exists(f"{root}/scans"):
        os.makedirs(f"{root}/scans")

    stations = stations or util.get_stations(root)
    years = years or util.get_years(root)

    print("***Aggregate by scan***")
    for year in years:

        outdir = f"{root}/scans/{year}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        for station in stations:

            print(f" - {station}-{year}")
            start = t.time()
            try:
                df, column_names = util.aggregate_single_station_year_by_scan(
                    profile_dir,
                    root,
                    station,
                    year,
                    max_scans=max_scans,
                    chunk_size=chunk_size,
                    max_workers=max_workers
                )

                outfile = f"{root}/scans/{year}/{station}-{year}.csv"
                df.to_csv(outfile, columns=column_names, index=False, float_format="%.4f")

                n_scans = len(df)
                elapsed = t.time() - start
                if n_scans > 0:
                    print(
                        f"  time={elapsed:.2f}, scans={n_scans}, per scan={elapsed/n_scans:.4f}"
                    )

            except ValueError as e:
                print(e)

                    

def resample_single_station_year(arg):

    root, outdir, station, year, freq = arg
    
    print(f"{station}-{year}")
    if not os.path.exists(f"{root}/scans/{year}/{station}-{year}.csv"):
        warnings.warn(
            f"{root}/scans/{year}/{station}-{year}.txt not found, must aggregate to scan level first"
        )
        return
        
    resampled_df, column_names = util.load_and_resample_station_year(root, station, year)
    resampled_df.insert(3, "date", resampled_df.index)

    outdir = f"{root}/{freq}/{year}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = f"{root}/{freq}/{year}/{station}-{year:4d}-{freq}.csv"

    resampled_df.to_csv(
        outfile,
        columns=column_names,
        date_format="%Y-%m-%d %H:%M:%SZ",
        float_format="%.4f",
        index=False,
    )


def resample_station_years(root, stations, years, freq="5min"):

    print(f"***Resampling to {freq}***")

    outdir = f"{root}/{freq}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    stations = stations or util.get_stations(root)
    years = years or util.get_years(root)

    for year in tqdm(years):
        args = [(root, outdir, station, year, freq) for station in stations]
        process_map(resample_single_station_year, args)

        

def aggregate_single_station_year_to_daily_helper(arg):
    
    root, station, year, freq = arg

    print(f"{station}-{year}")
    if not os.path.exists(f"{root}/{freq}/{year}/{station}-{year}-5min.csv"):
        warnings.warn(
            f"{root}/{freq}/{year}/{station}-{year}-{freq}.txt not found, must aggregate to scan level and resample first"
        )
        return
    
    df = util.aggregate_single_station_year_to_daily(
        root, station, year, freq=freq
    )

    outdir = f"{root}/daily/{year}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = f"{root}/daily/{year}/{station}-{year}-daily.csv"
    df.to_csv(outfile, index=False, float_format="%.6g")
    
    
def aggregate_station_years_to_daily(root, stations, years, freq="5min"):

    print("***Aggregating to daily***")

    if not os.path.exists(f"{root}/daily"):
        os.makedirs(f"{root}/daily")

    stations = stations or util.get_stations(root)
    years = years or util.get_years(root)

    for year in tqdm(years):
        args = [(root, station, year, freq) for station in stations]
        process_map(aggregate_single_station_year_to_daily_helper, args)


if __name__ == "__main__":
    main()
