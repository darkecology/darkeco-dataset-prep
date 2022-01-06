import argparse
import glob
import os
import sys
import time as t
import warnings

import pandas as pd
import pvlib
from tqdm import tqdm

import util
from nexrad import get_lat_lon


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
    parser.add_argument("--scans_chunk_size", type=int, default=500)
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
    profile_dir, root, stations, years, max_scans, chunk_size=100
):

    if not os.path.exists(root):
        raise FileNotFoundError(f"Path {root} does not exist")

    file_list_folder = f"{root}/file_lists/station_year_lists"

    if not os.path.exists(f"{root}/scan_level"):
        os.makedirs(f"{root}/scan_level")

    lat_lon = get_lat_lon()

    # TODO: could potentially be sped up by loading many profiles at once and using pandas commands, including
    # TODO: group by to group by scan

    paths = glob.glob(f"{file_list_folder}/*.txt")
    filenames = [os.path.basename(p) for p in paths]
    all_stations = set([f[:4] for f in filenames])

    stations = stations or all_stations

    num_scans = 0
    for station in stations:
        for year in years:

            print(f"***Aggregate by scan***")

            station_year = f"{station}-{year}"
            path = f"{file_list_folder}/{station}-{year}.txt"
            if path not in paths:
                warnings.warn(f"{path} not found")
                continue

            with open(path, "r") as infile:
                profile_paths = infile.read().split("\n")

            # turn into absolute paths
            profile_paths = [
                f"{profile_dir}/{profile_path}" for profile_path in profile_paths
            ]

            start = t.time()

            # Get rows from individual files
            if max_scans and len(profile_paths) > max_scans - num_scans:
                profile_paths = profile_paths[: max_scans - num_scans]

            num_scans += len(profile_paths)

            def read_chunks():
                print(f"{station}-{year}, chunks of {chunk_size}")
                for pos in tqdm(range(0, len(profile_paths), chunk_size)):
                    yield util.aggregate_profiles_to_scan_level(
                        profile_paths[pos : pos + chunk_size], lat_lon
                    )

            df = pd.concat(read_chunks())

            # Add solar elevation (note: much faster to do in batch at end than row-by-row)
            solar_elev = pvlib.solarposition.spa_python(
                df["date"], df["lat"], df["lon"]
            )
            df["solar_elevation"] = solar_elev["elevation"].values

            # Convert lat/lon to strings to preserve full precision --- other floats will be truncated
            df["lat"] = df["lat"].apply(str)
            df["lon"] = df["lon"].apply(str)

            column_names = [
                "station",  # 0
                "lat",  # 1
                "lon",  # 2
                "date",  # 3
                "solar_elevation",  # 4
                "density",  # 5
                "density_precip",  # 6
                "traffic_rate",  # 7
                "traffic_rate_precip",  # 8
                "u",  # 9
                "v",  # 10
                "speed",  # 11
                "direction",  # 12
                "percent_rain",  # 13
                "rmse",
            ]  # 14

            outfile = f"{root}/scan_level/{station}-{year}.csv"
            df.to_csv(outfile, columns=column_names, index=False, float_format="%.4f")

            n_scans = len(df)

            elapsed = t.time() - start
            if n_scans > 0:
                print(
                    f"  time={elapsed:.2f}, scans={n_scans}, per scan={elapsed/n_scans:.4f}"
                )


def resample_station_years(root, stations, years, freq="5min"):

    print(f"***Resampling to {freq}***")

    outdir = f"{root}/{freq}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for station in stations:
        for year in years:

            print(f"{station}-{year}")
            resampled_df = util.load_and_resample_station_year(root, station, year)

            resampled_df.insert(3, "date", resampled_df.index)

            outfile = f"{outdir}/{station}-{year:4d}-{freq}.csv"

            resampled_df.to_csv(
                outfile,
                date_format="%Y-%m-%d %H:%M:%SZ",
                float_format="%.4f",
                index=False,
            )


def aggregate_station_years_to_daily(root, stations, years, freq="5min"):

    print(f"***Aggregating to daily***")

    daily_data_folder = f"{root}/daily"
    if not os.path.exists(daily_data_folder):
        os.makedirs(daily_data_folder)

    for station in stations:
        for year in years:
            print(f"{station}-{year}")
            df = util.aggregate_single_station_year_to_daily(
                root, station, year, freq=freq
            )
            outfile = f"{daily_data_folder}/{station}-{year}-daily.csv"
            df.to_csv(outfile, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
