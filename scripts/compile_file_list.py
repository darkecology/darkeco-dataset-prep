import argparse
import glob
import os
import sys
from collections import defaultdict


def main():

    parser = argparse.ArgumentParser(
        description="Compile lists of cajun profiles organized by station, year, and (station, year)"
    )

    parser.add_argument(
        "--root", help="Output directory (default: ../data)", default="../data"
    )
    parser.add_argument(
        "--profile_dir",
        help="Profiles directory (default <root>/profiles)",
        default=None,
    )

    args = parser.parse_args()

    root = args.root
    if not os.path.exists(root):
        raise FileNotFoundError(f"Path {root} does not exist")

    meta_file_folder = f"{root}/file_lists"
    profile_dir = args.profile_dir or f"{root}/profiles"

    cwd = os.getcwd()
    os.chdir(profile_dir)

    station_lists = defaultdict(list)
    year_lists = defaultdict(list)
    station_year_lists = defaultdict(list)
    for year in range(1995, 2020):
        for month in range(1, 13):
            for day in range(1, 32):
                datestr_folder = f"{year}/{month:02d}/{day:02d}"
                if os.path.exists(
                    datestr_folder
                ):  # make sure this month/day exists in this year
                    print(datestr_folder)

                    for datestr_station_folder in glob.glob(f"{datestr_folder}/*"):
                        station = datestr_station_folder.split("/")[-1]
                        print(f"\t{station} ({datestr_folder})")

                        for scan_file in sorted(
                            glob.glob(f"{datestr_station_folder}/*.csv")
                        ):
                            # scan_file = os.path.relpath(scan_file, start=profile_dir)
                            station_lists[station].append(scan_file)
                            year_lists[year].append(scan_file)
                            station_year_lists[f"{station}-{year}"].append(scan_file)

    os.chdir(cwd)
    save_lists(station_lists, "station_lists", meta_file_folder)
    save_lists(year_lists, "year_lists", meta_file_folder)
    save_lists(station_year_lists, "station_year_lists", meta_file_folder)


def save_lists(lists, folder_name, meta_file_folder):

    folder_path = f"{meta_file_folder}/{folder_name}"

    print(folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for key, val in lists.items():
        with open(f"{folder_path}/{key}.txt", "w") as outfile:
            outfile.write("\n".join(sorted(val)))


if __name__ == "__main__":
    main()
