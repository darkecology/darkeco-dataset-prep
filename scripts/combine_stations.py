import argparse
import glob
import os

import pandas as pd
from tqdm import tqdm

import util

def main():

    parser = argparse.ArgumentParser(
        description='Aggregate by station'
    )

    parser.add_argument(
        "--root", help="data root directory (default: ../data)", default="../data"
    )
    parser.add_argument("--years", nargs="+", type=int, help="years to process", default=None)
    parser.add_argument(
        "--min", help="do 5min data", dest="do_5min", action="store_true"
    )
    parser.add_argument(
        "--no-5min", help="don't do 5min data", dest="do_5min", action="store_false"
    )
    parser.add_argument(
        "--daily", help="do daily data", dest="do_daily", action="store_true"
    )
    parser.add_argument(
        "--no-daily",
        help="don't do daily data",
        dest="do_daily",
        action="store_false",
    )

    parser.set_defaults(do_5min=True, do_daily=True)
    args = parser.parse_args()

    root = args.root
    if not os.path.exists(root):
        raise FileNotFoundError(f"Path {root} does not exist")


    years = args.years or util.get_years(root)
    
    for year in years:
        print(f"***{year}***")
        if args.do_5min:
            print(" * 5-minute")
            files = glob.glob(f"{root}/5min/????-{year}-5min.csv")

            print("   - reading files")
            
            def read_files(files):
                for file in tqdm(files):
                    yield pd.read_csv(
                        file,
                        usecols=['station',
                                 'date',
                                 'density',
                                 'traffic_rate',
                                 'u',
                                 'v',
                                 'percent_rain']
                    )

            df = pd.concat(read_files(files))

            # df = df.pivot(
            #     index=["date"],
            #     columns="station",
            #     values=["density", "traffic_rate", "u", "v", "percent_rain"],
            # )

            print("   - writing output")
            outdir = f"{root}/allstations/5min"
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = f"{outdir}/{year}-5min.csv"
            df.to_csv(outfile)

        if args.do_daily:
            print(" * Daily")
            files = glob.glob(f"{root}/daily/????-{year}-daily.csv")

            print("   - reading files")

            def read_files(files):
                for file in tqdm(files):
                    yield pd.read_csv(
                        file,
                        usecols=['station',
                                 'date',
                                 'period',
                                 'period_length',
                                 'density_hours',
                                 'u',
                                 'v',
                                 'percent_missing',
                                 'percent_rain']
                    )

            df = pd.concat(read_files(files))

            # df = df[df["percent_missing"] <= 0.2]

            # df = df.pivot(
            #     index=["date", "period"],
            #     columns="station",
            #     values=["density_hours", "u", "v", "percent_rain"],
            # )

            print("   - writing output")
            outdir = f"{root}/allstations/daily"
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = f"{outdir}/{year}-daily.csv"
            df.to_csv(outfile)


if __name__ == "__main__":
    main()
