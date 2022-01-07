import argparse
import glob
import os

import pandas as pd
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser(
        description='Aggregate by station and "pivot" so columns correspond to stations'
    )

    parser.add_argument(
        "--root", help="data root directory (default: ../data)", default="../data"
    )
    parser.add_argument("--years", nargs="+", type=int, help="years to process")
    parser.add_argument(
        "--min", help="pivot 5min data", dest="do_5min", action="store_true"
    )
    parser.add_argument(
        "--no-5min", help="don't pivot 5min data", dest="do_5min", action="store_false"
    )
    parser.add_argument(
        "--daily", help="pivot daily data", dest="do_daily", action="store_true"
    )
    parser.add_argument(
        "--no-daily",
        help="don't pivot daily data",
        dest="do_daily",
        action="store_false",
    )

    parser.set_defaults(do_5min=True, do_daily=True)
    args = parser.parse_args()

    def read_files(files):
        for file in tqdm(files):
            yield pd.read_csv(file)

    root = args.root
    if not os.path.exists(root):
        raise FileNotFoundError(f"Path {root} does not exist")

    for year in args.years:
        print(f"***{year}***")
        if args.do_5min:
            print(" * 5-minute")
            files = glob.glob(f"{root}/5min/????-{year}-5min.csv")

            print("   - reading files")
            df = pd.concat(read_files(files))

            df = df.pivot(
                index=["date"],
                columns="station",
                values=["density", "traffic_rate", "u", "v", "percent_rain"],
            )

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
            df = pd.concat(read_files(files))

            df = df[df["percent_missing"] <= 0.2]

            df = df.pivot(
                index=["date", "period"],
                columns="station",
                values=["density_hours", "u", "v", "percent_rain"],
            )

            print("   - writing output")
            outdir = f"{root}/allstations/daily"
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = f"{outdir}/{year}-daily.csv"
            df.to_csv(outfile)


if __name__ == "__main__":
    main()
