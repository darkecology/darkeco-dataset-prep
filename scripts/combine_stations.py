import argparse
import glob
import os

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import RLock, Pool

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
    
    params = [(i, root, year, args) for i, year in enumerate(years)]
    tqdm.set_lock(RLock())
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(do_one_year, params)
    

def do_one_year(params):

    i, root, year, args = params

    key_cols = ["date", "station"]
    data_cols = ["density", "traffic_rate", "u", "v", "percent_rain"]
    
    if args.do_5min:
        files = glob.glob(f"{root}/5min/{year}/????-{year}-5min.csv")
        if not files:
            warnings.warn("no files")
            return

        def read_files(files):

            for file in tqdm(
                    files,
                    desc=f"{year}, 5-minute",
                    lock_args=None,
                    position=i):
                
                yield pd.read_csv(
                    file,
                    usecols=key_cols+data_cols,
                )

        df = pd.concat(read_files(files))

        df.sort_values(key_cols, inplace=True)

        # df = df.pivot(
        #     index=["date"],
        #     columns="station",
        #     values=["density", "traffic_rate", "u", "v", "percent_rain"],
        # )

        outdir = f"{root}/combined-5min"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = f"{outdir}/{year}-5min.csv"
        df.to_csv(
            outfile,
            columns=key_cols+data_cols,
            index=False
        )

    if args.do_daily:
        files = glob.glob(f"{root}/daily/{year}/????-{year}-daily.csv")

        key_cols = ["date", "period", "station"]
        data_cols = ["period_length", "density_hours", "u", "v", "percent_missing", "percent_rain"]

        def read_files(files):
            for file in tqdm(
                    files,
                    desc=f"{year}, daily",
                    lock_args=None,
                    position=i):
                
                yield pd.read_csv(
                    file,
                    usecols=key_cols+data_cols
                )

        df = pd.concat(read_files(files))

        df.sort_values(key_cols, inplace=True)

        # df = df[df["percent_missing"] <= 0.2]

        # df = df.pivot(
        #     index=["date", "period"],
        #     columns="station",
        #     values=["density_hours", "u", "v", "percent_rain"],
        # )

        outdir = f"{root}/combined-daily"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = f"{outdir}/{year}-daily.csv"
        df.to_csv(
            outfile,
            columns=key_cols+data_cols,
            index=False,
        )


if __name__ == "__main__":
    main()
