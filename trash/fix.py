import argparse
import os
import glob

from tqdm import tqdm

import util


def main():

    parser = argparse.ArgumentParser(
        description="Fix folder structure"
    )

    parser.add_argument(
        "--root", help="data root directory (default: ../data)", default="../data"
    )
    args = parser.parse_args()

    root = args.root
    years = util.get_years(root)
    
    for folder in [f"{root}/scans",
                   f"{root}/5min",
                   f"{root}/daily"]:

        for year in years:

            dst = f"{folder}/{year:4d}"
            if not os.path.exists(dst):
                os.makedirs(dst)

            filenames = glob.glob(f"{folder}/????-{year:4d}*.csv")

            for filename in filenames:
                basename = os.path.basename(filename)
                print(f"{filename} --> {dst}/{basename}")
                os.rename(filename, f"{dst}/{basename}")


if __name__ == "__main__":
    main()
