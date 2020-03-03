import pandas as pd
import numpy as np
import sys
import os
import glob


def main():

    if len(sys.argv) < 2:
        raise ValueError('Must supply root directory for data)')

    root = sys.argv[1]

    if not os.path.exists(root):
        raise FileNotFoundError(f'Path {root} does not exist')

    os.chdir(root)

    summary_folder = f'{root}/summary_folder'

    summary_files = glob.glob(f'{summary_folder}/*.csv')

    for file in summary_files:
        print(file)
        df = pd.read_csv(file)
        print(df)


if __name__ == '__main__':
    main()
