#!/bin/bash

root=${1:-/scratch2/cajun_results/cajun_1_0}
VERSION=${2:-1.0.0}

OUT=/scratch2/darkecodata/$VERSION

dirs="scans 5min daily"
echo "root is $root"

for dir in $dirs; do
    echo "zipping $dir"
    # note: include meta folder in each .tar.bz2 file
    tar cvf $OUT/$dir.tar.bz2 --use-compress-program='lbzip2 -n32' -C $root $dir meta
done
