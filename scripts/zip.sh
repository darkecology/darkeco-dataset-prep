#!/bin/bash

root=${1:-/data/cajun_results/cajun-complete}
VERSION=${2:-0.1.0}

OUT=/scratch2/cajundata/$VERSION

dirs="scans 5min daily combined-5min combined-daily"
echo "root is $root"

for dir in $dirs; do
    echo "zipping $dir"
    tar cvf $OUT/$dir.tar.bz2 --use-compress-program='lbzip2 -n32' -C $root $dir
done
