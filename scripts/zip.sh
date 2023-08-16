#!/bin/bash

root=${1:-/scratch2/cajun_results/cajun_1_0}
VERSION=${2:-1.0.0}

OUT=/scratch2/cajundata/$VERSION

dirs="scans 5min daily combined-5min combined-daily"
echo "root is $root"

for dir in $dirs; do
    echo "zipping $dir"
    tar cvf $OUT/$dir.tar.bz2 --use-compress-program='lbzip2 -n32' -C $root $dir
done
