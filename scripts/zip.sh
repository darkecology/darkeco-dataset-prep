#!/bin/bash

root=${1:-/data/cajun_results/cajun-complete}
#dirs="scans 5min daily"
dirs="5min daily"

echo "root is $root"

for dir in $dirs; do
    tar cvf $root/$dir.tar.bz2 --use-compress-program='lbzip2 -n32' -C $root $dir
done
