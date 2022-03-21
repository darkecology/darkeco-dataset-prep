#!/bin/bash

root=${1:-/data/cajun_results/cajun-complete}
VERSION=0.1.0
OUT=/scratch2/cajundata/$VERSION

years=$(eval echo {1995..2019})
echo $years

mkdir -p log

N=4 # number to execute in parallel
for year in $years; do
    echo "zipping profiles for $year"
    
    # File lists start with the year directory; this command prepends `profiles` to
    # the path so all profiles go to a common subdirectory
    tar cvf $OUT/profiles_$year.tar.bz2 --use-compress-program='lbzip2 -n32' -C $root/profiles --files-from $root/file_lists/year_lists/$year.txt --transform s"|^|profiles/|" > log/$year.out &

    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then wait -n; fi
done
wait
