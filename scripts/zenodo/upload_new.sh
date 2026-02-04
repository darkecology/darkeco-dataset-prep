#!/bin/bash

LOGDIR=upload_logs
mkdir -p $LOGDIR

#./zenodo_upload.sh 18436879 /scratch2/darkecodata/1.0.0/profiles_2019.tar.bz2 > $LOGDIR/profiles_2019.log 2>&1 &
#./zenodo_upload.sh 18436874 /scratch2/darkecodata/1.0.0/profiles_2023.tar.bz2 > $LOGDIR/profiles_2023.log 2>&1 &
#./zenodo_upload.sh 18436874 /scratch2/darkecodata/1.0.0/profiles_2024.tar.bz2 > $LOGDIR/profiles_2024.log 2>&1 &
#./zenodo_upload.sh 18436969 /scratch2/darkecodata/1.0.0/profiles_2025.tar.bz2 > $LOGDIR/profiles_2025.log 2>&1 &

#./zenodo_upload.sh 18433334 /scratch2/darkecodata/1.0.0/5min.tar.bz2 > $LOGDIR/5min.log 2>&1 &
./zenodo_upload.sh 18433334 /scratch2/darkecodata/1.0.0/scans.tar.bz2 > $LOGDIR/scans.log 2>&1 &

#wait
#echo "All uploads finished."
