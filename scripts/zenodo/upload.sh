# 13345174 - 1995-1999
# 13345202 - 2000-2004
# 13345204 - 2005-2009
# 13345206 - 2010-2014
# 13345210 - 2015-2019
# 13345214 - 2020-2022
# 13345266 - time series

# years="1995 1996 1997 1998 1999"
# for year in $years; do
#     echo $year
#     ./zenodo_upload.sh 13345174 /scratch2/darkecodata/1.0.0/profiles_$year.tar.bz2
# done

# years="2000 2001 2002 2003 2004"
# years=""
# for year in $years; do
#     echo $year
#     ./zenodo_upload.sh 13345202 /scratch2/darkecodata/1.0.0/profiles_$year.tar.bz2
# done

# years="2005 2006 2007 2008 2009"
# years=""
# for year in $years; do
#     echo $year
#     ./zenodo_upload.sh 13345204 /scratch2/darkecodata/1.0.0/profiles_$year.tar.bz2
# done

# years="2010 2011 2012 2013 2014"
# years="2011"
# for year in $years; do
#     echo $year
#     ./zenodo_upload.sh 13345206 /scratch2/darkecodata/1.0.0/profiles_$year.tar.bz2
# done

# years="2015 2016 2017 2018 2019"
# years=""
# for year in $years; do
#     echo $year
#     ./zenodo_upload.sh 13345210 /scratch2/darkecodata/1.0.0/profiles_$year.tar.bz2
# done

# years="2020 2021 2022"
# years="2020"
# for year in $years; do
#     echo $year
#     ./zenodo_upload.sh 13345214 /scratch2/darkecodata/1.0.0/profiles_$year.tar.bz2
# done

files="5min.tar.bz2 daily.tar.bz2 scans.tar.bz2"
for file in $files; do
    echo $file
    ./zenodo_upload.sh 13345266 /scratch2/darkecodata/1.0.0/$file
done
