# 13345174 - 1995-1999 (source)
# 13345202 - 2000-2004
# 13345204 - 2005-2009
# 13345206 - 2010-2014
# 13345210 - 2015-2019
# 13345214 - 2020-2022
# 13345266 - time series

TARGET_IDS="13345202 13345204 13345206 13345210 13345214 13345266"

for DEPOSITION_ID in $TARGET_IDS; do
    echo $DEPOSITION_ID
    curl -H 'Content-Type: application/json' \
    	 -X PUT \
    	 -d @metadata.json \
    	 "https://zenodo.org/api/deposit/depositions/$DEPOSITION_ID?access_token=$ZENODO_TOKEN"
done
