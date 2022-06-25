#!/bin/bash
## Preprocessing ENS10

ENS10_FOLDER="${ENS10_FOLDER:-`pwd`}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-`pwd`/meanstd}"

mkdir -p $OUTPUT_FOLDER/ensemble

# Extracting 500hPa, 850hPa and sfc at 48h leading time
echo "Extracting variables at 500hPa and 850hPa at 48h leading time"
ls $ENS10_FOLDER/output.pl* | parallel --progress -j 10 grib_copy -w,level=500/850,stepRange=48 {} $OUTPUT_FOLDER/ensemble/output.pl.[dataDate].grib
echo "Extracting variables at surface level at 48h leading time"
ls $ENS10_FOLDER/output.sfc* | parallel --progress -j 10 grib_copy -w,stepRange=48 {} $OUTPUT_FOLDER/ensemble/output.sfc.[dataDate].grib

# Calculating mean and standard deviation of ensembles
echo "Calculating mean and standard deviation for variables at 500hPa and 850hPa"
cdo -P 16 -f nc4 -z zip6 -t ecmwf seldate,1998-01-03,2017-12-31 -daymean -select,level=50000,85000 "$OUTPUT_FOLDER/ensemble/output.pl.*.grib" "$OUTPUT_FOLDER/ENS10_pl_mean.nc"
cdo -P 16 -f nc4 -z zip6 -t ecmwf seldate,1998-01-03,2017-12-31 -daystd -select,level=50000,85000 "$OUTPUT_FOLDER/ensemble/output.pl.*.grib" "$OUTPUT_FOLDER/ENS10_pl_std.nc"
echo "Calculating mean and standard deviation for variables at surface level"
cdo -P 16 -f nc4 -z zip6 -t ecmwf seldate,1998-01-03,2017-12-31 -daymean -copy "$OUTPUT_FOLDER/ensemble/output.sfc.*.grib" "$OUTPUT_FOLDER/ENS10_sfc_mean.nc"
cdo -P 16 -f nc4 -z zip6 -t ecmwf seldate,1998-01-03,2017-12-31 -daystd -copy "$OUTPUT_FOLDER/ensemble/output.sfc.*.grib" "$OUTPUT_FOLDER/ENS10_sfc_std.nc"

echo "Normalizing data"
ls $OUTPUT_FOLDER/ENS10*.nc | parallel --progress -j 4 cdo -f nc4 div -sub [ {} -timmean {} ] -addc,0.1 -timstd {} {.}_normalized.nc 
