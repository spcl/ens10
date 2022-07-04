#!/bin/bash
## Preprocessing ERA5

ERA5_FOLDER="${ERA5_FOLDER:-`pwd`}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-`pwd`/meanstd}"

SELDATES=3jan,6jan,10jan,13jan,17jan,20jan,24jan,27jan,31jan,3feb,7feb,10feb,14feb,17feb,21feb,24feb,28feb,3mar,7mar,10mar,14mar,17mar,21mar,24mar,28mar,31mar,4apr,7apr,11apr,14apr,18apr,21apr,25apr,28apr,2may,5may,9may,12may,16may,19may,23may,26may,30may,2jun,6jun,9jun,13jun,16jun,20jun,23jun,27jun,30jun,4jul,7jul,11jul,14jul,18jul,21jul,25jul,28jul,1aug,4aug,8aug,11aug,15aug,18aug,22aug,25aug,29aug,1sep,5sep,8sep,12sep,15sep,19sep,22sep,26sep,29sep,3oct,6oct,10oct,13oct,17oct,20oct,24oct,27oct,31oct,3nov,7nov,10nov,14nov,17nov,21nov,24nov,28nov,1dec,5dec,8dec,12dec,15dec,19dec,22dec,26dec,29dec,2jan
cdo -P 16 -f nc4 -t ecmwf seldate,1998-01-03,2017-12-31 -select,name=T2M,dom=$SELDATES $ERA5_FOLDER/analysis_sfc* "$OUTPUT_FOLDER/ERA5_sfc_t2m.nc"
cdo -P 16 -f nc4 -t ecmwf seldate,1998-01-03,2017-12-31 -select,name=Z,level=50000,dom=$SELDATES $ERA5_FOLDER/analysis_pl* "$OUTPUT_FOLDER/ERA5_z500.nc"
cdo -P 16 -f nc4 -t ecmwf seldate,1998-01-03,2017-12-31 -select,name=T,level=85000,dom=$SELDATES $ERA5_FOLDER/analysis_pl* "$OUTPUT_FOLDER/ERA5_t850.nc"

ls $OUTPUT_FOLDER/ERA5*.nc | parallel --progress -j 3 cdo -f nc4 merge -setname,scale_mean -timmean {} -setname,scale_std -timstd {} {.}_scale.nc 
