#!/bin/sh

ENS10_FOLDER="${ENS10_FOLDER:-`pwd`}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-`pwd`/efi}"

echo "Preprocessing data"
grib_copy -w,shortName=z,level=500,stepRange=48 $ENS10_FOLDER/output.pl.2018* $OUTPUT_FOLDER/output.z500_48h.grib
grib_copy -w,shortName=t,level=850,stepRange=48 $ENS10_FOLDER/output.pl.2018* $OUTPUT_FOLDER/output.t850_48h.grib
grib_copy -w,shortName=2t,stepRange=48 $ENS10_FOLDER/output.sfc.2018* $OUTPUT_FOLDER/output.t2m_48h.grib

echo "Generating CDFs for model climate"
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=z500 --gen_climate --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=z500 --gen_climate --year=2017
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t850 --gen_climate --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t850 --gen_climate --year=2017
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t2m --gen_climate --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t2m --gen_climate --year=2017

echo "Generating CDFs for ensembles"
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=z500 --gen_ensemble --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=z500 --gen_ensemble --year=2017
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t850 --gen_ensemble --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t850 --gen_ensemble --year=2017
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t2m --gen_ensemble --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t2m --gen_ensemble --year=2017

echo "Generating efi"
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=z500 --gen_efi --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=z500 --gen_efi --year=2017
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t850 --gen_efi --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t850 --gen_efi --year=2017
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t2m --gen_efi --year=2016
python calculate_efi.py --data_path=$OUTPUT_FOLDER --target_var=t2m --gen_efi --year=2017

echo "Merging efi data"
cdo mergetime $OUTPUT_FOLDER/efi_z500_201*.nc $OUTPUT_FOLDER/efi_z500.nc
cdo mergetime $OUTPUT_FOLDER/efi_t850_201*.nc $OUTPUT_FOLDER/efi_t850.nc
cdo mergetime $OUTPUT_FOLDER/efi_t2m_201*.nc $OUTPUT_FOLDER/efi_t2m.nc