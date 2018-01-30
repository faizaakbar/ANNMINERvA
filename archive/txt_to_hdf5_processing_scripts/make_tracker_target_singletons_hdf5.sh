#!/bin/bash

# process image dat.gz (text) files into hdf5 files for the tracker and
# target regions.

START=0
STOP=0

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

INPUT_BASENAME=minosmatch_nukecczdefs_fullz_127x94_minerva1mc
TRACKER_OUTPUTNAME=minosmatch_nukecczdefs_tracker_127x72x36_minerva1mc
TARGET_OUTPUTNAME=minosmatch_nukecczdefs_genallz_pcodecap66_127x50x25_minerva1mc

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${TRACKER_OUTPUTNAME}_${paddednum}.hdf5 \
    -x \
    --trim_column_up_x 22 --trim_column_down_x 94 \
    --trim_column_up_uv 11 --trim_column_down_uv 47 \
    --min_keep_z 5810
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${TARGET_OUTPUTNAME}_${paddednum}.hdf5 \
    -x \
    --trim_column_down_x 50 \
    --trim_column_down_uv 25 \
    --cap_planecode 66
done
