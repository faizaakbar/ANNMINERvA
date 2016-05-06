#!/bin/bash

START=0
STOP=0

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
  python make_hdf5_fuelfiles.py \
    -b minosmatch_nukecczdefs_fullz_127x94_me1Amc_${paddednum} \
    -o minosmatch_nukecczdefs_tracker_127x72x36_me1Amc_${paddednum}.hdf5 \
    -x \
    --trim_column_up_x 22 --trim_column_down_x 94 \
    --trim_column_up_uv 11 --trim_column_down_uv 47 \
    --min_keep_z 5810
done
