#!/bin/bash

START=0
STOP=0

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

INBASE=ztest_times_minerva1mc
OUTBASE=ztest_times_minerva1mc

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
  python make_hdf5_fuelfiles.py \
    --skim time_dat \
    -b ${INBASE}_${paddednum} \
    -o ${OUTBASE}_${paddednum}.hdf5 \
    -x \
    --trim_column_down_x 94 \
    --trim_column_down_uv 47
done
