#!/bin/bash

# process kine files into hdf5

START=0
STOP=0

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

SAMPLE=me1Bmc

INPUT_BASENAME=minosmatch_kinematics_${SAMPLE}
OUTPUTNAME=minosmatch_kinedat_${SAMPLE}

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${OUTPUTNAME}_${paddednum}.hdf5 \
    -s kine_dat
done
