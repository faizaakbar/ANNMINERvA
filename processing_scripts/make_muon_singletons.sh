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

SAMPLE=minerva1mc

INPUT_BASENAME=minosmatch_muondat_${SAMPLE}
OUTPUTNAME=minosmatch_muondat_${SAMPLE}

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${OUTPUTNAME}_${paddednum}.hdf5 \
    -s muon_dat
done
