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
    -b minosmatch_hadmult_me1Amc_${paddednum} \
    -o minosmatch_hadmult_me1Amc_${paddednum}.hdf5 \
    --skim had_mult
  python make_hdf5_fuelfiles.py \
    -b minosmatch_hadmult_me1Amc_${paddednum} \
    -o minosmatch_singlepi0_me1Amc_${paddednum}.hdf5 \
    --skim single_pi0
done

# say we've finished
echo -e "\a"
