#!/bin/bash

# process hadron dat.gz (text) files into hdf5 files for the general hadron
# multiplicity and pi0 outputs

START=0
STOP=0

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

PLAYLIST=minerva1
MC=mc
INPUT_BASENAME=minosmatch_hadmult_${PLAYLIST}${MC}
HADMULT_OUTPUTNAME=minosmatch_hadmult_${PLAYLIST}${MC}
SINGLEPI0_OUTPUTNAME=minosmatch_singlepi0_${PLAYLIST}${MC}

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
cat << EOF
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${HADMULT_OUTPUTNAME}_${paddednum}.hdf5 \
    --skim had_mult
EOF
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${HADMULT_OUTPUTNAME}_${paddednum}.hdf5 \
    --skim had_mult
cat << EOF
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${SINGLEPI0_OUTPUTNAME}_${paddednum}.hdf5 \
    --skim single_pi0
EOF
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${SINGLEPI0_OUTPUTNAME}_${paddednum}.hdf5 \
    --skim single_pi0
done

# say we've finished
echo -e "\a"
