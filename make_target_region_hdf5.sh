#!/bin/bash


START=0
STOP=0

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

INBASE=minosmatch_nukecczdefs_fullz_tproc_127x94_me1Bmc
OUTBASE=minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xuv_me1Bmc

# time-lattice images
INBASET=minosmatch_nukecczdefs_fullzwitht_127x94_me1Bmc
OUTBASET=minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xuv_me1Bmc

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
# energy lattice
cat << EOF
  python make_hdf5_fuelfiles.py \
    -b ${INBASE}_${paddednum} \
    -o ${OUTBASE}_${paddednum}.hdf5 \
    -x \
    --trim_column_down_x 50 \
    --trim_column_down_uv 25 \
    --cap_planecode 66
EOF
  python make_hdf5_fuelfiles.py \
    -b ${INBASE}_${paddednum} \
    -o ${OUTBASE}_${paddednum}.hdf5 \
    -x \
    --trim_column_down_x 50 \
    --trim_column_down_uv 25 \
    --cap_planecode 66
# time lattice
# cat << EOF
#   python make_hdf5_fuelfiles.py \
#     --skim time_dat \
#     -b ${INBASET}_${paddednum} \
#     -o ${OUTBASET}_${paddednum}.hdf5 \
#     -x \
#     --trim_column_down_x 50 \
#     --trim_column_down_uv 25 \
#     --cap_planecode 66
# EOF
#   python make_hdf5_fuelfiles.py \
#     --skim time_dat \
#     -b ${INBASET}_${paddednum} \
#     -o ${OUTBASET}_${paddednum}.hdf5 \
#     -x \
#     --trim_column_down_x 50 \
#     --trim_column_down_uv 25 \
#     --cap_planecode 66
done
