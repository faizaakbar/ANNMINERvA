#!/bin/bash


START=0
STOP=0

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

DOENGY="no"
DOTIME="yes"

SAMPLE=me1Adata


# energy lattice images
INBASE=minosmatch_nukecczdefs_fullz_tproc_127x94_me1Adata
OUTBASE=minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xuv_${SAMPLE}

# time-lattice images
INBASET=minosmatch_nukecczdefs_fullz_tproc_127x94_me1Adata
OUTBASET=minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_txtutv_${SAMPLE}

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
# energy lattice
  if [ "$DOENGY" == "yes" ]; then
cat << EOF
  python make_hdf5_fuelfiles.py \
    -b ${INBASE}_${paddednum} \
    -o ${OUTBASE}_${paddednum}.hdf5 \
    -x \
    --trim_column_down_x 50 \
    --trim_column_down_uv 25 \
    --cap_planecode 66 \
    --min_keep_z -2e6
EOF
    python make_hdf5_fuelfiles.py \
      -b ${INBASE}_${paddednum} \
      -o ${OUTBASE}_${paddednum}.hdf5 \
      -x \
      --trim_column_down_x 50 \
      --trim_column_down_uv 25 \
      --cap_planecode 66 \
      --min_keep_z -2e6
fi
# time lattice
  if [ "$DOTIME" == "yes" ]; then
cat << EOF
  python make_hdf5_fuelfiles.py \
    --skim time_dat \
    -b ${INBASET}_${paddednum} \
    -o ${OUTBASET}_${paddednum}.hdf5 \
    -x \
    --trim_column_down_x 50 \
    --trim_column_down_uv 25 \
    --cap_planecode 66
EOF
    python make_hdf5_fuelfiles.py \
      --skim time_dat \
      -b ${INBASET}_${paddednum} \
      -o ${OUTBASET}_${paddednum}.hdf5 \
      -x \
      --trim_column_down_x 50 \
      --trim_column_down_uv 25 \
      --cap_planecode 66
fi
done
