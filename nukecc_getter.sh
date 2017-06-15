#!/bin/bash

START=0
STOP=0

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

echo "Grabbing runs $START to $STOP..."

fileroots="minosmatch_kinematics_me1Bmc_"
fileroots="minosmatch_nukecczdefs_fullz_tproc_127x94_me1Adata_"

REMOTE_DIR="/minerva/data/users/perdue/mlmpr/raw_dat/kine_skims"
REMOTE_DIR="/minerva/data/users/perdue/mlmpr/raw_dat/nukeccskimmer_minosmatch_127x94_nukecczdefs/with_t_processing/me1Adata"

for file in $fileroots
do
  for i in `seq ${START} 1 ${STOP}`
  do
    filenum=`echo $i | perl -ne 'printf "%04d",$_;'`
    filename=${file}${filenum}.dat.gz
    echo $filename
    scp perdue@minervagpvm02.fnal.gov:${REMOTE_DIR}/$filename .
  done
done



# say we've finished
echo -e "\a"
