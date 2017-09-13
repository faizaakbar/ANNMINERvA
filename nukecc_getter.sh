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

SAMPLE="me1Amc"

fileroots="minosmatch_nukecczdefs_fullz_tproc_127x94_${SAMPLE}_"
fileroots="minosmatch_nukecczdefs_fullzwitht_tproc_127x94_${SAMPLE}_"
REMOTE_DIR="/pnfs/minerva/persistent/users/perdue/mlmpr/201709/${SAMPLE}/rawtxt"

fileroots="minosmatch_kinematics_${SAMPLE}_"
REMOTE_DIR="/minerva/data/users/perdue/mlmpr/raw_dat/kine_skims"

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
