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

fileroots="minosmatch_127x94_me1Bmc_micro_"
fileroots="minosmatch_127x94_me1Bmc_"
fileroots="minosmatch_nukecczdefs_127x94_me1Bmc_"
fileroots="minosmatch_nukecczdefs_fullz_127x94_me1Amc_"

MINERVA_RELEASE="v10r8p8"
REMOTE_DIR="/minerva/app/users/perdue/cmtuser/Minerva_${MINERVA_RELEASE}/Ana/NuclearTargetVertexing/ana/make_hist"
REMOTE_DIR="/minerva/data/users/perdue/mlmpr/raw_dat/nukeccskimmer_minosmatch_127x94_nukecczdefs"

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
