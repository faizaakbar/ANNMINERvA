#!/bin/bash

START=0
STOP=7

if [[ $# == 1 ]]; then
  STOP=$1
elif [[ $# > 1 ]]; then
  START=$1
  STOP=$2
fi

echo "Grabbing runs $START to $STOP..."

fileroots="skim_data_test_zsegments
skim_data_valid_zsegments
skim_data_learn_zsegments"

MINERVA_RELEASE="v10r8p8"
REMOTE_DIR="/minerva/app/users/perdue/cmtuser/Minerva_${MINERVA_RELEASE}/Ana/NuclearTargetVertexing/ana/make_hist"

for file in $fileroots
do
  for i in `seq ${START} 1 ${STOP}`
  do
    filenum=`echo $i | perl -ne 'printf "%04d",$_;'`
    filename=${file}${filenum}.dat
    echo $filename
    scp perdue@minervagpvm02.fnal.gov:${REMOTE_DIR}/$filename .
  done
done

# say we've finished
echo -e "\a"
