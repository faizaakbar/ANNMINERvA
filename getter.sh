#!/bin/bash

fileroots="skim_data_test_zsegments
skim_data_valid_zsegments
skim_data_learn_zsegments"

MINERVA_RELEASE="v10r8p8"
REMOTE_DIR="/minerva/app/users/perdue/cmtuser/Minerva_${MINERVA_RELEASE}/Ana/NuclearTargetVertexing/ana/make_hist"

for file in $fileroots
do
  for i in {2..3}
  do
    filenum=`echo $i | perl -ne 'printf "%04d",$_;'`
    filename=${file}${filenum}.dat
    scp perdue@minervagpvm02.fnal.gov:${REMOTE_DIR}/$filename .
  done
done
