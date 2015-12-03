#!/bin/bash

filelist="skim_data_test_target0.dat
skim_data_valid_target0.dat
skim_data_learn_target0.dat"

MINERVA_RELEASE="v10r8p8"
REMOTE_DIR="/minerva/app/users/perdue/cmtuser/Minerva_${MINERVA_RELEASE}/Ana/NuclearTargetVertexing/ana/make_hist"

for file in $filelist
do
  scp perdue@minervagpvm02.fnal.gov:${REMOTE_DIR}/$file .
done
