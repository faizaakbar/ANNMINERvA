#!/bin/bash

filelist="skim_data_test_target0.dat
skim_data_valid_target0.dat
skim_data_learn_target0.dat"

REMOTE_DIR="/minerva/app/users/perdue/cmtuser/Minerva_v10r8p7/Ana/NuclearTargetVertexing/ana/make_hist"

for file in $filelist
do
  scp perdue@minervagpvm02.fnal.gov:${REMOTE_DIR}/$file .
done
