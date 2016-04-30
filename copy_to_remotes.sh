#!/bin/bash

filelist="
minosmatch_hadmult_127x50x25_xuv_me1Bmc.hdf5
minosmatch_hadmult_127x50x25_xuv_me1Bmc_README.txt"

REMOTE1="tev.fnal.gov:/phihome/perdue/theano/data/"
REMOTE2="minervagpvm02.fnal.gov:/minerva/data/users/perdue/mlmpr/nukecc"
for file in $filelist
do
  scp $file perdue@${REMOTE1}
  scp $file perdue@${REMOTE2}
done

mkdir temp_for_copy
for file in $filelist
do
  cp $file temp_for_copy
done
tar -cvzf temp_for_copy.tgz temp_for_copy
rm -rf temp_for_copy
echo "Next, run:"
echo " scp temp_for_copy.tgz gnperdue@titan.ccs.ornl.gov:/ccs/proj/hep105/data/theano"
