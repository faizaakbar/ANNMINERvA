#!/bin/bash

for i in {0..25}
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
  inpbasename=minosmatch_skim_me1Amc_zsegments${paddednum}
  outpname=minosmatch_fuel_me1Amc_zseg${paddednum}.hdf5
  python make_hdf5_fuelfiles.py -b $inpbasename -o $outpname
done
