#!/bin/sh

DATE="1465065844"
MODEL=lminervatriamese_epsilon${DATE}.npz

python Lasagne/minerva_triamese_epsilon.py \
  -d minosmatch_nukecczdefs_127x50x25_xuv_me1Amc.hdf5 \
  -p \
  -a \
  -s models/${MODEL} | tee evt_log_me1A_epsilon${DATE}.txt
  # -t \
  # -v \
