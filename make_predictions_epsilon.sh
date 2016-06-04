#!/bin/sh

python Lasagne/minerva_triamese_epsilon.py \
  -d minosmatch_nukecczdefs_127x50x25_xuv_me1Amc.hdf5 \
  -t \
  -p \
  -v \
  -a \
  -s models/lminervatriamese_epsilon1464985025.npz | tee evt_log_me1A_epsilon1464985025.txt
