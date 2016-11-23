#!/bin/sh

# test using 1A (trained with 1A, so 1A test events are preserved and available)

MODEL="models/lminerva_spacetime_11_epsilon1479750354.npz"
DATAFILE="../HDF5files/minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc.hdf5"
LOGFILE=spacetime_epsilon_me1Bmc_1479750354_test.txt
PERFMAT=perfmat11_me1Bmc_epsilon1479750354.npy

TGTIDX=5
NOUTPUTS=11
# TGTIDX=4
# NOUTPUTS=67

# `-p` == make predictions (db file)
# `-t` == run test (no db file)
# `-v` == run in verbose mode
python Lasagne/minerva_tricolumnar_spacetime_epsilon.py \
  -d $DATAFILE \
  -t \
  -v \
  -s $MODEL \
  -f $LOGFILE \
  --target_idx $TGTIDX \
  --noutputs $NOUTPUTS

mv perfmat${NOUTPUTS}.npy $PERFMAT
