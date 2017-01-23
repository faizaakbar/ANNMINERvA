#!/bin/sh

# test using 1B (trained with 1B, so 1B test events are preserved and available)

MODEL="models/lminerva_spacetime_67_epsilon1480703388.npz"
DATAFILE="../HDF5files/minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Bmc.hdf5"
LOGFILE=spacetime_epsilon_me1Bmc_1480703388_test.txt
PERFMAT=perfmat67_me1Bmc_epsilon1480703388.npy

# TGTIDX=5
# NOUTPUTS=11
TGTIDX=4
NOUTPUTS=67

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
