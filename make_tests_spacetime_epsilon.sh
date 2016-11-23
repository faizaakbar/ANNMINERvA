#!/bin/sh

MODEL="models/lminerva_spacetime_11_epsilon1479750354.npz"
DATAFILE="../HDF5files/minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Bmc_000.hdf5"
LOGFILE=spacetime_epsilon_me1Bmc_1479750354_test.txt
PERFMAT=perfmat11_me1Bmc_epsilon1479750354.npy

TGTIDX=5
NOUTPUTS=11
# TGTIDX=4
# NOUTPUTS=67

# `-p` == make predictions (db file)
# `-t` == run test (no db file)
# `-v` == run in verbose mode
# `-a` == use the whole file as if it were the "test" sample
python Lasagne/minerva_tricolumnar_spacetime_epsilon.py \
  -d $DATAFILE \
  -t \
  -v \
  -a \
  -s $MODEL \
  -f $LOGFILE \
  --target_idx $TGTIDX \
  --noutputs $NOUTPUTS
  # -p \

mv perfmat${NOUTPUTS}.npy $PERFMAT
