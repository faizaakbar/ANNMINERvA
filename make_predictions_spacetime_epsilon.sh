#!/bin/sh

# predictions using 1A (trained with 1B, so use all of 1A)

MODEL="models/lminerva_spacetime_67_epsilon1480703388.npz"
DATAFILE="../HDF5files/minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc.hdf5"
LOGFILE=spacetime_epsilon_me1Amc_1480703388_predict.txt
DBNAME=prediction67_me1Amc_epsilon1480703388.db

# TGTIDX=5
# NOUTPUTS=11
TGTIDX=4
NOUTPUTS=67

# `-p` == make predictions (db file)
# `-t` == run test (no db file)
# `-v` == run in verbose mode
# `-a` == use the whole file as if it were the "test" sample
python Lasagne/minerva_tricolumnar_spacetime_epsilon.py \
  -d $DATAFILE \
  -p \
  -v \
  -a \
  -s $MODEL \
  -f $LOGFILE \
  --target_idx $TGTIDX \
  --noutputs $NOUTPUTS
  # -t \

mv prediction${NOUTPUTS}.db $DBNAME
