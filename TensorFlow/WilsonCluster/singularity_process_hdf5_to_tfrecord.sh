#!/bin/bash

# file creation parameters
NEVTS=20000
MAXTRIPS=2
TRAINFRAC=0.88
VALIDFRAC=0.06
TESTREAD="--test_read"
TESTREAD=""

# tag the log file
DAT=`date +%s`

# which singularity image
SNGLRTY="/data/perdue/singularity/simone/ubuntu16-cuda-ml.img"

# file logistics
HDF5DIR="/data/perdue/minerva/targets"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_011"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Bmc"
OUTDIR="/data/perdue/minerva/tensorflow/data"
LOGDIR="/data/perdue/minerva/tensorflow/logs"
LOGFILE=$LOGDIR/log_hdf5_to_tfrec_minerva_xtxutuvtv${DAT}.txt

# show what we will do...
cat << EOF
singularity exec $SNGLRTY python hdf5_to_tfrec_minerva_xtxutuvtv.py \
  --nevents $NEVTS \
  --max_triplets $MAXTRIPS \
  --file_pattern $FILEPAT \
  --in_dir $HDF5DIR \
  --out_dir $OUTDIR \
  --train_fraction $TRAINFRAC \
  --valid_fraction $VALIDFRAC \
  --logfile $LOGFILE \
  --compress_to_gz $TESTREAD
EOF

singularity exec $SNGLRTY python hdf5_to_tfrec_minerva_xtxutuvtv.py \
  --nevents $NEVTS \
  --max_triplets $MAXTRIPS \
  --file_pattern $FILEPAT \
  --in_dir $HDF5DIR \
  --out_dir $OUTDIR \
  --train_fraction $TRAINFRAC \
  --valid_fraction $VALIDFRAC \
  --logfile $LOGFILE \
  --compress_to_gz $TESTREAD
