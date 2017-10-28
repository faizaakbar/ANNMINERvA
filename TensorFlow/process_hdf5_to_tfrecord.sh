#!/bin/bash

# file creation parameters
NEVTS=3000
MAXTRIPS=3
TRAINFRAC=0.88
VALIDFRAC=0.06

# tag the log file
DAT=`date +%s`

# file logistics
HDF5DIR="${HOME}/Documents/MINERvA/AI/hdf5"
FILEPAT="vtxfndingimgs_127x94_me1Bmc"
OUTDIR="${HOME}/Documents/MINERvA/AI/minerva_tf/tfrec"
LOGDIR="${HOME}/Documents/MINERvA/AI/minerva_tf/logs"
LOGFILE=$LOGDIR/log_hdf5_to_tfrec_minerva_xtxutuvtv${DAT}.txt

cat << EOF
python hdf5_to_tfrec_minerva_xtxutuvtv.py \
  --nevents $NEVTS \
  --max_triplets $MAXTRIPS \
  --file_pattern $FILEPAT \
  --in_dir $HDF5DIR \
  --out_dir $OUTDIR \
  --train_fraction $TRAINFRAC \
  --valid_fraction $VALIDFRAC \
  --logfile $LOGFILE \
  --compress_to_gz \
  --test_read
EOF

python hdf5_to_tfrec_minerva_xtxutuvtv.py \
  --nevents $NEVTS \
  --max_triplets $MAXTRIPS \
  --file_pattern $FILEPAT \
  --in_dir $HDF5DIR \
  --out_dir $OUTDIR \
  --train_fraction $TRAINFRAC \
  --valid_fraction $VALIDFRAC \
  --logfile $LOGFILE \
  --compress_to_gz \
  --test_read
# --dry_run
