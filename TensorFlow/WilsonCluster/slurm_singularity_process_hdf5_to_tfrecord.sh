#!/bin/bash

echo "started "`date`" "`date +%s`""

nvidia-smi -L

# file creation parameters
NEVTS=20000
NEVTS=10000
MAXTRIPS=1
MAXTRIPS=1000
TRAINFRAC=0.88
VALIDFRAC=0.06
TRAINFRAC=0.0
VALIDFRAC=0.0
TESTREAD="--test_read"
TESTREAD=""

# tag the log file
DAT=`date +%s`

# which singularity image
SNGLRTY="/data/perdue/singularity/tf_1_4.simg"

# file logistics
SAMPLE="me1Adata"
# STARTIDX=89
STARTIDX=0
HDF5DIR="/data/perdue/minerva/hdf5/201710"
FILEPAT="vtxfndingimgs_127x94_${SAMPLE}"
OUTDIR="/data/perdue/minerva/tensorflow/data/201710/${SAMPLE}"
LOGDIR="/data/perdue/minerva/tensorflow/logs/201710/${SAMPLE}"
LOGFILE=$LOGDIR/log_hdf5_to_tfrec_minerva_xtxutuvtv${DAT}.txt

mkdir -p $OUTDIR
mkdir -p $LOGDIR

# print identifying info for this job
echo "working directory is `pwd`"
GIT_VERSION=`git describe --abbrev=12 --dirty --always`
echo "Git repo version is $GIT_VERSION"
DIRTY=`echo $GIT_VERSION | perl -ne 'print if /dirty/'`
if [[ $DIRTY != "" ]]; then
  echo "Git repo contains uncomitted changes!"
  echo ""
  echo "Changed files:"
  git diff --name-only
  echo ""
  # exit 0
fi

cp /home/perdue/ANNMINERvA/TensorFlow/hdf5_to_tfrec_minerva_xtxutuvtv.py `pwd`

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
  --compress_to_gz $TESTREAD \
  --start_idx $STARTIDX
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
  --compress_to_gz $TESTREAD \
  --start_idx $STARTIDX

rm -f hdf5_to_tfrec_minerva_xtxutuvtv.py
rm -f hdf5_to_tfrec_minerva_xtxutuvtv.pyc

nvidia-smi -L >> $LOGFILE
nvidia-smi >> $LOGFILE

echo "finished "`date`" "`date +%s`""
exit 0
