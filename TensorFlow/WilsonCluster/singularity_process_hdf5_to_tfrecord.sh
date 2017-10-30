#!/bin/bash
#PBS -S /bin/bash
#PBS -N tfrec_prod
#PBS -j oe
#PBS -o ./batchlog_tfrec_prod.txt
# not 3 #PBS -l nodes=gpu2:gpu,walltime=24:00:00
# not 2 #PBS -l nodes=gpu2:gpu:ppn=2,walltime=24:00:00
# not 1 #PBS -l nodes=gpu1:gpu:ppn=2,walltime=24:00:00
#PBS -l nodes=1:gpu,walltime=24:00:00
# not short #PBS -l nodes=1:gpu,walltime=6:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email - doesn't work #PBS -m n

# file creation parameters
NEVTS=20000
NEVTS=10000
MAXTRIPS=1
MAXTRIPS=1000
TRAINFRAC=0.88
VALIDFRAC=0.06
TESTREAD="--test_read"
TESTREAD=""

# tag the log file
DAT=`date +%s`

# which singularity image
# SNGLRTY="/data/simone/singularity/ML/NEW/ubuntu16-cuda-tf1.3.img"
SNGLRTY="/data/simone/singularity/ubuntu16-cuda8-cudnn6-ml.img"

# file logistics
SAMPLE="me1Bmc"
HDF5DIR="/data/perdue/minerva/hdf5/201709"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_${SAMPLE}"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap172_127x94x47_xtxutuvtv_${SAMPLE}"
OUTDIR="/data/perdue/minerva/tensorflow/data/201709/${SAMPLE}"
LOGDIR="/data/perdue/minerva/tensorflow/logs/201709/${SAMPLE}"
LOGFILE=$LOGDIR/log_hdf5_to_tfrec_minerva_xtxutuvtv${DAT}.txt

# print identifying info for this job
echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"
cat ${PBS_NODEFILE}
cd ${PBS_O_WORKDIR}
echo "PBS_O_WORKDIR is `pwd`"
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

cp /home/perdue/ANNMINERvA/TensorFlow/hdf5_to_tfrec_minerva_xtxutuvtv.py ${PBS_O_WORKDIR}

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

rm -f ${PBS_O_WORKDIR}/hdf5_to_tfrec_minerva_xtxutuvtv.py
rm -f ${PBS_O_WORKDIR}/hdf5_to_tfrec_minerva_xtxutuvtv.pyc

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
