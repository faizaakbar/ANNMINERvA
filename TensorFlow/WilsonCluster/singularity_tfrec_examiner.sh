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

# tag the log file
DAT=`date +%s`

SAMPLE=me1Amc
IMGWX=50
IMGWUV=25
NPLANECODES=67
PCODECAP=$(($NPLANECODES - 1))
PLANECODES="--n_planecodes $NPLANECODES"
IMGPAR="--imgw_x $IMGWX --imgw_uv $IMGWUV"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap${PCODECAP}_127x${IMGWX}x${IMGWUV}_xtxutuvtv_${SAMPLE}"

BASEP="/data/perdue/minerva/tensorflow"
DATADIR="${BASEP}/data/201709/${SAMPLE}"
LOGFILE="${BASEP}/logs/201709/${SAMPLE}/log_examine_tfrec_pcodecap${PCODECAP}_127x${IMGWX}x${IMGWUV}_xtxutuvtv_${SAMPLE}_${DAT}.txt"
OUTPAT="${BASEP}/logs/201709/${SAMPLE}/results_examine_tfrec_pcodecap${PCODECAP}_127x${IMGWX}x${IMGWUV}_xtxutuvtv_${SAMPLE}_${DAT}"


# which singularity image
SNGLRTY="/data/simone/singularity/ML/NEW/ubuntu16-cuda-tf1.3.img"

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

PYPROG=tfrec_examiner.py
cp /home/perdue/ANNMINERvA/TensorFlow/${PYPROG} ${PBS_O_WORKDIR}

# show what we will do...
cat << EOF
singularity exec $SNGLRTY python $PYPROG \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --compression gz \
  --log_name $LOGFILE \
  --out_pattern $OUTPAT \
  $PLANECODES $IMGPAR
EOF

singularity exec $SNGLRTY python $PYPROG \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --compression gz \
  --log_name $LOGFILE \
  --out_pattern $OUTPAT \
  $PLANECODES $IMGPAR


echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
