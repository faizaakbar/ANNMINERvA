#!/bin/bash
#PBS -S /bin/bash
#PBS -N tfrec_prod
#PBS -j oe
#PBS -o ./batchlog_tfrec_prod.txt
# not 3 #PBS -l nodes=gpu3:gpu,walltime=24:00:00
# not 2 #PBS -l nodes=gpu2:gpu:ppn=2,walltime=24:00:00
# not 1 #PBS -l nodes=gpu1:gpu:ppn=2,walltime=24:00:00
#PBS -l nodes=1:gpu,walltime=24:00:00
# not short #PBS -l nodes=1:gpu,walltime=6:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email - doesn't work #PBS -m n

DAT=`date +%s`
SAMPLE="me1Amc"
MODEL_CODE="20170919_${SAMPLE}"

SHORT="--do_a_short_run"
SHORT=""
LOGLEVEL="--log_level DEBUG"
LOGLEVEL="--log_level INFO"

NEPOCHS="--num_epochs 3"

# which singularity image
SNGLRTY="/data/simone/singularity/ML/NEW/ubuntu16-cuda-tf1.3.img"

# targets
NCLASS=11
TARGETS="--n_classes $NCLASS --targets_label segments"
NCLASS=67
TARGETS="--n_classes $NCLASS --targets_label planecodes"

TRAINING="--nodo_training"
TRAINING="--do_training"
VALIDATION="--do_validaton"

TESTING="--nodo_testing"
TESTING="--do_testing"

SPECIAL="--use_all_for_test"
SPECIAL=""
SPECIAL="--use_test_for_train --use_valid_for_test"

PREDPATH="/data/perdue/minerva/tensorflow/predictions/"
PREDFILE="$PREDPATH/predictions_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}"
PREDICTIONS="--nodo_prediction"
PREDICTIONS="--do_prediction --pred_store_name $PREDFILE"

FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_${SAMPLE}"
DATADIR="/data/perdue/minerva/tensorflow/data/201709/${SAMPLE}"
LOGDIR="/data/perdue/minerva/tensorflow/logs/201709/${SAMPLE}"
LOGFILE=$LOGDIR/log_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}_${DAT}.txt
MODELDIR="/data/perdue/minerva/tensorflow/models/${NCLASS}/${MODEL_CODE}"


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

PYTHONLIST="
MnvDataReaders.py
MnvModelsTricolumnar.py
MnvRecorderSQLite.py
MnvRecorderText.py
mnv_run_st_epsilon.py
MnvTFRunners.py
mnv_utils.py"

for filename in $PYTHONLIST
do
  cp -v /home/perdue/ANNMINERvA/TensorFlow/$filename ${PBS_O_WORKDIR}
done

# show what we will do...
cat << EOF
singularity exec $SNGLRTY python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TARGETS $TRAINING $VALIDATION $TESTING $PREDICTIONS \
  $SPECIAL $SHORT $NEPOCHS
EOF

singularity exec $SNGLRTY python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TARGETS $TRAINING $VALIDATION $TESTING $PREDICTIONS \
  $SPECIAL $SHORT $NEPOCHS

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
