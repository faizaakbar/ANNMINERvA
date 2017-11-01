#!/bin/bash
#PBS -S /bin/bash
#PBS -N tfrec_prod
#PBS -j oe
#PBS -o ./batchlog_tfrec_prod.txt
# not 3 #PBS -l nodes=gpu3:gpu,walltime=24:00:00
#PBS -l nodes=gpu2:gpu:ppn=2,walltime=24:00:00
# not 1 #PBS -l nodes=gpu1:gpu:ppn=2,walltime=24:00:00
# not generic #PBS -l nodes=1:gpu,walltime=24:00:00
# not short #PBS -l nodes=1:gpu,walltime=6:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email - doesn't work #PBS -m n

DAT=`date +%s`
SAMPLE="me1ABmc"
SAMPLE="me1Amc"
SAMPLE="me1Bmc"
SAMPLE="me1Gmc"

SHORT="--do_a_short_run"
SHORT=""
LOGLEVEL="--log_level DEBUG"
LOGLEVEL="--log_level INFO"
LOGDEVS="--do_log_devices"
LOGDEVS=""

NEPOCHS="--num_epochs 5"
NEPOCHS="--num_epochs 10"
NEPOCHS="--num_epochs 1"

NCLASS=173
NPLANECODES=173
IMGWX=94
IMGWUV=47
TARGLABEL="planecodes"

# NCLASS=67
# NPLANECODES=67
# IMGWX=50
# IMGWUV=25
# TARGLABEL="planecodes"

PCODECAP=$(($NPLANECODES - 1))
# FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap${PCODECAP}_127x${IMGWX}x${IMGWUV}_xtxutuvtv_${SAMPLE}"
# FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap${PCODECAP}_127x${IMGWX}x${IMGWUV}_xtxutuvtv_"
FILEPAT="vtxfndingimgs_127x${IMGWX}_${SAMPLE}_"

PLANECODES="--n_planecodes $NPLANECODES"
TARGETS="--n_classes $NCLASS --targets_label ${TARGLABEL}"
IMGPAR="--imgw_x $IMGWX --imgw_uv $IMGWUV"

# OPTIMIZER="AdaGrad"
OPTIMIZER="Adam"
STRATEGY="--strategy ${OPTIMIZER}"

BATCHF="do_batch_norm"
BATCHF="nodo_batch_norm"
BATCHNORM="--$BATCHF"

MODEL_CODE="20171030_${OPTIMIZER}_${SAMPLE}_${BATCHF}_${TARGLABEL}${NCLASS}"
# use the me1A model for classification
MODEL_CODE="20171030_${OPTIMIZER}_me1Amc_${BATCHF}_${TARGLABEL}${NCLASS}"

BATCHSIZE=500
BATCH="--batch_size $BATCHSIZE"

# pick up singularity v2.2
export PATH=/usr/local/singularity/bin:$PATH
# which singularity image
SNGLRTY="/data/simone/singularity/ML/ubuntu16-ML.simg"


TRAINING="--do_training"
VALIDATION="--do_validaton"
TRAINING="--nodo_training"
VALIDATION="--nodo_validaton"

TESTING="--nodo_testing"
TESTING="--do_testing"

SPECIAL=""
SPECIAL="--use_test_for_train --use_valid_for_test"
SPECIAL="--use_all_for_test"

PREDPATH="/data/perdue/minerva/tensorflow/predictions/"
PREDFILE="$PREDPATH/predictions_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}"
PREDICTIONS="--nodo_prediction"
PREDICTIONS="--do_prediction --pred_store_name $PREDFILE"

BASEP="/data/perdue/minerva/tensorflow"
DBASE="${BASEP}/data/201710"
# DATADIR="${DBASE}/${SAMPLE}"
DATADIR="${DBASE}/me1Amc,${DBASE}/me1Bmc"
DATADIR="${DBASE}/me1Bmc"
DATADIR="${DBASE}/me1Gmc"
LOGDIR="${BASEP}/logs/201710/"
LOGFILE=$LOGDIR/log_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}_${DAT}.txt
MODELDIR="${BASEP}/models/${NCLASS}/${MODEL_CODE}"


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
  $SPECIAL $SHORT $NEPOCHS $PLANECODES $IMGPAR $STRATEGY $BATCH \
  $LOGDEVS $BATCHNORM
EOF

singularity exec $SNGLRTY python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TARGETS $TRAINING $VALIDATION $TESTING $PREDICTIONS \
  $SPECIAL $SHORT $NEPOCHS $PLANECODES $IMGPAR $STRATEGY $BATCH \
  $LOGDEVS $BATCHNORM

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
