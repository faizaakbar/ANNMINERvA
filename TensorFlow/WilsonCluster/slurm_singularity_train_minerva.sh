#!/bin/bash

echo "started "`date`" "`date +%s`""

nvidia-smi -L

DAT=`date +%s`

# me1BG -> train a model for A using B,G
# me1AG -> train a model for B using A,G
# be sure to update 'DATADIR'!

# "SAMPLE" -> here for the 'valid' label in the model code
SAMPLE="me1Amc"
SAMPLE="me1BGmc"
SAMPLE="me1AGmc"

# "TESTSAMPLE" -> here for the 'pred' label in the predictions file
PREDSAMPLE="me1Amc"

# "TRAINSAMPLE" -> here for the 'train' label in the model code
TRAINSAMPLE="me1Amc"
TRAINSAMPLE="me1BGmc"
TRAINSAMPLE="me1AGmc"

SHORT="--do_a_short_run"
SHORT=""
LOGLEVEL="--log_level DEBUG"
LOGLEVEL="--log_level INFO"
LOGDEVS="--do_log_devices"
LOGDEVS=""

NEPOCHS="--num_epochs 10"
NEPOCHS="--num_epochs 1"
NEPOCHS="--num_epochs 5"

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
FILEPAT="vtxfndingimgs_127x${IMGWX}_"

PLANECODES="--n_planecodes $NPLANECODES"
TARGETS="--n_classes $NCLASS --targets_label ${TARGLABEL}"
IMGPAR="--imgw_x $IMGWX --imgw_uv $IMGWUV"

# OPTIMIZER="AdaGrad"
OPTIMIZER="Adam"
STRATEGY="--strategy ${OPTIMIZER}"

BATCHF="do_batch_norm"
BATCHF="nodo_batch_norm"
BATCHNORM="--$BATCHF"

BATCHSIZE=1024
BATCH="--batch_size $BATCHSIZE"

GPU=`hostname`
MODEL_CODE="20180105_${GPU}_batch${BATCHSIZE}_${OPTIMIZER}_train${TRAINSAMPLE}_valid${SAMPLE}_${BATCHF}_${TARGLABEL}${NCLASS}"

# pick up singularity v2.2
export PATH=/usr/local/singularity/bin:$PATH
# which singularity image
SNGLRTY="/data/simone/singularity/ML/ubuntu16-ML.simg"
SNGLRTY="/data/perdue/singularity/tf_1_4.simg"


TRAINING="--nodo_training"
VALIDATION="--nodo_validaton"
TRAINING="--do_training"
VALIDATION="--do_validaton"

TESTING="--nodo_testing"
TESTING="--do_testing"

SPECIAL=""
SPECIAL="--use_all_for_test"
SPECIAL="--use_test_for_train --use_valid_for_test"

PREDPATH="/data/perdue/minerva/tensorflow/predictions/"
PREDFILE="$PREDPATH/mnv_st_epsilon_predictions${PREDSAMPLE}_model_${MODEL_CODE}"
PREDICTIONS="--do_prediction --pred_store_name $PREDFILE"
PREDICTIONS="--nodo_prediction"

BASEP="/data/perdue/minerva/tensorflow"
DBASE="${BASEP}/data/201710"
DATADIR="${DBASE}/${PREDSAMPLE}"
DATADIR="${DBASE}/me1Amc,${DBASE}/me1Gmc"
LOGDIR="${BASEP}/logs/201710/"
LOGFILE=$LOGDIR/log_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}_${DAT}.txt
MODELDIR="${BASEP}/models/${NCLASS}/${MODEL_CODE}"


# print identifying info for this job
cd /home/perdue/ANNMINERvA/TensorFlow/WilsonCluster
echo "Workdir is `pwd`"
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
  cp -v /home/perdue/ANNMINERvA/TensorFlow/$filename `pwd`
done

ARGSTR="--compression gz --data_dir $DATADIR --file_root $FILEPAT --model_dir $MODELDIR --log_name $LOGFILE $LOGLEVEL $TARGETS $TRAINING $VALIDATION $TESTING $PREDICTIONS $SPECIAL $SHORT $NEPOCHS $PLANECODES $IMGPAR $STRATEGY $BATCH $LOGDEVS $BATCHNORM"

# show what we will do...
cat << EOF
singularity exec $SNGLRTY python mnv_run_st_epsilon.py $ARGSTR
EOF

singularity exec $SNGLRTY python mnv_run_st_epsilon.py $ARGSTR


nvidia-smi -L >> $LOGFILE
nvidia-smi >> $LOGFILE

echo "finished "`date`" "`date +%s`""
exit 0
