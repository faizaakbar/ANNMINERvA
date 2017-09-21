#!/bin/bash

DAT=`date +%s`
SAMPLE=me1Amc

# targets - note, `n_planecodes` may be different than `nclass` - we need to
# know the number of planecodes when unpacking even when targeting semgnets.
NCLASS=173
NPLANECODES=173
PLANECODES="--n_planecodes $NPLANECODES"
TARGLABEL="planecodes"
TARGETS="--n_classes $NCLASS --targets_label ${TARGLABEL}"
IMGPAR="--imgw_x 94 --imgw_uv 47"

SHORT=""
SHORT="--do_a_short_run"
LOGLEVEL="--log_level INFO"
LOGLEVEL="--log_level DEBUG --be_verbose"

MODEL_CODE="20170920_${SAMPLE}_${TARGLABEL}${NCLASS}"

TRAINING="--nodo_training"
TRAINING="--do_training"
VALIDATION="--do_validaton"

TESTING="--nodo_testing"
TESTING="--do_testing"

SPECIAL="--use_all_for_test"
SPECIAL="--use_test_for_train --use_valid_for_test"
SPECIAL="--use_valid_for_test"
SPECIAL=""

BASEP="/Users/gnperdue/Documents/MINERvA/AI/minerva_tf"

PREDPATH="${BASEP}/predictions/"
PREDFILE="$PREDPATH/predictions_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}"
PREDICTIONS="--nodo_prediction"
PREDICTIONS="--do_prediction --pred_store_name $PREDFILE"

# data, log, and model logistics
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap172_127x94x47_xtxutuvtv_me1Amc_000000"
DATADIR="${BASEP}/tfrec"
MODELDIR="${BASEP}/models/${NCLASS}/${MODEL_CODE}"
LOGDIR="${BASEP}/logs"
LOGFILE=$LOGDIR/log_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}_${DAT}.txt

# show what we will do...
cat << EOF
python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TARGETS $TRAINING $VALIDATION $TESTING $PREDICTIONS \
  $SPECIAL $SHORT $PLANECODES $IMGPAR
EOF

python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TARGETS $TRAINING $VALIDATION $TESTING $PREDICTIONS \
  $SPECIAL $SHORT $PLANECODES $IMGPAR

echo "Job finished "`date`""
