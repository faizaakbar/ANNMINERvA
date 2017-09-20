#!/bin/bash

DAT=`date +%s`
MODEL_CODE="20170908"

# targets - use NCLASS when making the logfile & model dir names also
NCLASS=67
TARGLABEL="segments"
TARGETS="--n_classes $NCLASS --targets_label ${TARGLABEL}"
IMGPAR="--imgw_x 94 --imgw_uv 47"

TRAINING="--do_training"
VALIDATION="--do_validaton"
TRAINING="--nodo_training"

TESTING="--nodo_testing"
TESTING="--do_testing"

SPECIAL="--use_test_for_train --use_valid_for_test"
SPECIAL="--use_valid_for_test"
SPECIAL=""
SPECIAL="--use_all_for_test"

BASEP="/Users/gnperdue/Documents/MINERvA/AI/minerva_tf"

PREDPATH="${BASEP}/predictions/"
PREDFILE="$PREDPATH/predictions_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}"
PREDICTIONS="--nodo_prediction"
PREDICTIONS="--do_prediction --pred_store_name $PREDFILE"

# data, log, and model logistics
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_011"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_0000"
DATADIR="${BASEP}/tfrec"
MODELDIR="${BASEP}/models/${NCLASS}/${MODEL_CODE}"
LOGDIR="${BASEP}/logs"
LOGFILE=$LOGDIR/log_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}_${DAT}.txt
LOGLEVEL="--log_level INFO"
LOGLEVEL="--log_level DEBUG --be_verbose"

SHORT="--do_a_short_run"
SHORT=""

# show what we will do...
cat << EOF
python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TARGETS $TRAINING $VALIDATION $TESTING $PREDICTIONS $SPECIAL $SHORT
EOF

python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TARGETS $TRAINING $VALIDATION $TESTING $PREDICTIONS $SPECIAL $SHORT

echo "Job finished "`date`""
