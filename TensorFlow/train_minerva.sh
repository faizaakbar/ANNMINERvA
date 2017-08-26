#!/bin/bash

DAT=`date +%s`
MODEL_CODE="20170825"

# targets
TARGETS="--n_classes 11 --targets_label segments"
TARGETS="--n_classes 67 --targets_label planecodes"

TRAINING="--nodo_training"
TRAINING="--do_training"
VALIDATION="--do_validaton"

TESTING="--nodo_testing"
TESTING="--do_testing"

SPECIAL=""
SPECIAL="--use_all_for_test"
SPECIAL="--use_test_for_train --use_valid_for_test"
SPECIAL="--use_valid_for_test"

BASEP="/Users/gnperdue/Documents/MINERvA/AI/minerva_tf"

PREDPATH="${BASEP}/predictions/"
PREDFILE="$PREDPATH/predictions_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}"
PREDICTIONS="--nodo_prediction"
PREDICTIONS="--do_prediction --pred_store_name $PREDFILE"

# data, log, and model logistics
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_011"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_0000"
DATADIR="${BASEP}/tfrec"
LOGDIR="${BASEP}/logs"
LOGFILE=$LOGDIR/log_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}_${DAT}.txt
LOGLEVEL="--log_level INFO"
LOGLEVEL="--log_level DEBUG --be_verbose"
MODELDIR="${BASEP}/models/${NCLASS}/${MODEL_CODE}"

# show what we will do...
cat << EOF
python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TRAINING $VALIDATION $TESTING $PREDICTIONS $SPECIAL
EOF

python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TRAINING $VALIDATION $TESTING $PREDICTIONS $SPECIAL

echo "Job finished "`date`""
