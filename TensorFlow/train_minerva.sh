#!/bin/bash

DAT=`date +%s`
MODEL_CODE="20170817"

# targets
NCLASS=11
TARGETS="--n_classes $NCLASS --targets_label segments"

TRAINING="--nodo_training"
TRAINING="--do_training"
VALIDATION="--do_validaton"

TESTING="--do_testing"
TESTING="--nodo_testing"

BASEP="/Users/gnperdue/Documents/MINERvA/AI/minerva_tf"

PREDPATH="${BASEP}/predictions/"
PREDFILE="$PREDPATH/predictions_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}"
PREDICTIONS="--do_prediction --pred_store_name $PREDFILE"
PREDICTIONS="--nodo_prediction"

# data, log, and model logistics
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_011"
DATADIR="${BASEP}/tfrec"
LOGDIR="${BASEP}/logs"
LOGFILE=$LOGDIR/log_mnv_st_epsilon_${NCLASS}_${MODEL_CODE}_${DAT}.txt
LOGLEVEL="--log_level INFO"
LOGLEVEL="--log_level DEBUG"
MODELDIR="${BASEP}/models/${NCLASS}/${MODEL_CODE}"

# show what we will do...
cat << EOF
python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TRAINING $VALIDATION $TESTING $PREDICTIONS
EOF

python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --model_dir $MODELDIR \
  --log_name $LOGFILE $LOGLEVEL \
  $TRAINING $VALIDATION $TESTING $PREDICTIONS

echo "Job finished "`date`""
