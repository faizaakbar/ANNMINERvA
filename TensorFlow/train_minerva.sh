#!/bin/bash

DAT=`date +%s`
MODEL_CODE="12345"

# default is training, validation, and testing, no prediction

cat << EOF
python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir /Users/gnperdue/Documents/MINERvA/AI/ANNMINERvA/TensorFlow \
  --file_root minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_ \
  --model_dir /tmp/minerva/models/${MODEL_CODE} \
  --log_name log_mnv_st_epsilon${DAT}.txt \
  --do_training \
  --do_validation \
  --nodo_testing \
  --nodo_prediction 
EOF

python mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir /Users/gnperdue/Documents/MINERvA/AI/ANNMINERvA/TensorFlow \
  --file_root minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_ \
  --model_dir /tmp/minerva/models/${MODEL_CODE} \
  --log_name log_mnv_st_epsilon${DAT}.txt \
  --do_training \
  --do_validation \
  --nodo_testing \
  --nodo_prediction 
