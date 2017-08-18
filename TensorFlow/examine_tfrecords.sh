#!/bin/bash

# tag the log file
DAT=`date +%s`

# file logistics
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_011_000000_test"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_011_000000_valid"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_011_000002_train"
DIR="/Users/gnperdue/Documents/MINERvA/AI/minerva_tf/tfrec"
LOGFILE=log_examine_tfrec${DAT}.txt

cat << EOF
python tfrec_examiner.py \
  --file_pattern $FILEPAT \
  --dir $DIR \
  --logfile $LOGFILE \
  --compressed_to_gz
EOF

python tfrec_examiner.py \
  --file_pattern $FILEPAT \
  --dir $DIR \
  --logfile $LOGFILE \
  --compressed_to_gz \
