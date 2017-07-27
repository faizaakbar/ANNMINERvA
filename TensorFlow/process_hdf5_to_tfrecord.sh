#!/bin/bash

DAT=`date +%s`
LOGFILE=log_hdf5_to_tfrec_minerva_xtxutuvtv${DAT}.txt

cat << EOF
python hdf5_to_tfrec_minerva_xtxutuvtv.py \
  --nevents 10000 \
  --file_pattern minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_ \
  --train_fraction 0.87 \
  --valid_fraction 0.07 \
  --logfile $LOGFILE \
  --compress_to_gz \
  --test_read
EOF
# --max_triplets 3 \
# --file_list minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_0000.hdf5 \
# --dry_run

python hdf5_to_tfrec_minerva_xtxutuvtv.py \
  --nevents 10000 \
  --file_pattern minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_ \
  --train_fraction 0.87 \
  --valid_fraction 0.07 \
  --logfile $LOGFILE \
  --compress_to_gz \
  --test_read
