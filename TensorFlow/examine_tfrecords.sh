#!/bin/bash

# tag the log file
DAT=`date +%s`

BASEP="${HOME}/Documents/MINERvA/AI/minerva_tf"
DATADIR="${BASEP}/tfrec2"
LOGFILE="${BASEP}/logs/log_examine_tfrec${DAT}.txt"
OUTPAT="${BASEP}/logs/result_examine_tfrec${DAT}"

SAMPLE=me1Bmc
IMGWX=50
IMGWUV=25
NPLANECODES=67
PCODECAP=$(($NPLANECODES - 1))
PLANECODES="--n_planecodes $NPLANECODES"
IMGPAR="--imgw_x $IMGWX --imgw_uv $IMGWUV"
FILEPAT="minosmatch_nukecczdefs_genallzwitht_pcodecap${PCODECAP}_127x${IMGWX}x${IMGWUV}_xtxutuvtv_${SAMPLE}"

cat << EOF
python tfrec_examiner.py \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --compression gz \
  --log_name $LOGFILE \
  --out_pattern $OUTPAT \
  $PLANECODES $IMGPAR
EOF

python tfrec_examiner.py \
  --data_dir $DATADIR \
  --file_root $FILEPAT \
  --compression gz \
  --log_name $LOGFILE \
  --out_pattern $OUTPAT \
  $PLANECODES $IMGPAR
