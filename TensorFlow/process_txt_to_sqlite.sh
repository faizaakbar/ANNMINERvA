#!/bin/bash

DPATH="/Users/perdue/Documents/MINERvA/AI/minerva_tf/predictions/aghosh"
INPT="${DPATH}/DANN_me1Btrain_me1Adatapred.txt,${DPATH}/DANN_me1Btrain_me1Adatapred_last24events.txt"

cat << EOF
    python txt_to_sqlite.py -i $INPT
EOF
# default output name, n_classes, format all okay
python txt_to_sqlite.py -i $INPT
