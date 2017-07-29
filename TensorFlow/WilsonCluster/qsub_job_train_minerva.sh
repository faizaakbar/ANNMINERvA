#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./batchlog_singularity_train_minerva${DAT}.txt"
JOBNAME="train_minerva${DAT}"
qsub -o $OUTFILENAME singularity_train_minerva.sh -N $JOBNAME
