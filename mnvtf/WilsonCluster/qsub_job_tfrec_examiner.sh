#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./batchlog_singularity_tfrec_examiner${DAT}.txt"
JOBNAME="tfrec_producer${DAT}"
qsub -o $OUTFILENAME singularity_tfrec_examiner.sh -N $JOBNAME
