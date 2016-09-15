#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./hadmult_conv_out_job${DAT}.txt"
JOBNAME="mnv-conv-${DAT}"
qsub -o $OUTFILENAME job_hadmult_epsilon.sh -N $JOBNAME
