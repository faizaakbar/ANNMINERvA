#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./lasagne_conv_out_job${DAT}.txt"
JOBNAME="mnv-conv-${DAT}"
qsub -o $OUTFILENAME job_lasagne_beta_x.sh -N $JOBNAME
