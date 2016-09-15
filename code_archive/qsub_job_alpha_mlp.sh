#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./lasagne_alpha_out_job${DAT}.txt"
JOBNAME="mnv-alpha-${DAT}"
qsub -o $OUTFILENAME job_lasagne_alpha.sh -N $JOBNAME
