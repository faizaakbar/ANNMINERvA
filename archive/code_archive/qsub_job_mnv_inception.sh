#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./triamese_inception_out_job${DAT}.txt"
JOBNAME="mnv-inception-${DAT}"
qsub -o $OUTFILENAME job_mnv_triamese_inception.sh -N $JOBNAME
