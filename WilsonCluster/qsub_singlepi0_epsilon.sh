#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./singlepi0_conv_out_job${DAT}.txt"
JOBNAME="mnv-conv-${DAT}"
qsub -o $OUTFILENAME job_singlepi0_epsilon.sh -N $JOBNAME
