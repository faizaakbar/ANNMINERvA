#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./job_kine_hdf5${DAT}.txt"
JOBNAME="job_kine_hdf5${DAT}"
qsub -o $OUTFILENAME job_kine_hdf5.sh -N $JOBNAME
