#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./job_fulldet_hdf5${DAT}.txt"
JOBNAME="job_fulldet_hdf5${DAT}"
qsub -o $OUTFILENAME job_fulldet_hdf5.sh -N $JOBNAME
