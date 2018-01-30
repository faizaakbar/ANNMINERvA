#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./job_targ_hdf5${DAT}.txt"
JOBNAME="job_targ_hdf5${DAT}"
qsub -o $OUTFILENAME job_target_region_hdf5.sh -N $JOBNAME
