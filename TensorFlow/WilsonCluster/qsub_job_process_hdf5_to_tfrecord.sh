#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./batchlog_singularity_process_hdf5_to_tfrecord${DAT}.txt"
JOBNAME="tfrec_producer${DAT}"
qsub -o $OUTFILENAME singularity_process_hdf5_to_tfrecord.sh -N $JOBNAME
