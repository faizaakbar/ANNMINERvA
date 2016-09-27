#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./caffe_vtx_dann_job${DAT}.txt"
JOBNAME="mnv-conv-${DAT}"
qsub -o $OUTFILENAME job_caffe_vtx_dann.sh -N $JOBNAME
