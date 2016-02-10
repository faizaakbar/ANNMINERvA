#!/bin/bash
OUTFILENAME="./lasagne_conv_out_job`date +%s`.txt"
qsub -o $OUTFILENAME job_lasagne_conv_mlp.sh
