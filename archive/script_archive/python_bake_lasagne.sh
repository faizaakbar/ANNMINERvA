#!/bin/bash

export CUDA_VISIBLE_DEVICES=`cat /tmp/pbs.prologue.$PBS_JOBID`

export PATH=/usr/local/python2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/python2/lib:$LD_LIBRARY_PATH

# hdf5
export LD_LIBRARY_PATH=/usr/local/hdf5/lib:$LD_LIBRARY_PATH

# nvcc compiler - maybe wrap in a hostname check for gpus...
export PATH=/usr/local/cuda-7.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
