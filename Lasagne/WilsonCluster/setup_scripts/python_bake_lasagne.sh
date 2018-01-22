#!/bin/bash

export PATH=/usr/local/python2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/python2/lib:$LD_LIBRARY_PATH

# hdf5
export LD_LIBRARY_PATH=/usr/local/hdf5/lib:$LD_LIBRARY_PATH

# nvcc compiler - maybe wrap in a hostname check for gpus...
export PATH=/usr/local/cuda/bin:$PATH
