export CUDA_VISIBLE_DEVICES=`cat /tmp/pbs.prologue.$PBS_JOBID`

export PATH=/usr/local/Python-2.7.8/bin:$PATH
export PATH=/usr/local/protobuf/bin:$PATH
export PATH=/usr/local/caffe.gabe-patch-2/bin:$PATH
# export PATH=/usr/local/caffe.gradient-reversal/bin:$PATH
export PATH=/usr/local/cuda/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/boost_1_59_0/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/gcc-4.9.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/gcc-4.9.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64/save:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/leveldb-1.18/lib64:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=/usr/local/opencv/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Python-2.7.8/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/protobuf/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/caffe.gabe-patch-2/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/caffe.gradient-reversal/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/hdf5/lib:$LD_LIBRARY_PATH
