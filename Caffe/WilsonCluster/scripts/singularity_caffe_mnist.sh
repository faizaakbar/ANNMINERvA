#!/bin/bash
#PBS -S /bin/bash
#PBS -N caffe-mnist-test
#PBS -j oe
#PBS -o ./out_job_caffe_mnist.txt
#PBS -l nodes=1:gpu,walltime=24:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n

# print identifying info for this job
echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

# these are broken?...
# nCores=$['cat ${PBS_COREFILE} | wc --lines']
# nNodes=$['cat ${PBS_NODEFILE} | wc --lines']
# echo "NODEFILE nNodes=$nNodes (nCores=$nCores):"

cat ${PBS_NODEFILE}

# pick up singularity v2.2
export PATH=/usr/local/singularity/bin:$PATH
# which singularity image
SNGLRTY="/data/simone/singularity/ML/ubuntu16-ML.simg"
CAFFE="/usr/local/caffe/bin/caffe"

cd ${PBS_O_WORKDIR}
echo "PBS_O_WORKDIR is `pwd`"

LMDBDIR=/data/perdue/mnist
NETWORKDIR=/data/perdue/mnist

echo " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
echo "                     Train                             "

singularity exec $SNGLRTY $CAFFE train \
  -solver $NETWORKDIR/lenet_solver.prototxt

echo " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
echo "                     Test                              "

singularity exec $SNGLRTY $CAFFE test \
  -model $NETWORKDIR/lenet_train_test.prototxt \
  -weights $NETWORKDIR/lenet_iter_10000.caffemodel \
  -gpu 0

exit 0
