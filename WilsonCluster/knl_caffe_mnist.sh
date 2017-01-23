#!/bin/bash
#PBS -S /bin/bash
#PBS -N caffe-mnist-knl
#PBS -j oe
#PBS -o ./out_caffe_mnist_knl.txt
#PBS -l nodes=1:knl,walltime=24:00:00
#PBS -A minervaG
#PBS -q knl
#restore to turn off email #PBS -m n

# print identifying info for this job
echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

cat ${PBS_NODEFILE}

cd $HOME
source knl_caffe_setup.sh

cd ${PBS_O_WORKDIR}
echo "PBS_O_WORKDIR is `pwd`"

LMDBDIR=/data/perdue/mnist
NETWORKDIR=/data/perdue/mnist

echo " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
echo "                     Train                             "

caffe train -solver $NETWORKDIR/lenet_solver_cpu.prototxt

echo " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
echo "                     Test                              "

caffe test -model $NETWORKDIR/lenet_train_test.prototxt \
  -weights $NETWORKDIR/lenet_iter_10000.caffemodel -gpu 0

exit 0
