#!/bin/bash
#PBS -S /bin/bash
#PBS -N caffe-vtx-dann
#PBS -j oe
#PBS -o ./out_job_caffe_vtx_dann.txt
#PBS -l nodes=1:gpu,walltime=24:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n

# print identifying info for this job
echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"
cat ${PBS_NODEFILE}

cd $HOME
source caffe_setup.sh

cd ${PBS_O_WORKDIR}
echo "PBS_O_WORKDIR is `pwd`"

SOLVERDIR=/phihome/perdue/caffe/solvers
PROTODIR=/phihome/perdue/caffe/proto
SNAPSHOTDIR=/phihome/perdue/caffe/snapshots

# iters are defined in the solver
SOLVER=$SOLVERDIR/vertex_epsilon_adv.solver
ITERS=2000
PROTO_TRAIN=$PROTODIR/vertex_epsilon_adv.prototxt
PROTO_TEST=$PROTODIR/vertex_epsilon_adv_test.prototxt
SNAPS=$SNAPSHOTDIR/vertex_epsilon_adv_iter_${ITERS}.caffemodel

echo " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
echo "                     Train                             "

caffe.bin train -solver $SOLVER

echo " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
echo "                     Test                              "

caffe.bin test \
  -model $PROTO_TEST \
  -weights $SNAPS -gpu 0

exit 0
