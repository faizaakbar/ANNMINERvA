#!/bin/bash

NGPU=2

NODES=gpu4

# show what we will do...
cat << EOF
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} slurm_singularity_horovod_mnist.sh
EOF

# do the thing, etc.
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} slurm_singularity_horovod_mnist.sh
