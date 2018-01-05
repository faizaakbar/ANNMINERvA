#!/bin/bash

NGPU=1
NODES=gpu2
NODES=gpu4
SCRIPT=slurm_singularity_train_minerva.sh

# show what we will do...
cat << EOF
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} ${SCRIPT}
EOF

# do the thing, etc.
sbatch --gres=gpu:${NGPU} --nodelist=${NODES} ${SCRIPT}
