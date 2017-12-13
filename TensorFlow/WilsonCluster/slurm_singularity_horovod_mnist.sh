#!/bin/bash

echo "started "`date`" "`date +%s`""

nvidia-smi -L

NGPU=2

# pick up singularity v2.2
export PATH=/usr/local/singularity/bin:$PATH
# which singularity image
SNGLRTY="/data/perdue/singularity/tf_1_4.simg"

# get mpirun in PATH
# export PATH=/usr/local/openmpi/bin:$PATH
export PATH=/usr/local/openmpi-1.10.7/bin:$PATH

cp -v /home/perdue/ANNMINERvA/TensorFlow/horovod_*.py `pwd`

# show what we will do...
cat << EOF
singularity exec $SNGLRTY python horovod_test.py
EOF
# do the thing...
singularity exec $SNGLRTY python horovod_test.py

# show what we will do...
cat << EOF
mpirun -np ${NGPU} -H localhost -bind-to core -map-by core singularity exec $SNGLRTY python horovod_mnist.py
EOF
# do the thing...
mpirun -np ${NGPU} -H localhost -bind-to core -map-by core singularity exec $SNGLRTY python horovod_mnist.py

nvidia-smi

echo "finished "`date`" "`date +%s`""
exit 0
