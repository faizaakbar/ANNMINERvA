#!/bin/bash
#PBS -S /bin/bash
#PBS -N mlp-test
#PBS -j oe
#PBS -o ./out_job.txt
#PBS -l nodes=1:gpu,walltime=02:00:00
#PBS -A fwk
#PBS -q gpu
#restore to turn off email #PBS -m n

# print identifying info for this job
echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

# these are broken?...
# nCores=$['cat ${PBS_COREFILE} | wc --lines']
# nNodes=$['cat ${PBS_NODEFILE} | wc --lines']
# echo "NODEFILE nNodes=$nNodes (nCores=$nCores):"

cat ${PBS_NODEFILE}

cd $HOME
source caffe_gpu_setup.sh

cd ${PBS_O_WORKDIR}
echo "PBS_O_WORKDIR is `pwd`"

# Always use fcp to stage any large input files from the cluster file server
# to your job's control worker node. All worker nodes have attached 
# disk storage in /scratch.

# There is no fcp on the gpu nodes...
# /usr/local/bin/fcp -c /usr/bin/rcp tevnfsp:/home/perdue/Datasets/mnist.pkl.gz /scratch
# ls /scratch

cp /home/perdue/ANNMINERvA/logistic_sgd.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/mlp_1h.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/mlp_2h.py ${PBS_O_WORKDIR}

export THEANO_FLAGS=device=gpu,floatX=float32
python mlp_2h.py -t -p -d "/home/perdue/ANNMINERvA/skim_data_target0.pkl.gz"

# Always use fcp to copy any large result files you want to keep back
# to the file server before exiting your script. The /scratch area on the
# workers is wiped clean between jobs.

# not really large, but okay... but, no fcp available
# /usr/local/bin/fcp -c /usr/bin/rcp mlp_best_model.pkl /home/perdue
# the pkl should just be in my launch dir...

exit 0
