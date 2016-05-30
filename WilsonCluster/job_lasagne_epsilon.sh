#!/bin/bash
#PBS -S /bin/bash
#PBS -N lasagne-conv-mnv
#PBS -j oe
#PBS -o ./lasagne_conv_out_job.txt
# not 2 #PBS -l nodes=gpu2:gpu:ppn=2,walltime=24:00:00
# not 1 #PBS -l nodes=gpu1:gpu:ppn=2,walltime=24:00:00
#PBS -l nodes=1:gpu,walltime=24:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n

NEPOCHS=12
NEPOCHS=40
LRATE=0.001
L2REG=0.0001
NOUTPUTS=67
TGTIDX=4

DOTEST=""
DOTEST="-t"

DATAFILENAME="/phihome/perdue/theano/data/minosmatch_nukecczdefs_127x50x25_xuv_me1Bmc.hdf5"
SAVEMODELNAME="./lminervatriamese_epsilon`date +%s`.npz"
# SAVEMODELNAME="./transfer_to_epsilon_test2.npz"
# SAVEMODELNAME="./transfer_to_epsilon_noutputs67_test3.npz"
PYTHONPROG="minerva_triamese_epsilon.py"

# print identifying info for this job
echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

cat ${PBS_NODEFILE}

cd $HOME
source python_bake_lasagne.sh

cd ${PBS_O_WORKDIR}
echo "PBS_O_WORKDIR is `pwd`"
GIT_VERSION=`git describe --abbrev=12 --dirty --always`
echo "Git repo version is $GIT_VERSION"
DIRTY=`echo $GIT_VERSION | perl -ne 'print if /dirty/'`
if [[ $DIRTY != "" ]]; then
  echo "Git repo contains uncomitted changes! Please commit your changes"
  echo "before submitting a job. If you feel your changes are experimental,"
  echo "just use a feature branch."
  echo ""
  echo "Changed files:"
  git diff --name-only
  echo ""
  # exit 0
fi

cp /home/perdue/ANNMINERvA/Lasagne/${PYTHONPROG} ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/minerva_ann_*.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/network_repr.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/predictiondb.py ${PBS_O_WORKDIR}

cat << EOF
python ${PYTHONPROG} -l $DOTEST \
  -n $NEPOCHS \
  -r $LRATE \
  -g $L2REG \
  -s $SAVEMODELNAME \
  -d $DATAFILENAME \
  --noutputs $NOUTPUTS --tgtidx $TGTIDX
EOF
export THEANO_FLAGS=device=gpu,floatX=float32
python ${PYTHONPROG} -l $DOTEST \
  -n $NEPOCHS \
  -r $LRATE \
  -g $L2REG \
  -s $SAVEMODELNAME \
  -d $DATAFILENAME \
  --noutputs $NOUTPUTS --tgtidx $TGTIDX

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
