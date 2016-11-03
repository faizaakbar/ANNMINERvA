#!/bin/bash
#PBS -S /bin/bash
#PBS -N lasagne-conv-mnv
#PBS -j oe
#PBS -o ./lasagne_conv_out_job.txt
# not 2 #PBS -l nodes=gpu2:gpu:ppn=2,walltime=24:00:00
# not 1 #PBS -l nodes=gpu1:gpu:ppn=2,walltime=24:00:00
#PBS -l nodes=1:gpu,walltime=24:00:00
# not short #PBS -l nodes=1:gpu,walltime=6:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n

NEPOCHS=12
NEPOCHS=1
NEPOCHS=4
LRATE=0.001
L2REG=0.0001

DATET=`date +%s`

DOTEST=""
DOTEST="-t"

TGTIDX=5
NOUTPUTS=11

# TGTIDX=4
# NOUTPUTS=67

DATAFILENAME="/phihome/perdue/theano/data/minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Bmc.hdf5"
# SAVEMODELNAME="./lminerva_spacetime_${NOUTPUTS}_epsilon${DATET}.npz"
SAVEMODELNAME="./lminerva_spacetime_11_epsilon1478090717.npz"
PYTHONPROG="minerva_tricolumnar_spacetime_epsilon.py"

LOAD_SAVEMODEL=""
LOAD_SAVEMODEL="--load_params"

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
LOGFILENAME=minerva_tricolumnar_spacetime_epsilon_${NOUTPUTS}_${DATET}_${GIT_VERSION}.log

cp /home/perdue/ANNMINERvA/Lasagne/${PYTHONPROG} ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/minerva_ann_*.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/network_repr.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/predictiondb.py ${PBS_O_WORKDIR}

cat << EOF
python ${PYTHONPROG} -l $DOTEST \
  -n $NEPOCHS \
  -r $LRATE \
  -g $L2REG \
  -s $SAVEMODELNAME $LOAD_SAVEMODEL \
  -d $DATAFILENAME \
  -f $LOGFILENAME \
  --target_idx $TGTIDX \
  --noutputs $NOUTPUTS
EOF
export THEANO_FLAGS=device=gpu,floatX=float32
python ${PYTHONPROG} -l $DOTEST \
  -n $NEPOCHS \
  -r $LRATE \
  -g $L2REG \
  -s $SAVEMODELNAME $LOAD_SAVEMODEL \
  -d $DATAFILENAME \
  -f $LOGFILENAME \
  --target_idx $TGTIDX \
  --noutputs $NOUTPUTS

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
