#!/bin/bash
#PBS -S /bin/bash
#PBS -N lasagne-conv-mnv
#PBS -j oe
#PBS -o ./lasagne_conv_out_job.txt
#for gpu2 #PBS -l nodes=gpu2:gpu:ppn=1,walltime=24:00:00
#PBS -l nodes=1:gpu,walltime=24:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n

NEPOCHS=1
NEPOCHS=20
LRATE=0.0025
L2REG=0.0001

# minerva_triamese_lasagnefuel.py style...
DATAFILENAME="/phihome/perdue/theano/data/minosmatch_fuel_me1Amc_zseg0000.hdf5"

# SAVEMODELNAME="./lminervatriamese_model`date +%s`.npz"
SAVEMODELNAME="./lminervatriamese_model1457383290_alpha_v1r0.npz"
PYTHONPROG="minerva_triamese_lasagnefuel.py"

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

# cat << EOF
# python ${PYTHONPROG} -l \
#   -n $NEPOCHS \
#   -r $LRATE \
#   -g $L2REG \
#   -s $SAVEMODELNAME \
#   -d $DATAFILENAME 
# # $START_FROM
# EOF
# export THEANO_FLAGS=device=gpu,floatX=float32
# python ${PYTHONPROG} -l \
#   -n $NEPOCHS \
#   -r $LRATE \
#   -g $L2REG \
#   -s $SAVEMODELNAME \
#   -d $DATAFILENAME
 # $START_FROM

# nepochs and lrate don't matter for prediction, but setting them for log-file
# homogeneity
cat << EOF
python ${PYTHONPROG} -t \
  -n $NEPOCHS \
  -r $LRATE \
  -g $L2REG \
  -d $DATAFILENAME \
  -s $SAVEMODELNAME -a
EOF
python ${PYTHONPROG} -t \
  -n $NEPOCHS \
  -r $LRATE \
  -g $L2REG \
  -d $DATAFILENAME \
  -s $SAVEMODELNAME -a

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
