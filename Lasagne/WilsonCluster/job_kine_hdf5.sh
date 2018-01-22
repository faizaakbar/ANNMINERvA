#!/bin/bash
#PBS -S /bin/bash
#PBS -N hdf5-prod
#PBS -j oe
#PBS -o ./hdf5_prod_out_job.txt
# not 2 #PBS -l nodes=gpu2:gpu:ppn=2,walltime=24:00:00
# not 1 #PBS -l nodes=gpu1:gpu:ppn=2,walltime=24:00:00
#PBS -l nodes=1:gpu,walltime=24:00:00
# not short #PBS -l nodes=1:gpu,walltime=6:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n

# process kine files into hdf5

START=11
STOP=256


SAMPLE=me1Amc
INPATH="/data/perdue/minerva/rawtxt/201700"
OUTPATH="/data/perdue/minerva/hdf5/201700"

INPUT_BASENAME=${INPATH}/minosmatch_kinematics_${SAMPLE}
OUTPUTNAME=${OUTPATH}/minosmatch_kinedat_${SAMPLE}

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

cp /home/perdue/ANNMINERvA/make_hdf5_fuelfiles.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/plane_codes.py ${PBS_O_WORKDIR}

for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
  python make_hdf5_fuelfiles.py \
    -b ${INPUT_BASENAME}_${paddednum} \
    -o ${OUTPUTNAME}_${paddednum}.hdf5 \
    -s kine_dat
done
