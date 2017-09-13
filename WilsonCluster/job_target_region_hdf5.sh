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


START=0
STOP=58

DOENGY="no"
DOTIME="yes"

SAMPLE=me1Bmc

INPATH="/data/perdue/minerva/rawtxt/201709"
OUTPATH="/data/perdue/minerva/hdf5/201709"

# energy lattice images
INBASE="${INPATH}/minosmatch_nukecczdefs_fullz_tproc_127x94_${SAMPLE}"
OUTBASE="${OUTPATH}/minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xuv_${SAMPLE}"

# time-lattice images
INBASET="${INPATH}/minosmatch_nukecczdefs_fullzwitht_tproc_127x94_${SAMPLE}"
OUTBASET="${OUTPATH}/minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_txtutv_${SAMPLE}"


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

# do the work...
for i in `seq ${START} 1 ${STOP}`
do
  paddednum=`echo $i | perl -ne 'printf "%04d",$_;'`
# energy lattice
  if [ "$DOENGY" == "yes" ]; then
cat << EOF
  python make_hdf5_fuelfiles.py \
    -b ${INBASE}_${paddednum} \
    -o ${OUTBASE}_${paddednum}.hdf5 \
    -x \
    --trim_column_down_x 50 \
    --trim_column_down_uv 25 \
    --cap_planecode 66 \
    --min_keep_z -2e6
EOF
    python make_hdf5_fuelfiles.py \
      -b ${INBASE}_${paddednum} \
      -o ${OUTBASE}_${paddednum}.hdf5 \
      -x \
      --trim_column_down_x 50 \
      --trim_column_down_uv 25 \
      --cap_planecode 66 \
      --min_keep_z -2e6
fi
# time lattice
  if [ "$DOTIME" == "yes" ]; then
cat << EOF
  python make_hdf5_fuelfiles.py \
    --skim time_dat \
    -b ${INBASET}_${paddednum} \
    -o ${OUTBASET}_${paddednum}.hdf5 \
    -x \
    --trim_column_down_x 50 \
    --trim_column_down_uv 25 \
    --cap_planecode 66
EOF
    python make_hdf5_fuelfiles.py \
      --skim time_dat \
      -b ${INBASET}_${paddednum} \
      -o ${OUTBASET}_${paddednum}.hdf5 \
      -x \
      --trim_column_down_x 50 \
      --trim_column_down_uv 25 \
      --cap_planecode 66
fi
done

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
