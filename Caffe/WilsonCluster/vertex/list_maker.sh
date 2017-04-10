#!/bin/bash

BASED=/data/perdue/minerva/caffe/data
DESTD=/data/perdue/minerva/caffe/lists

# vertex adv trainlist based on ME 1B MC (source domain)
TRAINMIN=0
TRAINMAX=75
VALIDMIN=76
VALIDMAX=81
TESTMIN=82
TESTMAX=88
FILEROOT=minosmatch_nukecczdefs_genallz_pcodecap66_127x50x25_xuv_me1Bmc
OUTROOT=vertex_epsilon_adv_source
rm -f $OUTROOT.trainlist
rm -f $OUTROOT.validlist
rm -f $OUTROOT.testlist
for i in `seq $TRAINMIN 1 $TRAINMAX`
do
  filenum=`echo $i | perl -ne 'printf "%03d",$_;'`
  ls -1 ${BASED}/${FILEROOT}_${filenum}.hdf5 >> $OUTROOT.trainlist
done
for i in `seq $VALIDMIN 1 $VALIDMAX`
do
  filenum=`echo $i | perl -ne 'printf "%03d",$_;'`
  ls -1 ${BASED}/${FILEROOT}_${filenum}.hdf5 >> $OUTROOT.validlist
done
for i in `seq $TESTMIN 1 $TESTMAX`
do
  filenum=`echo $i | perl -ne 'printf "%03d",$_;'`
  ls -1 ${BASED}/${FILEROOT}_${filenum}.hdf5 >> $OUTROOT.testlist
done
mv $OUTROOT.trainlist $DESTD
mv $OUTROOT.validlist $DESTD
mv $OUTROOT.testlist $DESTD

# vertex adv trainlist based on LE Minerva1 & Minerva 13B MC (target domain)
# (no validation here - these are DANN partners...)
TRAINMIN=0
TRAINMAX=40
TESTMIN=41
TESTMAX=44
FILEROOT=minosmatch_nukecczdefs_genallz_pcodecap66_127x50x25_xuv_minerva1mc
OUTROOT=vertex_epsilon_adv_target
rm -f $OUTROOT.trainlist
rm -f $OUTROOT.testlist
for i in `seq $TRAINMIN 1 $TRAINMAX`
do
  filenum=`echo $i | perl -ne 'printf "%03d",$_;'`
  ls -1 ${BASED}/${FILEROOT}_${filenum}.hdf5 >> $OUTROOT.trainlist
done
for i in `seq $TESTMIN 1 $TESTMAX`
do
  filenum=`echo $i | perl -ne 'printf "%03d",$_;'`
  ls -1 ${BASED}/${FILEROOT}_${filenum}.hdf5 >> $OUTROOT.testlist
done
# vertex adv trainlist based on LE Minerva1 & Minerva 13B MC (target domain), cont
TRAINMIN=0
TRAINMAX=3
TESTMIN=4
TESTMAX=7
FILEROOT=minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xuv_minerva13Bmc
for i in `seq $TRAINMIN 1 $TRAINMAX`
do
  filenum=`echo $i | perl -ne 'printf "%03d",$_;'`
  ls -1 ${BASED}/${FILEROOT}_${filenum}.hdf5 >> $OUTROOT.trainlist
done
for i in `seq $TESTMIN 1 $TESTMAX`
do
  filenum=`echo $i | perl -ne 'printf "%03d",$_;'`
  ls -1 ${BASED}/${FILEROOT}_${filenum}.hdf5 >> $OUTROOT.testlist
done
mv $OUTROOT.trainlist $DESTD
mv $OUTROOT.testlist $DESTD
