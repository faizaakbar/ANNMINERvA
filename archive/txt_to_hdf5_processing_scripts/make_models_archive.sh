#!/bin/bash
DATET=`date +%s`
ARCHIVE=models_archive$DATET
cp -r models $ARCHIVE
tar -cvzf ${ARCHIVE}.tgz $ARCHIVE
scp ${ARCHIVE}.tgz perdue@minervagpvm02.fnal.gov:/minerva/data/users/perdue/mlmpr
rm -rf $ARCHIVE
