#!/bin/bash

tar -cvzf models.tgz models
scp models.tgz perdue@minervagpvm02.fnal.gov:/minerva/data/users/perdue/mlmpr
