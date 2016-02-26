# MLMPR

Draft started as of 2016/Feb/25. I will try to keep it current as things
develop.

## Overview

The general steps for running the classifier are:

1. Generate ntuples. We are currently using the `NukeCCInclusive` analysis tool,
CVS revision 1.92 or later (it must run the `MLVFTool`). Some samples are 
already in existence (see below).
2. Produce a skim. The skim code lives in `Personal/wospakrk/NuclearTargetVertexing/ana/make_hist`. The current preferred skimmer is `NukeCCSkimmer_chunked_zsegments.cxx`. TODO: update this skimmer so it takes a playlist name and keeps that in the file
name string. There is a runner script for the skimmer. Some additional instructions
may be found below.
3. Use the skim files to produce a HDF5 file containing all the event data. (Yes,
all of it - for now.) The script for processing the files is `fuel_up_nukecc.py`.
The `minervagpvm` machines do _not_ have the needed Python packages installed
as part of the MINERvA software framework. Either use [Anaconda](https://www.continuum.io/downloads) on the `minervagpvm` machines or on a local machine of your choice.
More instructions are provided below.  

## Existing ntuple samples

NukeCC samples may be found in:

* minervame1B (112200 -> 112201): `/minerva/data/users/perdue/mc_ana_minervame1B_some_date`
* minervame1B (112202 -> 112205): `/minerva/data/users/perdue/mc_ana_minervame1B_some_other_date` (still in production)

## Skimmer instructions

Coming soon.

## HDF5 file production

Coming soon.

## To run on Wilson

    $ qsub job_mlp.sh
    $ qstat
