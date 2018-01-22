# MLMPR

To process deep learning codes using TensorFlow on the Wilson Cluster, see the
[DLRunScripts](https://github.com/gnperdue/DLRunScripts) package. For legacy
Theano and Caffe, see the scripts here, but note that they have not been
updated for the new batch processing system on the Wilson Cluster.

* `Caffe/` - scripts and prototxts for running vertex finding using PBS and
Slurm on the Wilson Cluster at Fermilab.
* `Lasagne/` - Theano and Lasagne code for the MINERvA nuclear targets
vertex finder and PBS scripts and some logs for runs on the Wilson Cluster at
Fermilab
* `TensorFlow/` - TensorFlow code for the MINERvA nuclear targets vertex
finder and some legacy run scripts.
* `archive/` - old code kept visible for reference.
* `dset_visualize.py` - event display viewer that may consume HDF5 or TFRecord
files.
* `evtid_utils.py` - utilties for decoding the `eventid` fields in 64 bit or
double-32 bit combos.
* `examine_hdf5.py` - simple script to examine the structure and sizes of a
MINERvA HDF5 file.
* `make_hdf5_fuelfiles.py` - legacy script for converting text files to HDF5.
* `perf_plots.py` - plotter that consumes the performance confusion matrices
produced by the ML codes.
* `plane_codes.py` - legacy utilities code for converting the 'old' MINERvA
framework plane id numbers into sequential planecodes.
* `processing_scripts/` - mostly legacy scripts for converting text files
to HDF5.
