# MLMPR

To process deep learning codes using TensorFlow on the Wilson Cluster, see the
[DLRunScripts](https://github.com/gnperdue/DLRunScripts) package. For legacy
Theano and Caffe, see the scripts here, but note that they have not been
updated for the new batch processing system on the Wilson Cluster.

* `Caffe/` - scripts and prototxts for running vertex finding.
* `mnvtf/` - TensorFlow code for the MINERvA nuclear targets vertex
finder and some legacy run scripts.
* `archive/` - old code kept visible for reference.
* `dset_visualize.py` - event display viewer that may consume HDF5 or TFRecord
files.
* `evtid_utils.py` - utilties for decoding the `eventid` fields in 64 bit or
double-32 bit combos.
* `examine_hdf5.py` - simple script to examine the structure and sizes of a
MINERvA HDF5 file.
* `hdf5_to_tfrec_minerva_xtxutuvtv.py` - script for converting HDF5 files to
TensorFlow TFRecord.
* `horovod_mnist.py` - script from Uber to run MNIST classification using
Horovod.
* `horovod_test.py` - test to see if we can initialize Horovod.
* `mnv_run_st_epsilon.py` - run classification using the "space-time" version
of the "epsilon" network architecture for vertex finding in the target
analysis. 
* `plane_codes.py` - legacy utilities code for converting the 'old' MINERvA
framework plane id numbers into sequential planecodes.
* `tfrec_examiner.py` - script that checks the number of records in a TFRecord
file and prints the `eventid` values to a log.
* `txt_to_sqlite.py` - converter script for writing text-based prediction
files into SQLite files.
