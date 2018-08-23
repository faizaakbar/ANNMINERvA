# MLMPR

To process deep learning codes using TensorFlow on the Wilson Cluster, see the
[DLRunScripts](https://github.com/gnperdue/DLRunScripts) package. For legacy
Theano and Caffe, see the scripts here, but note that they have not been
updated for the new batch processing system on the Wilson Cluster.

* `Caffe/` - scripts and prototxts for running vertex finding.
* `mnvtf/` - TensorFlow code for the MINERvA nuclear targets vertex
finder and some legacy run scripts.
  * `data_constants.py` - strings, structures, and constants used to specify HDF5 and TFRecord files.
  * `data_readers.py` - specialized classes for reading TFRecord files using the "old" batch-queue API.
  * `dset_data_readers.py` - specialized classes for reading TFRecord files using the `tf.data.Dataset` API.
  * `evtid_utils.py` - utilties for decoding the `eventid` fields in 64 bit or
  double-32 bit combos.
  * `hf5_readers.py` - specialized classes for reading HDF5 files.
  * `models_menagerie.py` - functions for specifying models (as dictionaries to be parsed by the `mnvtf` mini-framework).
  * `models_tricolumnar.py` - specialized classes for parsing models specified as dictionaries into three-branch convolutional models.
  * `reader_sqlite.py` - specialized classes for reading predictions recorded as SQLite files (requires `sqlalchemy`).
  * `reader_text.py` - specialized classes for reading predictions recorded as text files.
  * `recorder_sqlite.py` - specialized classes for recording predictions as SQLite files (requires `sqlalchemy`).
  * `recorder_text.py` - specialized classes for recording predictions as text files.
  * `runners.py` - specialized classes for running training and inference tasks.
  * `utils.py` - misc. utility functions.
* `archive/` - old code kept visible for reference.
* `dset_visualize.py` - event display viewer that may consume HDF5 or TFRecord
files.
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
