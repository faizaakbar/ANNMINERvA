#!/usr/bin/env python
"""
This is an attempt at a "triamese" network operating on Minerva X, U, V.

Execution:
    python minerva_triamese_lasagnefuel.py -h / --help

At a minimum, we must supply either the `--train` or `--predict` flag.

See ANNMINERvA/fuel_up_nukecc.py for an HDF5 builder that sets up an
appropriate data file.

"""
from __future__ import print_function

import sys
import os

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.server import start_server


def load_dataset(data_file, load_in_memory=False):
    """
    See ANNMINERvA/fuel_up_convdata.py for an HDF5 builder that sets up an
    appropriate data file.
    """
    if os.path.exists(data_file):
        train_set = H5PYDataset(data_file, which_sets=('train',),
                                load_in_memory=load_in_memory)
        valid_set = H5PYDataset(data_file, which_sets=('valid',),
                                load_in_memory=load_in_memory)
        test_set = H5PYDataset(data_file, which_sets=('test',),
                                load_in_memory=load_in_memory)
    else:
        raise Exception('Data file', data_file, 'not found!')

    return train_set, valid_set, test_set


def make_scheme_and_stream(dset, batchsize, msg_string, shuffle=True):
    """
    dset is a Fuel `DataSet` and batchsize is an int representing the number of
    examples requested per minibatch
    """
    if shuffle:
        print(msg_string +
              " Preparing shuffled datastream for {} examples.".format(
                  dset.num_examples))
        scheme = ShuffledScheme(examples=dset.num_examples,
                                batch_size=batchsize)
    else:
        print(msg_string +
              "Preparing sequential datastream for {} examples.".format(
                  dset.num_examples))
        scheme = SequentialScheme(examples=dset.num_examples,
                                  batch_size=batchsize)
    data_stream = DataStream(dataset=dset,
                             iteration_scheme=scheme)
    return scheme, data_stream


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data', dest='dataset',
                      default='./skim_data_convnet.hdf5',
                      help='Data set', metavar='DATASET')
    parser.add_option('-p', '--port', dest='port',
                      default=55557, type='int',
                      help='Network port', metavar='PORT')
    parser.add_option('-m', '--hwm', dest='hwm',
                      default=10, type='int',
                      help='Highwater mark', metavar='HIGH_WATER')
    parser.add_option('-y', '--load_in_memory', dest='load_in_memory',
                      default=False, help='Attempt to load full dset in memory',
                      metavar='LOAD_IN_MEMORY', action='store_true')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=500,
                      help='Batch size for SGD', metavar='BATCH_SIZE',
                      type='int')
    parser.add_option('-l', '--learn', dest='do_learn', default=False,
                      help='Serve learning data', metavar='SERVE_LEARN',
                      action='store_true')
    parser.add_option('-v', '--valid', dest='do_valid', default=False,
                      help='Serve validation data', metavar='SERVE_VALID',
                      action='store_true')
    parser.add_option('-t', '--test', dest='do_test', default=False,
                      help='Serve test data', metavar='SERVE_TEST',
                      action='store_true')
    (options, args) = parser.parse_args()

    print("Starting...")
    print(__file__)
    dataset_statsinfo = os.stat(options.dataset)
    print(" Dataset:", options.dataset)
    print(" Dataset size:", dataset_statsinfo.st_size)

    nopts = 0
    for i in [options.do_learn, options.do_valid, options.do_test]:
        if i == True:
            nopts += 1

    if nopts != 1:
        print("\nMust specify one of learn, valid, or test:\n\n")
        print(__doc__)        
        sys.exit(1)

    learn_dset, valid_dset, test_dset = load_dataset(options.dataset,
                                                     options.load_in_memory)

    data_stream = None
    if options.do_test:
        _, data_stream = make_scheme_and_stream(test_dset,
                                                options.batch_size,
                                                "Processing training data:")
    elif options.do_valid:
        _, data_stream = make_scheme_and_stream(valid_dset,
                                                options.batch_size,
                                                "Processing training data:")
    elif options.do_learn:
        _, data_stream = make_scheme_and_stream(learn_dset,
                                                options.batch_size,
                                                "Processing training data:")
    
    if data_stream is not None:
        start_server(data_stream,
                     port=options.port,
                     hwm=options.hwm)
    else:
        print("Failure to create a data stream!")
