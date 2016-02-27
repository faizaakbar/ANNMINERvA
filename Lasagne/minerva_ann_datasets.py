#!/usr/bin/env python
"""
Functions for loading and handling examples for MINERvA ANN work.
"""
from __future__ import print_function

import os

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream


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
