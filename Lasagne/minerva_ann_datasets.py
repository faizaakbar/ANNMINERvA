#!/usr/bin/env python
"""
Functions for loading and handling examples for MINERvA ANN work.
"""
import os

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from six.moves import range


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


def make_scheme_and_stream(dset, batchsize, shuffle=True):
    """
    dset is a Fuel `DataSet` and batchsize is an int representing the number of
    examples requested per minibatch - note assume we are always operating
    on minibatches (although they can be size 1)
    """
    if shuffle:
        scheme = ShuffledScheme(examples=dset.num_examples,
                                batch_size=batchsize)
    else:
        scheme = SequentialScheme(examples=dset.num_examples,
                                  batch_size=batchsize)
    data_stream = DataStream(dataset=dset,
                             iteration_scheme=scheme)
    return scheme, data_stream


def get_dataset_sizes(data_file):
    """
    Assume the data file has H5PYDatasets for 'train', 'valid', and 'test'; get
    the size of each and return a tuple with (#train, #valid, #test)
    """
    if os.path.exists(data_file):
        train_set = H5PYDataset(data_file, which_sets=('train',),
                                load_in_memory=False)
        train_size = train_set.num_examples
        valid_set = H5PYDataset(data_file, which_sets=('valid',),
                                load_in_memory=False)
        valid_size = valid_set.num_examples
        test_set = H5PYDataset(data_file, which_sets=('test',),
                               load_in_memory=False)
        test_size = test_set.num_examples
    else:
        raise Exception('Data file', data_file, 'not found!')

    return (train_size, valid_size, test_size)


def slices_maker(n, slice_size=100000):
    if n < slice_size:
        return [(0, n)]

    remainder = n % slice_size
    n = n - remainder
    nblocks = n // slice_size
    counter = 0
    slices = []
    for i in range(nblocks):
        end = counter + slice_size
        slices.append((counter, end))
        counter += slice_size

    slices.append((counter, counter + remainder))
    return slices


def load_datasubset(data_file, subset, slice_to_load):
    """
    Always load data in memory
    """
    if os.path.exists(data_file):
        dset = H5PYDataset(data_file, which_sets=(subset,),
                           subset=slice(slice_to_load[0], slice_to_load[1]),
                           load_in_memory=True)
    else:
        raise Exception('Data file', data_file, 'not found!')

    return dset
