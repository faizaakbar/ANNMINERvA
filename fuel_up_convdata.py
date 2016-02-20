#!/usr/bin/env python
"""
Usage:
    python fuel_up_convdata.py <file key> [output name - optional]

The default output name is 'convdata_fuel.hdf5'
"""
from __future__ import print_function
import numpy as np
import sys
import os
import re

import h5py

from fuel.datasets.hdf5 import H5PYDataset


if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

if len(sys.argv) < 2:
    print('The filekey argument is mandatory.')
    print(__doc__)
    sys.exit(1)

filekey = sys.argv[1]

hdf5file = 'convdata_fuel.hdf5'
if len(sys.argv) > 2:
    hdf5file = sys.argv[2]

# "pixel" size of data images
#  here - W corresponds to MINERvA Z, and H correpsonds to the view axis
IMGW = 50
IMGH = 50


def get_data_from_file(filename):
    print("...loading data")
    targs = []
    data = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            elems = line.split()
            targs.append(int(elems[0]))
            rowdat = elems[1:]
            hitsX = []
            hitsU = []
            hitsV = []
            for point in rowdat:
                hit_data = point.split(':')
                view = hit_data[0]
                energy = float(hit_data[1])
                if view == 'X':
                    hitsX.append(energy)
                elif view == 'U':
                    hitsU.append(energy)
                    hitsU.append(0.0)
                elif view == 'V':
                    hitsV.append(energy)
                    hitsV.append(0.0)
            hitsX = np.asarray(hitsX, dtype=np.float32).reshape(IMGH, IMGW)
            hitsU = np.asarray(hitsU, dtype=np.float32).reshape(IMGH, IMGW)
            hitsV = np.asarray(hitsV, dtype=np.float32).reshape(IMGH, IMGW)
            energies = np.asarray([hitsX, hitsU, hitsV])
            data.append(energies)

    targs = np.asarray(targs, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    storedat = (data, targs)
    print("...finished loading")
    return storedat


# TODO: let the user pass in the base string here too
# get all the files and organize into a dictionary by category
filebase = re.compile(r"^nukecc_skim_data.*dat$")
files = os.listdir('.')
files = [f for f in files if re.match(filebase, f)]
files = [f for f in files if re.search(filekey, f)]

if os.path.exists(hdf5file):
    os.remove(hdf5file)
f = h5py.File(hdf5file, 'w')

data_set = f.create_dataset('hits', (0, 3, IMGW, IMGH),
                            dtype='float32', compression='gzip',
                            maxshape=(None, 3, IMGW, IMGH))
target_set = f.create_dataset('segments', (0,),
                              dtype='uint8', compression='gzip',
                              maxshape=(None,))

# `Fuel.H5PYDataset` allows us to label axes with semantic information;
# we record that in the file using "dimensional scales" (see h5py docs)
data_set.dims[0].label = 'batch'
data_set.dims[1].label = 'view(xuv)'
data_set.dims[2].label = 'height(view-coord)'
data_set.dims[3].label = 'width(z)'
target_set.dims[0].label = 'z-segment'

total_examples = 0

for fname in files:
    print("Iterating over file:", fname)
    data, targs = get_data_from_file(fname)
    examples_in_file = len(targs)
    print(" examples_in_file =", examples_in_file)
    existing_examples = np.shape(f['hits'])[0]
    print(" existing_examples =", existing_examples)
    total_examples = examples_in_file + existing_examples
    print(" resize =", total_examples)
    print(" idx slice = %d:%d" % (existing_examples, total_examples))
    f['hits'].resize(total_examples, axis=0)
    f['segments'].resize(total_examples, axis=0)
    f['hits'][existing_examples: total_examples] = data
    f['segments'][existing_examples: total_examples] = targs

final_train_index = int(total_examples * 0.8)
final_valid_index = int(total_examples * 0.9)

split_dict = {
    'train': {'hits': (0, final_train_index),
              'segments': (0, final_train_index)},
    'valid': {'hits': (final_train_index, final_valid_index),
              'segments': (final_train_index, final_valid_index)},
    'test': {'hits': (final_valid_index, total_examples),
             'segments': (final_valid_index, total_examples)}
    }
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.close()
