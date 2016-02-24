#!/usr/bin/env python
"""
Usage:
    python fuel_up_nukecc.py [output name - optional]

The default output name is 'nukecc_fuel.hdf5'. This script expects files named
`nukecc_skim_data*.dat` and expects data layout like:
    <zsegment> <run> <subrun> <gate> <slice[0]> X:[E] U:[E] ... etc.
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

filebase = 'nukecc_skim_data'

hdf5file = 'nukecc_fuel.hdf5'
if len(sys.argv) > 1:
    hdf5file = sys.argv[1]

# "pixel" size of data images
#  here - W corresponds to MINERvA Z, and H correpsonds to the view axis
IMGW = 50
IMGH = 50


def get_data_from_file(filename):
    print("...loading data")
    targs = []
    data = []
    eventids = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            elems = line.split()
            targs.append(int(elems[0]))
            eventid = elems[1] + elems[2].zfill(4) + elems[3].zfill(4) \
                      + elems[4].zfill(2)
            eventids.append(eventid)
            rowdat = elems[5:]
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
    eventids = np.asarray(eventids, dtype=np.int64)
    data = np.asarray(data, dtype=np.float32)
    storedat = (data, targs, eventids)
    print("...finished loading")
    return storedat

# look for "filebase"+(_learn/_valid/_test/ - zero or more times)+whatever
filebase = re.compile(r"^%s(_learn|_test|_valid)*.*dat$" % filebase)
files = os.listdir('.')
files = [f for f in files if re.match(filebase, f)]
print(files)

if os.path.exists(hdf5file):
    os.remove(hdf5file)
f = h5py.File(hdf5file, 'w')

data_set = f.create_dataset('hits', (0, 3, IMGW, IMGH),
                            dtype='float32', compression='gzip',
                            maxshape=(None, 3, IMGW, IMGH))
target_set = f.create_dataset('segments', (0,),
                              dtype='uint8', compression='gzip',
                              maxshape=(None,))
events_set = f.create_dataset('eventids', (0,),
                              dtype='uint64', compression='gzip',
                              maxshape=(None,))

# `Fuel.H5PYDataset` allows us to label axes with semantic information;
# we record that in the file using "dimensional scales" (see h5py docs)
data_set.dims[0].label = 'batch'
data_set.dims[1].label = 'view(xuv)'
data_set.dims[2].label = 'height(view-coord)'
data_set.dims[3].label = 'width(z)'
target_set.dims[0].label = 'z-segment'
events_set.dims[0].label = 'run+subrun+gate+slices[0]'

total_examples = 0

for fname in files:
    print("Iterating over file:", fname)
    data, targs, eventids = get_data_from_file(fname)
    examples_in_file = len(targs)
    print(" examples_in_file =", examples_in_file)
    existing_examples = np.shape(f['hits'])[0]
    print(" existing_examples =", existing_examples)
    total_examples = examples_in_file + existing_examples
    print(" resize =", total_examples)
    print(" idx slice = %d:%d" % (existing_examples, total_examples))
    f['hits'].resize(total_examples, axis=0)
    f['segments'].resize(total_examples, axis=0)
    f['eventids'].resize(total_examples, axis=0)
    f['hits'][existing_examples: total_examples] = data
    f['segments'][existing_examples: total_examples] = targs
    f['eventids'][existing_examples: total_examples] = eventids

# TODO: investiage the "reference" stuff so we can pluck validation
# and testing events evenly from the sample
final_train_index = int(total_examples * 0.8)
final_valid_index = int(total_examples * 0.9)

split_dict = {
    'train': {'hits': (0, final_train_index),
              'segments': (0, final_train_index),
              'eventids': (0, final_train_index)
          },
    'valid': {'hits': (final_train_index, final_valid_index),
              'segments': (final_train_index, final_valid_index),
              'eventids': (final_train_index, final_valid_index)
          },
    'test': {'hits': (final_valid_index, total_examples),
             'segments': (final_valid_index, total_examples),
             'eventids': (final_valid_index, total_examples)
         }
    }
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.close()
