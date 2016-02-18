#!/usr/bin/env python
"""
python make_conv_hdf5.py <file key> [output name - optional]

The default output name is 'skim_data_convnet.hdf5'

This script looks for three files like:

    skim_data_learn_<file key><#>.dat
    skim_data_test_<file key><#>.dat
    skim_data_valid_<file key><#>.dat

This script further assumes the files were produced using one of:

    Skimmer_theanoconvnet.cxx
    Skimmer_zsegments.cxx
    Skimmer_chunked_zsegments.cxx

This means that the entries are presented in order of monotonic increasing z
position, all for one bucket, then resetting to the lowest z and spanning
across the detector for the next bucket (1), and so on for the next (2), etc.
up until the number of buckets.

Because there are twice as many X as U or V planes in the target and tracker
regions, we will add a zero in addition to the hit value read for each U and V.
"""
from __future__ import print_function
import h5py
import numpy as np
import sys
import os
import re

if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

if len(sys.argv) < 2:
    print('The filekey argument is mandatory.')
    print(__doc__)
    sys.exit(1)

filekey = sys.argv[1]

hdf5file = 'skim_data_convnet.hdf5'
if len(sys.argv) > 2:
    hdf5file = sys.argv[2]

# "pixel" size of data images
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
            hitsX = np.asarray(hitsX, dtype=np.float32).reshape(IMGW, IMGH)
            hitsU = np.asarray(hitsU, dtype=np.float32).reshape(IMGW, IMGH)
            hitsV = np.asarray(hitsV, dtype=np.float32).reshape(IMGW, IMGH)
            energies = np.asarray([hitsX, hitsU, hitsV])
            data.append(energies)

    targs = np.asarray(targs, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    storedat = (data, targs)  # no need to zip, just store as a tuple
    print("...finished loading")
    return storedat

# get all the files and organize into a dictionary by category
filebase = re.compile(r"^skim_data.*dat$")
files = os.listdir('.')
files = [f for f in files if re.match(filebase, f)]
files = [f for f in files if re.search(filekey, f)]

learn_files = [f for f in files if re.search('learn', f)]
valid_files = [f for f in files if re.search('valid', f)]
test_files = [f for f in files if re.search('test', f)]
files = {'learn': learn_files,
         'valid': valid_files,
         'test': test_files}

if os.path.exists(hdf5file):
    os.remove(hdf5file)
f = h5py.File(hdf5file, 'w')

# set up the hdf5 directory structure - we will resize on the fly
for category in files.iterkeys():
    grp = f.create_group(category)
    data_set = grp.create_dataset('hits', (0, 3, IMGW, IMGH),
                                  dtype='f', compression='gzip',
                                  maxshape=(None, 3, IMGW, IMGH))
    target_set = grp.create_dataset('segments', (0,),
                                    dtype='f', compression='gzip',
                                    maxshape=(None,))

# loop through all the files and add them to resized blocks in the hdf5
for cat, fils in files.iteritems():
    for fil in fils:
        hits = cat + '/hits'
        segs = cat + '/segments'
        print("Iterating over file:", fil)
        data, targs = get_data_from_file(fil)
        n_entries = len(targs)
        print(" n_entries =", n_entries)
        existing_sz = np.shape(f[hits])[0]
        print(" existing_sz =", existing_sz)
        print(" resize =", n_entries + existing_sz)
        print(" idx slice = %d:%d" % (existing_sz, n_entries + existing_sz))
        f[hits].resize(n_entries + existing_sz, axis=0)
        f[segs].resize(n_entries + existing_sz, axis=0)
        f[hits][existing_sz: existing_sz + n_entries] = data
        f[segs][existing_sz: existing_sz + n_entries] = targs

f.close()
