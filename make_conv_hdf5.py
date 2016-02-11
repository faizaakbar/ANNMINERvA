#!/usr/bin/env python
"""
python make_conv_hdf5.py <target number>

This script looks for three files like:

    skim_data_learn_target0.dat
    skim_data_test_target0.dat
    skim_data_valid_target0.dat

where "0" is the "target number" provided as an argument.

This script further assumes the files were produced using:

    Skimmer_theanoconvnet.cxx

or

    Skimmer_zsegments.cxx

This means that the entries are presented in order of monotonic increasing z
position, all for one bucket, then resetting to the lowest z and spanning
across the detector for the next bucket (1), and so on for the next (2), etc.
up until the number of buckets.

Because there are twice as many X as U or V planes in the target and tracker
regions, we will add a zero in addition to the hit value read for each U and V.

TODO: work on data files in chunks - they're getting to be too big to just
read it all into memory...
"""
from __future__ import print_function
import h5py
import numpy as np
import sys
import os

if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

if not len(sys.argv) == 2:
    print('The target number argument is mandatory.')
    print(__doc__)
    sys.exit(1)

targnum = sys.argv[1]


def get_data_from_file(filename):
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
            hitsX = np.asarray(hitsX, dtype=np.float32).reshape(22, 50)
            hitsU = np.asarray(hitsU, dtype=np.float32).reshape(22, 50)
            hitsV = np.asarray(hitsV, dtype=np.float32).reshape(22, 50)
            energies = np.asarray([hitsX, hitsU, hitsV])
            data.append(energies)

    targs = np.asarray(targs, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    storedat = (data, targs)  # no need to zip, just store as a tuple
    return storedat


fileroots = ['skim_data_learn_target',
             'skim_data_test_target',
             'skim_data_valid_target']

hdf5file = 'skim_data_convnet.hdf5'
if os.path.exists(hdf5file):
    os.remove(hdf5file)
f = h5py.File(hdf5file, 'w')

for filer in fileroots:
    filen = filer + targnum + '.dat'
    data, targs = get_data_from_file(filen)
    data_label = filer.split('_')[2]
    grp = f.create_group(data_label)
    data_set = grp.create_dataset('hits', np.shape(data),
                                  dtype='f', compression='gzip')
    target_set = grp.create_dataset('segments', np.shape(targs),
                                    dtype='f', compression='gzip')
    data_set[...] = data
    target_set[...] = targs

f.close()
