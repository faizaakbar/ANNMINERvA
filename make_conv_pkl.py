#!/usr/bin/env python
"""
python make_conv_pkl.py <target number>

This script looks for three files like:

    skim_data_learn_target0.dat
    skim_data_test_target0.dat
    skim_data_valid_target0.dat

where "0" is the "target number" provided as an argument.

This script further assumes the files were produced using:
    Skimmer_theanoconvnet.cxx
This means that the entries are presented in order of monotonic increasing z
position, all for one bucket, then resetting to the lowest z and spanning
across the detector for the next bucket (1), and so on for the next (2), etc.
up until the number of buckets.

Because there are twice as many X as U or V planes in the target and tracker
regions, we will add a zero in addition to the hit value read for each U and V.
"""
from __future__ import print_function
import cPickle
import numpy as np
import sys
import os
import gzip

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
            hitsX = np.asarray(hitsX, dtype=np.float32).reshape(50, 50)
            hitsU = np.asarray(hitsU, dtype=np.float32).reshape(50, 50)
            hitsV = np.asarray(hitsV, dtype=np.float32).reshape(50, 50)
            energies = np.asarray([hitsX, hitsU, hitsV])
            data.append(energies)

    targs = np.asarray(targs, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    storedat = (data, targs)  # no need to zip, just store as a tuple
    return storedat


fileroots = ['skim_data_learn_target',
             'skim_data_test_target',
             'skim_data_valid_target']
final_data = []

for filer in fileroots:
    filen = filer + targnum + '.dat'
    dtuple = get_data_from_file(filen)
    final_data.append(dtuple)

print("Loaded all data... attempting to pickle it...")

filepkl = 'skim_data_convnet_target' + targnum + '.pkl.gz'

if os.path.exists(filepkl):
    os.remove(filepkl)

storef = gzip.open(filepkl, 'wb')
cPickle.dump(final_data, storef, protocol=cPickle.HIGHEST_PROTOCOL)
storef.close()
