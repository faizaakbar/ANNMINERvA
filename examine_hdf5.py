#!/usr/bin/env python
"""
Usage:
    python examine_hdf5.py <hdf5 file>

Note: you _must_ have NumPy and h5py installed.
"""
from __future__ import print_function
import numpy as np
import h5py
import sys

if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

filename = sys.argv[1]
f = h5py.File(filename, 'r')

print("\ngroups/datasets in the hdf5 file")
print("-----------------------")
for group in f:
    for dataset in f[group]:
        print('{:>10} / {:<10}: {:>8}: shape = {}'.format(
            group,
            dataset,
            np.dtype(f[group][dataset]),
            np.shape(f[group][dataset])
        ))
    
print("\nto examine data, type things like:")
print("----------------------------------")
print("f['mygroup']['mydataset'][:5]     - 1st 5 items")
print("f['mygroup']['mydataset'][0][:5]  - 1st 5 items in 2nd idx of 1st item")
print(" - etc. - basically you may use numpy slice notation")

f.close()
