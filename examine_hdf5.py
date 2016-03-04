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


def printname(name):
    print(name)

filename = sys.argv[1]
f = h5py.File(filename, 'r')

print("\ngroups in the hdf5 file")
print("-----------------------")
print(f.visit(printname))

print("\nsizes of groups in the hdf5 file")
print("--------------------------------")
for dset in f:
    print(np.shape(f[dset]))

print("\nto examine data, type things like:")
print("----------------------------------")
print("f['mygroup'][:5]     - 1st 5 items")
print("f['mygroup'][0][:5]  - 1st 5 items in second index of first item")
print(" - etc. - basically you may use numpy slice notation")

f.close
