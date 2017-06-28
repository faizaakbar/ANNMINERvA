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

print("\ngroups in the hdf5 file")
print("-----------------------")
for name in f:
    print('{:>12}: {:>8}: shape = {}'.format(
        name, np.dtype(f[name]), np.shape(f[name])
    ))
    
print("\nto examine data, type things like:")
print("----------------------------------")
print("f['mygroup'][:5]     - 1st 5 items")
print("f['mygroup'][0][:5]  - 1st 5 items in second index of first item")
print(" - etc. - basically you may use numpy slice notation")

f.close()
