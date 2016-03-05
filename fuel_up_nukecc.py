#!/usr/bin/env python
"""
Usage:
    python fuel_up_nukecc.py -b 'base name' -o 'output'

The default output name is 'nukecc_fuel.hdf5'. This script expects data layout
like:
    <zsegment> <run> <subrun> <gate> <slice[0]> X:[E] U:[E] ... etc.
"""
from __future__ import print_function
import numpy as np
import os
import re

import h5py

from fuel.datasets.hdf5 import H5PYDataset
from plane_codes import build_indexed_codes


def get_data_from_file(filename, imgh, imgw):
    print("...loading data")
    targs = []
    zs = []
    planeids = []
    eventids = []
    data = []
    icodes = build_indexed_codes()
    # format:
    # 0   1   2   3   4   5   6   7
    # seg z   pln run sub gt  slc data...

    with open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            elems = line.split()
            targs.append(int(elems[0]))
            zs.append(float(elems[1]))
            rawid = int(elems[2])
            planeid = icodes[rawid]
            planeids.append(planeid)
            eventid = elems[3] + elems[4].zfill(4) + elems[5].zfill(4) \
                + elems[6].zfill(2)
            eventids.append(eventid)
            rowdat = elems[7:]
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
            hitsX = np.asarray(hitsX, dtype=np.float32).reshape(imgh, imgw)
            hitsU = np.asarray(hitsU, dtype=np.float32).reshape(imgh, imgw)
            hitsV = np.asarray(hitsV, dtype=np.float32).reshape(imgh, imgw)
            energies = np.asarray([hitsX, hitsU, hitsV])
            data.append(energies)

    targs = np.asarray(targs, dtype=np.uint8)
    zs = np.asarray(zs, dtype=np.float32)
    planeids = np.asarray(planeids, dtype=np.uint16)
    eventids = np.asarray(eventids, dtype=np.uint64)
    data = np.asarray(data, dtype=np.float32)
    storedat = (data, targs, zs, planeids, eventids)
    print("...finished loading")
    return storedat


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-b', '--basename', default='nukecc_skim_me1Bmc',
                      help='Input files base name', metavar='BASE_NAME',
                      dest='filebase')
    parser.add_option('-o', '--output', default='nukecc_fuel.hdf5',
                      help='Output filename', metavar='OUTPUT_NAME',
                      dest='hdf5file')
    parser.add_option('-t', '--height', default=50, type='int',
                      help='Image height', metavar='IMG_HEIGHT',
                      dest='imgh')
    parser.add_option('-w', '--width', default=50, type='int',
                      help='Image width', metavar='IMG_WIDTH',
                      dest='imgw')
    (options, args) = parser.parse_args()

    filebase = options.filebase
    hdf5file = options.hdf5file

    # "pixel" size of data images
    #  here - W corresponds to MINERvA Z, and H correpsonds to the view axis
    imgh = options.imgh
    imgw = options.imgw

    # look for "filebase"+(_learn/_valid/_test/ - zero or more times)+whatever
    filebase = re.compile(r"^%s(_learn|_test|_valid)*.*dat$" % filebase)
    files = os.listdir('.')
    files = [f for f in files if re.match(filebase, f)]
    print(files)

    if os.path.exists(hdf5file):
        os.remove(hdf5file)
    f = h5py.File(hdf5file, 'w')

    data_set = f.create_dataset('hits', (0, 3, imgh, imgw),
                                dtype='float32', compression='gzip',
                                maxshape=(None, 3, imgh, imgw))
    target_set = f.create_dataset('segments', (0,),
                                  dtype='uint8', compression='gzip',
                                  maxshape=(None,))
    zs_set = f.create_dataset('zs', (0,),
                              dtype='float32', compression='gzip',
                              maxshape=(None,))
    plane_set = f.create_dataset('planecodes', (0,),
                                 dtype='uint16', compression='gzip',
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
    zs_set.dims[0].label = 'z'
    plane_set.dims[0].label = 'plane-id-code'
    events_set.dims[0].label = 'run+subrun+gate+slices[0]'

    total_examples = 0

    for fname in files:
        print("Iterating over file:", fname)
        data, targs, zs, planecodes, eventids = \
            get_data_from_file(fname, imgh, imgw)
        examples_in_file = len(targs)
        print(" examples_in_file =", examples_in_file)
        existing_examples = np.shape(f['hits'])[0]
        print(" existing_examples =", existing_examples)
        total_examples = examples_in_file + existing_examples
        print(" resize =", total_examples)
        print(" idx slice = %d:%d" % (existing_examples, total_examples))
        f['hits'].resize(total_examples, axis=0)
        f['segments'].resize(total_examples, axis=0)
        f['zs'].resize(total_examples, axis=0)
        f['planecodes'].resize(total_examples, axis=0)
        f['eventids'].resize(total_examples, axis=0)
        f['hits'][existing_examples: total_examples] = data
        f['segments'][existing_examples: total_examples] = targs
        f['zs'][existing_examples: total_examples] = zs
        f['planecodes'][existing_examples: total_examples] = planecodes
        f['eventids'][existing_examples: total_examples] = eventids

    # TODO: investiage the "reference" stuff so we can pluck validation
    # and testing events evenly from the sample
    final_train_index = int(total_examples * 0.83)
    final_valid_index = int(total_examples * 0.93)

    split_dict = {
        'train': {'hits': (0, final_train_index),
                  'segments': (0, final_train_index),
                  'zs': (0, final_train_index),
                  'planecodes': (0, final_train_index),
                  'eventids': (0, final_train_index)
                  },
        'valid': {'hits': (final_train_index, final_valid_index),
                  'segments': (final_train_index, final_valid_index),
                  'zs': (final_train_index, final_valid_index),
                  'planecodes': (final_train_index, final_valid_index),
                  'eventids': (final_train_index, final_valid_index)
                  },
        'test': {'hits': (final_valid_index, total_examples),
                 'segments': (final_valid_index, total_examples),
                 'zs': (final_valid_index, total_examples),
                 'planecodes': (final_valid_index, total_examples),
                 'eventids': (final_valid_index, total_examples)
                 }
    }
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.close()
