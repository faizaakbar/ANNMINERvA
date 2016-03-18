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


def compute_target_padding():
    """
    When adding padding, we traverse 8 planes before reaching target1,
    then 8 planes before reaching target 2, then 8 planes before reaching
    target 3, which we count as 2 planes thick, then ? planes before
    reaching the water target, which we regard as ? planes thick, then
    16 - ? planes before reaching target 4, then 4 planes before reaching
    target 5.

    return tuple of lists - first list is the locations in minerva plane
    occurence index (start with zero, count up as we go along) where we
    should skip two planes, and the second is the list of locations
    where we should skip four

    note in the steps below we are always traveling in steps of groups
    of four - uxvx - so we will only count _x_ planes we have _traversed_
    while building these steps so when we loop and insert padding we have
    moved the right amount of space through the detector (because u and v
    are encoded "in step" with x, with one layer of padding already inserted
    between each u or v to account for the sparsity in those views.
    """
    base_steps = 1      # insert _after_ traversing n cols, etc.
    target1_steps = (8 // 2) + base_steps
    target2_steps = (8 // 2) + target1_steps
    target3_steps = (8 // 2) + target2_steps
    water_steps = (8 // 2) + target3_steps
    target4_steps = (8 // 2) + water_steps
    target5_steps = (4 // 2) + target4_steps
    two_breaks = [target1_steps, target2_steps,
                  target4_steps, target5_steps]
    four_breaks = [target3_steps, water_steps]
    return two_breaks, four_breaks


def get_total_target_padding():
    """
    get the sum of all the spaces following the "breaks" where targets sit
    """
    two_breaks, four_breaks = compute_target_padding()
    target_padding = 2 * len(two_breaks) + 4 * len(four_breaks)
    water_padding = 0
    npadcol = target_padding + water_padding
    return npadcol


def pad_for_targets(imgw, imgh, hitsX, hitsU, hitsV):
    two_breaks, four_breaks = compute_target_padding()
    imgh_padding = get_total_target_padding()
    tempX = np.zeros(imgw * (imgh_padding + imgh),
                     dtype=np.float32).reshape(
                         imgw, imgh_padding + imgh)
    tempU = np.zeros(imgw * (imgh_padding + imgh),
                     dtype=np.float32).reshape(
                         imgw, imgh_padding + imgh)
    tempV = np.zeros(imgw * (imgh_padding + imgh),
                     dtype=np.float32).reshape(
                         imgw, imgh_padding + imgh)
    shifted_column_counter = 0

    def col_copy(frm, to):
        tempX[:, to] = hitsX[:, frm]
        tempU[:, to] = hitsU[:, frm]
        tempV[:, to] = hitsV[:, frm]

    for i in range(imgh):
        j = i + 1
        if j in two_breaks:
            shifted_column_counter += 2
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
        elif j in four_breaks:
            shifted_column_counter += 4
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
        else:
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
    return tempX, tempU, tempV


def get_data_from_file(filename, imgw, imgh, add_target_padding=False):
    """
    """
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
            hitsX = np.asarray(hitsX, dtype=np.float32).reshape(imgw, imgh)
            hitsU = np.asarray(hitsU, dtype=np.float32).reshape(imgw, imgh)
            hitsV = np.asarray(hitsV, dtype=np.float32).reshape(imgw, imgh)
            if add_target_padding:
                hitsX, hitsU, hitsV = pad_for_targets(imgw, imgh,
                                                      hitsX, hitsU, hitsV)
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


def make_hdf5_file(imgw, imgh, filebase, hdf5file, add_target_padding=False):
    """
    imgw, imgh - ints that specify the image size for `reshape`
    filebase - pattern used for files to match into the output
    hdf5file - name of the output file

    note that imgw traverses the "y" direction and imgh traverses the "x"
    direction in the classic mathematician's graph

    note that filebase is a pattern - if multiple files match
    the pattern, then multiple files will be included in the
    single output file
    """

    # look for "filebase"+(_learn/_valid/_test/ - zero or more times)+whatever
    filebase = re.compile(r"^%s(_learn|_test|_valid)*.*dat$" % filebase)
    files = os.listdir('.')
    files = [f for f in files if re.match(filebase, f)]
    print(files)

    if os.path.exists(hdf5file):
        os.remove(hdf5file)
    f = h5py.File(hdf5file, 'w')

    imgh_padding = 0
    if add_target_padding:
        imgh_padding = get_total_target_padding()

    data_set = f.create_dataset('hits', (0, 3, imgw, imgh + imgh_padding),
                                dtype='float32', compression='gzip',
                                maxshape=(None, 3, imgw, imgh + imgh_padding))
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
            get_data_from_file(fname, imgw, imgh, add_target_padding)
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
    parser.add_option('-p', '--padded_targets', default=False,
                      dest='padding', help='Include target padding',
                      metavar='TARG_PAD', action='store_true')
    (options, args) = parser.parse_args()

    filebase = options.filebase
    hdf5file = options.hdf5file

    # imgw, imgh - "pixel" size of data images
    #  here - H corresponds to MINERvA Z, and W correpsonds to the view axis
    make_hdf5_file(options.imgw, options.imgh,
                   filebase, hdf5file, options.padding)
