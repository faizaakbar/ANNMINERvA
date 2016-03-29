#!/usr/bin/env python
"""
Usage:
    python fuel_up_nukecc.py -b 'base name' -o 'output'

The default output name is 'nukecc_fuel.hdf5'. This script expects data layout
like:
    # 0   1   2   3   4   5     6     7
    # seg z   pln run sub gate  slice data (X:[E] U:[E] ... etc.)...
"""
from __future__ import print_function
import numpy as np
import sys
import os
import re
from collections import OrderedDict

import h5py

from fuel.datasets.hdf5 import H5PYDataset
from plane_codes import build_indexed_codes


def compute_target_padding():
    """
    When adding padding, we traverse 8 planes before reaching target1,
    then 8 planes before reaching target 2, then 8 planes before reaching
    target 3, which we count as 2 planes thick, then 8 planes before
    reaching the water target, which we regard as 6 modules thick, then
    8 planes before reaching target 4, then 4 planes before reaching
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
    four_breaks = [target3_steps]
    six_breaks = [water_steps]
    return two_breaks, four_breaks, six_breaks


def get_total_target_padding():
    """
    get the sum of all the spaces following the "breaks" where targets sit
    """
    two_breaks, four_breaks, six_breaks = compute_target_padding()
    target_padding = 2 * len(two_breaks) + 4 * len(four_breaks) + \
        6 * len(six_breaks)
    return target_padding


def get_output_imgh(imgh, add_target_padding=False):
    if add_target_padding:
        imgh += get_total_target_padding()
    return imgh


def pad_for_targets(imgw, imgh, hitsX, hitsU, hitsV):
    two_breaks, four_breaks, six_breaks = compute_target_padding()
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

    def col_copy(frm, to):
        tempX[:, to] = hitsX[:, frm]
        tempU[:, to] = hitsU[:, frm]
        tempV[:, to] = hitsV[:, frm]

    shifted_column_counter = 0
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
        elif j in six_breaks:
            shifted_column_counter += 6
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
        else:
            col_copy(i, shifted_column_counter)
            shifted_column_counter += 1
    return tempX, tempU, tempV


def trim_columns(img, col_up, col_dn):
    """
    keep columns between `col_up` and `col_dn`
    """
    img = img[:, col_up:col_dn]
    return img


def shape_and_flip_image(hits, imgw, imgh):
    hits = np.asarray(hits, dtype=np.float32).reshape(imgw, imgh)
    # we're "upside down" by default in images, flip back just
    # so things look normal - shouldn't affect anything
    hits = hits[::-1, :]
    return hits


def get_data_from_file(filename, imgw, imgh,
                       trim_column_up, trim_column_dn,
                       add_target_padding=False):
    """
    imgw, imgh - specify the size of the raw data image in the file

    trim_column_up - specify if we want to trim the target region of
    the detector off for tracker analysis
    trim_column_dn - specify if we want to cut the downstream part of
    the detector off to speed up target analysis

    NOTE: trim_column_up and trim_column_dn use AFTER PADDING NUMBERS!

    add_target_padding - add in blanks for the targets in the target region?
    """
    print("...loading data")
    targs = []
    zs = []
    planeids = []
    eventids = []
    dataX = []
    dataU = []
    dataV = []
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
            hitsX = shape_and_flip_image(hitsX, imgw, imgh)
            hitsU = shape_and_flip_image(hitsU, imgw, imgh)
            hitsV = shape_and_flip_image(hitsV, imgw, imgh)
            if add_target_padding:
                hitsX, hitsU, hitsV = pad_for_targets(imgw, imgh,
                                                      hitsX, hitsU, hitsV)
            hitsX = trim_columns(hitsX, trim_column_up, trim_column_dn)
            hitsU = trim_columns(hitsU, trim_column_up, trim_column_dn)
            hitsV = trim_columns(hitsV, trim_column_up, trim_column_dn)
            dataX.append(hitsX)
            dataU.append(hitsU)
            dataV.append(hitsV)

    targs = np.asarray(targs, dtype=np.uint8)
    zs = np.asarray(zs, dtype=np.float32)
    planeids = np.asarray(planeids, dtype=np.uint16)
    eventids = np.asarray(eventids, dtype=np.uint64)
    dataX = np.asarray(dataX, dtype=np.float32)
    dataU = np.asarray(dataU, dtype=np.float32)
    dataV = np.asarray(dataV, dtype=np.float32)
    storedat = (dataX, dataU, dataV, targs, zs, planeids, eventids)
    print("...finished loading")
    return storedat


def transform_to_4d_tensor(tensr):
    shpvar = np.shape(tensr)
    shpvar = (shpvar[0], 1, shpvar[1], shpvar[2])
    tensr = np.reshape(tensr, shpvar)
    return tensr


def make_file_list(filebase):
    # look for "filebase"+(_learn/_valid/_test/ - zero or more times)+whatever
    filebase = re.compile(r"^%s(_learn|_test|_valid)*.*dat$" % filebase)
    files = os.listdir('.')
    files = [f for f in files if re.match(filebase, f)]
    print(files)
    return files


def prepare_hdf5_file(hdf5file):
    if os.path.exists(hdf5file):
        os.remove(hdf5file)
    f = h5py.File(hdf5file, 'w')
    return f


def create_view_dset(hdf5file, name, imgw, imgh):
    data_set = hdf5file.create_dataset(name, (0, 1, imgw, imgh),
                                       dtype='float32', compression='gzip',
                                       maxshape=(None, 1, imgw, imgh))
    # `Fuel.H5PYDataset` allows us to label axes with semantic information;
    # we record that in the file using "dimensional scales" (see h5py docs)
    data_set.dims[0].label = 'batch'
    data_set.dims[1].label = 'view(xuv)'
    data_set.dims[2].label = 'height(view-coord)'
    data_set.dims[3].label = 'width(z)'


def create_1d_dset(hdf5file, name, dtype, label):
    data_set = hdf5file.create_dataset(name, (0,),
                                       dtype=dtype, compression='gzip',
                                       maxshape=(None,))
    data_set.dims[0].label = label


def add_split_dict(hdf5file, names, total_examples,
                   train_frac=0.83, valid_frac=0.10):
    # TODO: investiage the "reference" stuff so we can pluck validation
    # and testing events evenly from the sample
    final_train_index = int(total_examples * train_frac)
    final_valid_index = int(total_examples * (train_frac + valid_frac))

    train_dict = {name: (0, final_train_index)
                  for name in names}
    valid_dict = {name: (final_train_index, final_valid_index)
                  for name in names}
    test_dict = {name: (final_valid_index, total_examples)
                 for name in names}
    split_dict = {
        'train': train_dict,
        'valid': valid_dict,
        'test': test_dict
    }
    hdf5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)


def add_data_to_hdf5file(f, dset_names, dset_vals):
        examples_in_file = len(dset_vals[-1])
        print(" examples_in_file =", examples_in_file)
        existing_examples = np.shape(f[dset_names[-1]])[0]
        print(" existing_examples =", existing_examples)
        total_examples = examples_in_file + existing_examples
        print(" resize =", total_examples)
        print(" idx slice = %d:%d" % (existing_examples, total_examples))
        for name in dset_names:
            f[name].resize(total_examples, axis=0)
        for i, data in enumerate(dset_vals):
            f[dset_names[i]][existing_examples: total_examples] = data
        return total_examples


def make_hdf5_file(imgw, imgh, trim_col_up, trim_col_dn,
                   filebase, hdf5file, add_target_padding=False):
    """
    imgw, imgh - ints that specify the image size for `reshape`
    filebase - pattern used for files to match into the output
    hdf5file - name of the output file

    NOTE: trim_col_up and trim_col_dn are for AFTER padding numbers!

    note that imgw traverses the "y" direction and imgh traverses the "x"
    direction in the classic mathematician's graph

    note that filebase is a pattern - if multiple files match
    the pattern, then multiple files will be included in the
    single output file
    """
    print('Making hdf5 file for img-in: {} x {} and out {} x {}-{}'.format(
        imgw, imgh, imgw, trim_col_up, trim_col_dn))

    files = make_file_list(filebase)
    f = prepare_hdf5_file(hdf5file)
    imgh_out = trim_col_dn - trim_col_up

    img_dim = (imgw, imgh_out)
    dset_description = OrderedDict(
        (('hits-x', img_dim),
         ('hits-u', img_dim),
         ('hits-v', img_dim),
         ('segments', ('uint8', 'z-segment')),
         ('zs', ('float32', 'z')),
         ('planecodes', ('uint16', 'plane-id-code')),
         ('eventids', ('uint64', 'run+subrun+gate+slices[0]')))
    )
    dset_names = dset_description.keys()

    create_view_dset(f, dset_names[0], img_dim[0], img_dim[1])
    create_view_dset(f, dset_names[1], img_dim[0], img_dim[1])
    create_view_dset(f, dset_names[2], img_dim[0], img_dim[1])
    create_1d_dset(f, dset_names[3], 'uint8', 'z-segment')
    create_1d_dset(f, dset_names[4], 'float32', 'z')
    create_1d_dset(f, dset_names[5], 'uint16', 'plane-id-code')
    create_1d_dset(f, dset_names[6], 'uint64', 'run+subrun+gate+slices[0]')

    total_examples = 0

    for fname in files:
        print("Iterating over file:", fname)
        dataX, dataU, dataV, targs, zs, planecodes, eventids = \
            get_data_from_file(fname, imgw, imgh, trim_col_up, trim_col_dn,
                               add_target_padding)
        dataX = transform_to_4d_tensor(dataX)
        dataU = transform_to_4d_tensor(dataU)
        dataV = transform_to_4d_tensor(dataV)
        print('data shapes:',
              np.shape(dataX), np.shape(dataU), np.shape(dataV))
        dset_vals = [dataX, dataU, dataV, targs, zs, planecodes, eventids]
        total_examples = add_data_to_hdf5file(f, dset_names, dset_vals)

    add_split_dict(f, dset_names, total_examples)

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
    parser.add_option('-t', '--inp_height', default=94, type='int',
                      help='Image input height', metavar='IMG_HEIGHT',
                      dest='imgh')
    parser.add_option('-w', '--inp_width', default=127, type='int',
                      help='Image input width', metavar='IMG_WIDTH',
                      dest='imgw')
    parser.add_option('-p', '--padded_targets', default=False,
                      dest='padding', help='Include target padding',
                      metavar='TARG_PAD', action='store_true')
    parser.add_option('-u', '--trim_column_up', default=0, type='int',
                      help='Trim column upstream', metavar='TRIM_COL_UP',
                      dest='trim_column_up')
    parser.add_option('-d', '--trim_column_down', default=94, type='int',
                      help='Trim column downstream', metavar='TRIM_COL_DN',
                      dest='trim_column_down')
    parser.add_option('-c', '--check_target_padding', default=False,
                      dest='check_target_padding', help='Check target padding',
                      metavar='CHECK_TARG_PAD', action='store_true')
    (options, args) = parser.parse_args()

    if options.check_target_padding:
        padding = get_total_target_padding()
        print("Total target padding is {} columns.".format(padding))
        sys.exit(0)

    filebase = options.filebase
    hdf5file = options.hdf5file

    # imgw, imgh - "pixel" size of data images
    #  here - H corresponds to MINERvA Z, and W correpsonds to the view axis
    make_hdf5_file(options.imgw, options.imgh,
                   options.trim_column_up, options.trim_column_down,
                   filebase, hdf5file, options.padding)
