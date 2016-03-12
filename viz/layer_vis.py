#!/usr/bin/env python
"""
Usage:
    python layer_vis.py <event id>
"""
from __future__ import print_function
import sys
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def decode_eventid(eventid):
    """
    assume encoding from fuel_up_nukecc.py, etc.
    """
    eventid = str(eventid)
    phys_evt = eventid[-2:]
    eventid = eventid[:-2]
    gate = eventid[-4:]
    eventid = eventid[:-4]
    subrun = eventid[-4:]
    eventid = eventid[:-4]
    run = eventid
    return (run, subrun, gate, phys_evt)


def make_vis(eventid):
    files = os.listdir('.')
    genrlpat = re.compile(r"^vis_[0-9]+_(conv|pool)_[0-9]_[0-9]*_%s\.npy$" %
                          eventid)
    files = [f for f in files if re.match(genrlpat, f)]
    nums = re.findall(r'[0-9]+', files[0])
    target = nums[0]
    conv1pat = re.compile(r"^vis_[0-9]+_conv_1_[0-9]*_%s\.npy$" % eventid)
    pool1pat = re.compile(r"^vis_[0-9]+_pool_1_[0-9]*_%s\.npy$" % eventid)
    conv2pat = re.compile(r"^vis_[0-9]+_conv_2_[0-9]*_%s\.npy$" % eventid)
    pool2pat = re.compile(r"^vis_[0-9]+_pool_2_[0-9]*_%s\.npy$" % eventid)
    # should be just one file in each list
    conv1file = [f for f in files if re.match(conv1pat, f)][0]
    pool1file = [f for f in files if re.match(pool1pat, f)][0]
    conv2file = [f for f in files if re.match(conv2pat, f)][0]
    pool2file = [f for f in files if re.match(pool2pat, f)][0]
    print(conv1file)
    print(pool1file)
    print(conv2file)
    print(pool2file)

    run, subrun, gate, pe = decode_eventid(eventid)

    conv1 = np.load(conv1file)
    pool1 = np.load(pool1file)
    conv2 = np.load(conv2file)
    pool2 = np.load(pool2file)

    views = ['x', 'u', 'v']

    # conv1 shapes will be ~ (3, 1, 1, 32, 48, 48)
    #                       (x/u/v, 1, 1, filt#, imgw, imgh)
    # pool1 shapes will be ~ (3, 1, 1, 32, 24, 24)
    #                       (x/u/v, 1, 1, filt#, imgw, imgh)
    # conv2 shapes will be ~ (3, 1, 1, 32, 22, 22)
    #                       (x/u/v, 1, 1, filt#, imgw, imgh)
    # pool2 shapes will be ~ (3, 1, 1, 32, 11, 11)
    #                       (x/u/v, 1, 1, filt#, imgw, imgh)

    for v, view in enumerate(views):
        # conv
        for i, layer in enumerate([conv1, conv2]):
            j = str(i + 1)
            fig = plt.figure(figsize=(18, 30))
            gs = plt.GridSpec(8, 4)
            for i in range(32):
                ax = plt.subplot(gs[i])
                ax.imshow(layer[v][0][0][i])
            figname = 'activations_' + target + '_' + view + j + '_conv_' + \
                run + '_' + subrun + '_' + gate + '_' + pe + '.pdf'
            print(' plotting into {}'.format(figname))
            plt.savefig(figname)
            plt.close()

        # pool
        for i, layer in enumerate([pool1, pool2]):
            j = str(i + 1)
            fig = plt.figure(figsize=(18, 30))
            gs = plt.GridSpec(8, 4)
            for i in range(32):
                ax = plt.subplot(gs[i])
                ax.imshow(layer[v][0][0][i])
            figname = 'activations_' + target + '_' + view + j + '_pool_' + \
                run + '_' + subrun + '_' + gate + '_' + pe + '.pdf'
            print(' plotting into {}'.format(figname))
            plt.savefig(figname)
            plt.close()


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        print(__doc__)
        sys.exit(1)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    eventid = sys.argv[1]
    make_vis(eventid)
