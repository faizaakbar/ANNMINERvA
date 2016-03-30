#!/usr/bin/env python
"""
Usage:
    python dset_visualize.py [file name] [opt: max # of evts, def==10]

The default file name is: "./nukecc_fuel.hdf5".
"""
import pylab
# import numpy as np
import sys
import h5py
# Note, one can, if one wants, when working with `Fuel`d data sets, do:
# from fuel.datasets import H5PYDataset
# train_set = H5PYDataset('./nukecc_convdata_fuel.hdf5', which_sets=('train',))
# handle = train_set.open()
# nexamp = train_set.num_examples
# data = train_set.get_data(handle, slice(0, nexamp))
# ...work with the data
# train_set.close(handle)
max_evts = 10
evt_plotted = 0

if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

filename = './nukecc_fuel.hdf5'
if len(sys.argv) > 1:
    filename = sys.argv[1]
if len(sys.argv) > 2:
    max_evts = int(sys.argv[2])


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

f = h5py.File(filename, 'r')
try:
    data_x = pylab.zeros(pylab.shape(f['hits-x']), dtype='f')
except KeyError:
    print("'hits-x' does not exist.")
    data_x = None
try:
    data_u = pylab.zeros(pylab.shape(f['hits-u']), dtype='f')
except KeyError:
    print("'hits-u' does not exist.")
    data_u = None
try:
    data_v = pylab.zeros(pylab.shape(f['hits-v']), dtype='f')
except KeyError:
    print("'hits-v' does not exist.")
    data_v = None
labels = pylab.zeros(pylab.shape(f['segments']), dtype='f')
evtids = pylab.zeros(pylab.shape(f['eventids']), dtype='uint64')
if data_x is not None:
    f['hits-x'].read_direct(data_x)
if data_u is not None:
    f['hits-u'].read_direct(data_u)
if data_v is not None:
    f['hits-v'].read_direct(data_v)
f['segments'].read_direct(labels)
f['eventids'].read_direct(evtids)
f.close()

for counter, evtid in enumerate(evtids):
    if evt_plotted > max_evts:
        break
    run, subrun, gate, phys_evt = decode_eventid(evtid)
    print('{} - {} - {} - {}'.format(run, subrun, gate, phys_evt))
    targ = labels[counter]
    evt = []
    if data_x is not None:
        evt.append(data_x[counter])
    if data_u is not None:
        evt.append(data_u[counter])
    if data_v is not None:
        evt.append(data_v[counter])
    fig = pylab.figure(figsize=(9, 3))
    gs = pylab.GridSpec(1, len(evt))
    # print np.where(evt == np.max(evt))
    # print np.max(evt)
    for i in range(len(evt)):
        ax = pylab.subplot(gs[i])
        ax.axis('off')
        # images are normalized such the max e-dep has val 1, independent
        # of view, so set vmin, vmax here to keep matplotlib from
        # normalizing each view on its own
        ax.imshow(evt[i][0], cmap=pylab.get_cmap('jet'),
                  interpolation='nearest', vmin=0, vmax=1)
    figname = 'evt_%s_%s_%s_%s_targ_%d.pdf' % \
        (run, subrun, gate, phys_evt, targ)
    pylab.savefig(figname)
    pylab.close()
    evt_plotted += 1
