#!/usr/bin/env python
"""
Usage:
    python dset_visualize.py [opt: max # of evts, def==10]
"""
import pylab
import sys
import h5py

max_evts = 10
evt_plotted = 0

if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

if len(sys.argv) > 1:
    max_evts = int(sys.argv[1])

f = h5py.File('./nukecc_convdata_fuel.hdf5', 'r')
valid_data = pylab.zeros(pylab.shape(f['hits']), dtype='f')
valid_labels = pylab.zeros(pylab.shape(f['segments']), dtype='f')
f['hits'].read_direct(valid_data)
f['segments'].read_direct(valid_labels)
f.close()

for counter, evt in enumerate(valid_data):
    if evt_plotted > max_evts:
        break
    targ = valid_labels[counter]
    fig = pylab.figure(figsize=(9, 3))
    gs = pylab.GridSpec(1, 3)
    for i in range(3):
        ax = pylab.subplot(gs[i])
        ax.axis('off')
        ax.imshow(evt[i])
    figname = 'evt_%d_targ_%d.pdf' % (counter, targ)
    pylab.savefig(figname)
    pylab.close()
    evt_plotted += 1
