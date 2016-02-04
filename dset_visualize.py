#!/usr/bin/env python
"""
Usage:
    python dset_visualize.py [opt: max # of evts, def==10]
"""
import gzip
import cPickle
import pylab
import sys

max_evts = 10
evt_plotted = 0

if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

if len(sys.argv) > 1:
    max_evts = int(sys.argv[1])

f = gzip.open('skim_data_convnet_target0.pkl.gz', 'rb')
learn_data, test_data, valid_data = cPickle.load(f)
f.close()

for counter, evt in enumerate(valid_data[0]):
    if evt_plotted > max_evts:
        break
    targ = valid_data[1][counter]
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
