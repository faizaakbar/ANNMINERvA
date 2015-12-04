#!/usr/bin/env python
import gzip
import cPickle
import pylab

f = gzip.open('skim_data_convnet_target0.pkl.gz', 'rb')
learn_data, test_data, valid_data = cPickle.load(f)
f.close()

for counter, evt in enumerate(learn_data[0]):
    targ = learn_data[1][counter]
    fig = pylab.figure(figsize=(9, 3))
    gs = pylab.GridSpec(1, 3)
    for i in range(3):
        ax = pylab.subplot(gs[i])
        ax.axis('off')
        ax.imshow(evt[i].reshape(22, 50))
    figname = 'evt_%d_targ_%d.pdf' % (counter, targ)
    pylab.savefig(figname)
