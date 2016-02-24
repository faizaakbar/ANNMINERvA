from __future__ import print_function
import pylab
import sys

filename = sys.argv[1]
filebase = filename.split('.')[0]

arr = pylab.load(filename)

zsegs = [0, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10]
desc = ['upstream of target 1',
        'target 1',
        'between 1 and 2',
        'target 2',
        'between 2 and 3',
        'target 3',
        'between 3 and 4',
        'target 4',
        'between 4 and 5',
        'target 5',
        'downstream of target 5']
zdesc = dict(zip(zsegs, desc))

fig = pylab.figure(figsize=(15, 15))
gs = pylab.GridSpec(4, 3)
for i, v in enumerate(zsegs):
    ax = pylab.subplot(gs[i])
    ax.set_autoscaley_on(False)
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel('fraction of events')
    ax.set_title('true ' + str(v) + ':' + zdesc[v], loc='right')
    ax.plot(pylab.arange(len(zsegs)), arr[v]/pylab.sum(arr[v]))
figname = filebase + '.pdf'
pylab.savefig(figname)
pylab.close()
