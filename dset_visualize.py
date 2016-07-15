#!/usr/bin/env python
"""
Usage:
    python dset_visualize.py [file name] [opt: max # of evts, def==10]

The default file name is: "./nukecc_fuel.hdf5".

Note, one can, if one wants, when working with `Fuel`d data sets, do:

    from fuel.datasets import H5PYDataset
    train_set = H5PYDataset('./mydat_fuel.hdf5', which_sets=('train',))
    handle = train_set.open()
    nexamp = train_set.num_examples
    data = train_set.get_data(handle, slice(0, nexamp))
    # ...work with the data
    train_set.close(handle)

...but we don't do that here. (Just use h5py to cut any requirement of Fuel
to look at the dsets.)
"""
import pylab
import sys
import h5py

max_evts = 10
evt_plotted = 0

if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

filename = './minerva_fuel.hdf5'
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

have_times = False
# look for x, u, v data hits
try:
    data_x_shp = pylab.shape(f['hits-x'])
    xname = 'hits-x'
except KeyError:
    print("'hits-x' does not exist.")
    data_x_shp = None
try:
    data_u_shp = pylab.shape(f['hits-u'])
    uname = 'hits-u'
except KeyError:
    print("'hits-u' does not exist.")
    data_u_shp = None
try:
    data_v_shp = pylab.shape(f['hits-v'])
    vname = 'hits-v'
except KeyError:
    print("'hits-v' does not exist.")
    data_v_shp = None
# maybe we have times instead...
if data_x_shp is None:
    try:
        data_x_shp = pylab.shape(f['times-x'])
        have_times = True
        xname = 'times-x'
    except KeyError:
        print("'times-x' does not exist.")
        data_x_shp = None
if data_u_shp is None:
    try:
        data_u_shp = pylab.shape(f['times-u'])
        have_times = True
        uname = 'times-u'
    except KeyError:
        print("'times-u' does not exist.")
        data_u_shp = None
if data_v_shp is None:
    try:
        data_v_shp = pylab.shape(f['times-v'])
        have_times = True
        vname = 'times-v'
    except KeyError:
        print("'times-v' does not exist.")
        data_v_shp = None

# if we have hits, get them, else set those containers to None
if data_x_shp is not None:
    data_x_shp = (max_evts, data_x_shp[1], data_x_shp[2], data_x_shp[3])
    data_x = pylab.zeros(data_x_shp, dtype='f')
    data_x = f[xname][:max_evts]
else:
    data_x = None
if data_u_shp is not None:
    data_u_shp = (max_evts, data_u_shp[1], data_u_shp[2], data_u_shp[3])
    data_u = pylab.zeros(data_u_shp, dtype='f')
    data_u = f[uname][:max_evts]
else:
    data_u = None
if data_v_shp is not None:
    data_v_shp = (max_evts, data_v_shp[1], data_v_shp[2], data_v_shp[3])
    data_v = pylab.zeros(data_v_shp, dtype='f')
    data_v = f[vname][:max_evts]
else:
    data_v = None

labels_shp = (max_evts,)
evtids_shp = (max_evts,)
labels = pylab.zeros(labels_shp, dtype='f')
evtids = pylab.zeros(evtids_shp, dtype='uint64')
evtids = f['eventids'][:max_evts]
try:
    labels = f['segments'][:max_evts]
    pcodes = f['planecodes'][:max_evts]
    zs = f['zs'][:max_evts]
except KeyError:
    labels_shp = None

f.close()

colorbar_tile = 'scaled energy'
if have_times:
    colorbar_tile = 'scaled times'

for counter, evtid in enumerate(evtids):
    if evt_plotted > max_evts:
        break
    run, subrun, gate, phys_evt = decode_eventid(evtid)
    if labels_shp is not None:
        targ = labels[counter]
        pcode = pcodes[counter]
        zz = zs[counter]
        pstring = '{} - {} - {} - {}: tgt: {:02d}; plncd {:03d}; z {}'.format(
            run, subrun, gate, phys_evt, targ, pcode, zz)
    else:
        pstring = '{} - {} - {} - {}'.format(
            run, subrun, gate, phys_evt)
    print(pstring)
    evt = []
    titles = []
    if data_x is not None:
        evt.append(data_x[counter])
        titles.append('x view')
    if data_u is not None:
        evt.append(data_u[counter])
        titles.append('u view')
    if data_v is not None:
        evt.append(data_v[counter])
        titles.append('v view')
    fig = pylab.figure(figsize=(9, 3))
    gs = pylab.GridSpec(1, len(evt))
    # print np.where(evt == np.max(evt))
    # print np.max(evt)
    for i in range(len(evt)):
        ax = pylab.subplot(gs[i])
        ax.axis('on')
        ax.xaxis.set_major_locator(pylab.NullLocator())
        ax.yaxis.set_major_locator(pylab.NullLocator())
        # images are normalized such the max e-dep has val 1, independent
        # of view, so set vmin, vmax here to keep matplotlib from
        # normalizing each view on its own
        minv = 0
        cmap = 'jet'
        if have_times:
            minv = -1
            cmap = 'bwr'
        im = ax.imshow(evt[i][0], cmap=pylab.get_cmap(cmap),
                       interpolation='nearest', vmin=minv, vmax=1)
        cbar = pylab.colorbar(im, fraction=0.04)
        cbar.set_label(colorbar_tile, size=9)
        cbar.ax.tick_params(labelsize=6)
        pylab.title(titles[i], fontsize=12)
        pylab.xlabel("plane", fontsize=10)
        pylab.ylabel("strip", fontsize=10)
    if labels_shp is not None:
        figname = 'evt_%s_%s_%s_%s_targ_%d_pcode_%d.pdf' % \
                  (run, subrun, gate, phys_evt, targ, pcode)
    else:
        figname = 'evt_%s_%s_%s_%s.pdf' % \
                  (run, subrun, gate, phys_evt)
    pylab.savefig(figname)
    pylab.close()
    evt_plotted += 1
