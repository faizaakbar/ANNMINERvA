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
import tensorflow as tf

# max_evts = 10
# evt_plotted = 0

# if '-h' in sys.argv or '--help' in sys.argv:
#     print(__doc__)
#     sys.exit(1)

# filename = './minerva_fuel.hdf5'
# if len(sys.argv) > 1:
#     filename = sys.argv[1]
# if len(sys.argv) > 2:
#     max_evts = int(sys.argv[2])


class MnvDataReader:
    def __init__(self, filename, n_events=10, views=['x', 'u', 'v']):
        self.filename = filename
        self.n_events = n_events
        self.views = views
        self.filetype = filename.split('.')[-1]

        self.hdf5_extensions = ['hdf5', 'h5']
        self.tfr_extensions = ['tfrecord']
        self.all_extensions = self.hdf5_extensions + self.tfr_extensions

        if self.filetype not in self.all_extensions:
            msg = 'Invalid file type extension! '
            msg += 'Valid extensions: ' + ','.join(self.all_extensions)
            raise ValueError(msg)
        self._f = None

        self.times_names = ['times-x', 'times-u', 'times-v']
        self.energies_names = ['hits-x', 'hits-u', 'hits-v']
        self.energiestimes_names = ['hitimes-x', 'hitimes-u', 'hitimes-v']

    def _tfrecord_to_graph_ops_et(self):
        """
        TODO - handle cases with just energy or time tensors; this is a bit
        tricky - TFRecords are not super-flexible about missing dsets the way
        hdf5 files are. Options are to carefully build the TFRrecod reader to
        be robust against missing values (not sure how to do this yet), or
        use an argument key to pick which values are expected. For now,
        specialize at the function name level ('_et' for 'engy+tm')
        """
        def proces_hitimes(inp, shape):
            """ Keep (N, C, H, W) structure """
            return tf.reshape(tf.decode_raw(inp, tf.float32), shape)

        file_queue = tf.train.string_input_producer(
            [self.filename], name='file_queue'
        )
        reader = tf.TFRecordReader()
        _, tfrecord = reader.read(file_queue)

        tfrecord_features = tf.parse_single_example(
            tfrecord,
            features={
                'eventids': tf.FixedLenFeature([], tf.string),
                'hitimes-x': tf.FixedLenFeature([], tf.string),
                'hitimes-u': tf.FixedLenFeature([], tf.string),
                'hitimes-v': tf.FixedLenFeature([], tf.string),
                'planecodes': tf.FixedLenFeature([], tf.string),
            },
            name='data'
        )
        evtids = tf.decode_raw(tfrecord_features['eventids'], tf.int64)
        hitimesx = proces_hitimes(
            tfrecord_features['hitimes-x'], [-1, 2, 127, 50]
        )
        hitimesu = proces_hitimes(
            tfrecord_features['hitimes-u'], [-1, 2, 127, 25]
        )
        hitimesv = proces_hitimes(
            tfrecord_features['hitimes-v'], [-1, 2, 127, 25]
        )
        pcodes = tf.decode_raw(tfrecord_features['planecodes'], tf.int16)
        pcodes = tf.cast(pcodes, tf.int32)
        pcodes = tf.one_hot(indices=pcodes, depth=67, on_value=1, off_value=0)
        return evtids, hitimesx, hitimesu, hitimesv, pcodes

    def _batch_generator_et(self):
        es, hx, hu, hv, ps = self._tfrecord_to_graph_ops_et()
        capacity = 2 * self.n_events
        es_b, hx_b, hu_b, hv_b, ps_b = tf.train.batch(
            [es, hx, hu, hv, ps],
            batch_size=self.n_events,
            capacity=capacity,
            enqueue_many=True
        )
        return es_b, hx_b, hu_b, hv_b, ps_b
    
    def _read_tfr(self):
        data_dict = {}
        data_dict['energies+times'] = {}
        data_dict['energies'] = {}
        data_dict['times'] = {}

        es_b, hx_b, hu_b, hv_b, ps_b = self._batch_generator_et()
        with tf.Session() as sess:
            # have to run local variable init for `string_input_producer`
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                # TODO - also get and return evtids, pcodes
                _, hx, hu, hv, _ = sess.run(
                    [es_b, hx_b, hu_b, hv_b, ps_b]
                )
                data_dict['energies+times']['x'] = hx
                data_dict['energies+times']['u'] = hu
                data_dict['energies+times']['v'] = hv
            # specifically catch `tf.errors.OutOfRangeError` or we won't handle
            # the exception correctly.
            except tf.errors.OutOfRangeError:
                print('Training stopped - queue is empty.')
            except Exception as e:
                print(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        return data_dict

    def _read_hdf5(self):
        """
        possibilities: energy tensors, time tensors, energy+time tensors
        (2-deep). get everything there into a dictionary keyed by type,
        and then by view.
        """
        def extract_data(dset_name, data_dict, tensor_type):
            view = dset_name[-1]
            try:
                shp = pylab.shape(self._f[dset_name])
            except KeyError:
                print("'{}' does not exist.".format(dset_name))
                shp = None
            if shp is not None:
                if len(shp) == 4:
                    shp = (self.n_events, shp[1], shp[2], shp[3])
                    data_dict[tensor_type][view] = pylab.zeros(shp, dtype='f')
                    data_dict[tensor_type][view] = \
                        self._f[dset_name][:self.n_events]
                else:
                    raise ValueError('Data shape has a bad length!')

        self._f = h5py.File(self.filename, 'r')
        
        # TDOD - get planecodes, eventids, etc. also
        data_dict = {}
        data_dict['energies+times'] = {}
        data_dict['energies'] = {}
        data_dict['times'] = {}
        
        for dset_name in self.energiestimes_names:
            extract_data(dset_name, data_dict, 'energies+times')
        for dset_name in self.energies_names:
            extract_data(dset_name, data_dict, 'energies')
        for dset_name in self.times_names:
            extract_data(dset_name, data_dict, 'times')

        self._f.close()
        
        return data_dict

    def read_data(self):
        """
        return a dictionary of ndarrays, keyed by 'x', 'u', and 'v',
        each with shape (N, C, H, W) - could be anywhere from 1 to 3 views.
        """
        if self.filetype in self.hdf5_extensions:
            return self._read_hdf5()
        elif self.filetype in self.tfr_extensions:
            return self._read_tfr()
        else:
            raise ValueError('Invalid file type extension!')
    

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

# labels_shp = (max_evts,)
# evtids_shp = (max_evts,)
# labels = pylab.zeros(labels_shp, dtype='f')
# evtids = pylab.zeros(evtids_shp, dtype='uint64')
# evtids = f['eventids'][:max_evts]
# try:
#     labels = f['segments'][:max_evts]
#     pcodes = f['planecodes'][:max_evts]
#     zs = f['zs'][:max_evts]
# except KeyError:
#     labels_shp = None


# reader = MnvDataReader(filename=filename)
# data_dict = reader.read_data()



# colorbar_tile = 'scaled energy'
# if have_times:
#     colorbar_tile = 'scaled times'

# for counter, evtid in enumerate(evtids):
#     if evt_plotted > max_evts:
#         break
#     run, subrun, gate, phys_evt = decode_eventid(evtid)
#     if labels_shp is not None:
#         targ = labels[counter]
#         pcode = pcodes[counter]
#         zz = zs[counter]
#         pstring = '{} - {} - {} - {}: tgt: {:02d}; plncd {:03d}; z {}'.format(
#             run, subrun, gate, phys_evt, targ, pcode, zz)
#     else:
#         pstring = '{} - {} - {} - {}'.format(
#             run, subrun, gate, phys_evt)
#     print(pstring)
#     evt = []
#     titles = []
#     if data_x is not None:
#         evt.append(data_x[counter])
#         titles.append('x view')
#     if data_u is not None:
#         evt.append(data_u[counter])
#         titles.append('u view')
#     if data_v is not None:
#         evt.append(data_v[counter])
#         titles.append('v view')
#     fig = pylab.figure(figsize=(9, 3))
#     gs = pylab.GridSpec(1, len(evt))
#     # print np.where(evt == np.max(evt))
#     # print np.max(evt)
#     for i in range(len(evt)):
#         ax = pylab.subplot(gs[i])
#         ax.axis('on')
#         ax.xaxis.set_major_locator(pylab.NullLocator())
#         ax.yaxis.set_major_locator(pylab.NullLocator())
#         # images are normalized such the max e-dep has val 1, independent
#         # of view, so set vmin, vmax here to keep matplotlib from
#         # normalizing each view on its own
#         minv = 0
#         cmap = 'jet'
#         if have_times:
#             minv = -1
#             cmap = 'bwr'
#         im = ax.imshow(evt[i][0], cmap=pylab.get_cmap(cmap),
#                        interpolation='nearest', vmin=minv, vmax=1)
#         cbar = pylab.colorbar(im, fraction=0.04)
#         cbar.set_label(colorbar_tile, size=9)
#         cbar.ax.tick_params(labelsize=6)
#         pylab.title(titles[i], fontsize=12)
#         pylab.xlabel("plane", fontsize=10)
#         pylab.ylabel("strip", fontsize=10)
#     if labels_shp is not None:
#         figname = 'evt_%s_%s_%s_%s_targ_%d_pcode_%d.pdf' % \
#                   (run, subrun, gate, phys_evt, targ, pcode)
#     else:
#         figname = 'evt_%s_%s_%s_%s.pdf' % \
#                   (run, subrun, gate, phys_evt)
#     pylab.savefig(figname)
#     pylab.close()
#     evt_plotted += 1
