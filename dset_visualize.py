#!/usr/bin/env python
"""
Usage:
    python dset_visualize.py -f [file name] -n [optional: # of evts, def==10]
"""
import pylab
import sys
import h5py
import tensorflow as tf
import numpy as np


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


def make_mnv_vertex_finder_data_dict():
    data_dict = {}
    data_dict['energies+times'] = {}
    data_dict['energies'] = {}
    data_dict['times'] = {}
    data_dict['eventids'] = {}
    data_dict['planecodes'] = {}
    data_dict['segments'] = {}
    data_dict['zs'] = {}
    return data_dict


class MnvDataReader:
    def __init__(
            self,
            filename,
            n_events=10,
            views=['x', 'u', 'v'],
            img_sizes=(50, 25),
            n_planecodes=67
    ):
        """
        currently, only work with compressed tfrecord files; assume compression
        for hdf5 is inside, etc.
        """
        self.filename = filename
        self.n_events = n_events
        self.views = views
        self.filetype = filename.split('.')[-1]
        self.img_sizes = img_sizes
        self.n_planecodes = n_planecodes

        self.compression = tf.python_io.TFRecordCompressionType.NONE
        if self.filetype == 'gz':
            self.compression = tf.python_io.TFRecordCompressionType.GZIP
            self.filetype = filename.split('.')[-2]
        elif self.filetype == 'zz':
            self.compression = tf.python_io.TFRecordCompressionType.ZLIB
            self.filetype = filename.split('.')[-2]

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
            [self.filename], name='file_queue', num_epochs=1
        )
        reader = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(
                compression_type=self.compression
            )
        )
        _, tfrecord = reader.read(file_queue)

        tfrecord_features = tf.parse_single_example(
            tfrecord,
            features={
                'eventids': tf.FixedLenFeature([], tf.string),
                'hitimes-x': tf.FixedLenFeature([], tf.string),
                'hitimes-u': tf.FixedLenFeature([], tf.string),
                'hitimes-v': tf.FixedLenFeature([], tf.string),
                'planecodes': tf.FixedLenFeature([], tf.string),
                'segments': tf.FixedLenFeature([], tf.string),
                'zs': tf.FixedLenFeature([], tf.string),
            },
            name='data'
        )
        evtids = tf.decode_raw(tfrecord_features['eventids'], tf.int64)
        hitimesx = proces_hitimes(
            tfrecord_features['hitimes-x'], [-1, 2, 127, self.img_sizes[0]]
        )
        hitimesu = proces_hitimes(
            tfrecord_features['hitimes-u'], [-1, 2, 127, self.img_sizes[1]]
        )
        hitimesv = proces_hitimes(
            tfrecord_features['hitimes-v'], [-1, 2, 127, self.img_sizes[1]]
        )
        pcodes = tf.decode_raw(tfrecord_features['planecodes'], tf.int32)
        pcodes = tf.one_hot(
            indices=pcodes, depth=self.n_planecodes, on_value=1, off_value=0
        )
        segs = tf.decode_raw(tfrecord_features['segments'], tf.uint8)
        zs = tf.decode_raw(tfrecord_features['zs'], tf.float32)
        return evtids, hitimesx, hitimesu, hitimesv, pcodes, segs, zs

    def _batch_generator_et(self):
        es, hx, hu, hv, ps, sg, zs = self._tfrecord_to_graph_ops_et()
        capacity = 2 * self.n_events
        es_b, hx_b, hu_b, hv_b, ps_b, sg_b, zs_b = tf.train.batch(
            [es, hx, hu, hv, ps, sg, zs],
            batch_size=self.n_events,
            capacity=capacity,
            allow_smaller_final_batch=True,
            enqueue_many=True
        )
        return es_b, hx_b, hu_b, hv_b, ps_b, sg_b, zs_b
    
    def _read_tfr(self):
        data_dict = {}
        data_dict['energies+times'] = {}
        data_dict['energies'] = {}
        data_dict['times'] = {}

        es_b, hx_b, hu_b, hv_b, ps_b, sg_b, zs_b = self._batch_generator_et()
        with tf.Session() as sess:
            # have to run local variable init for `string_input_producer`
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                # TODO - also get and return evtids, pcodes
                evts, hx, hu, hv, pcds, segs, zs = sess.run(
                    [es_b, hx_b, hu_b, hv_b, ps_b, sg_b, zs_b]
                )
                data_dict['energies+times']['x'] = hx
                data_dict['energies+times']['u'] = hu
                data_dict['energies+times']['v'] = hv
                data_dict['eventids'] = evts
                data_dict['planecodes'] = np.argmax(
                    pcds, axis=1
                ).reshape(pcds.shape[0], 1)  # pcds
                data_dict['segments'] = segs
                data_dict['zs'] = zs
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
        def extract_data(
                group_name, dset_name, data_dict, tensor_type, dtype='f'
        ):
            view = dset_name[-1]
            try:
                shp = pylab.shape(self._f[group_name][dset_name])
            except KeyError:
                print("'{}/{}' does not exist.".format(
                    group_name, dset_name
                ))
                shp = None
            if shp is not None:
                if len(shp) == 4:
                    shp = (self.n_events, shp[1], shp[2], shp[3])
                    data_dict[tensor_type][view] = \
                        pylab.zeros(shp, dtype=dtype)
                    data_dict[tensor_type][view] = \
                        self._f[group_name][dset_name][:self.n_events]
                elif len(shp) == 2:
                    shp = (self.n_events, 1)
                    data_dict[dset_name] = pylab.zeros(shp, dtype=dtype)
                    data_dict[dset_name] = \
                        self._f[group_name][dset_name][:self.n_events]
                elif len(shp) == 1:
                    shp = (self.n_events,)
                    data_dict[dset_name] = pylab.zeros(shp, dtype=dtype)
                    data_dict[dset_name] = \
                        self._f[group_name][dset_name][:self.n_events]
                else:
                    raise ValueError('Data shape has a bad length!')

        self._f = h5py.File(self.filename, 'r')
        
        data_dict = make_mnv_vertex_finder_data_dict()

        if 'energies+times' in data_dict.keys():
            for dset_name in self.energiestimes_names:
                extract_data(
                    'img_data', dset_name, data_dict, 'energies+times'
                )
        if 'energies' in data_dict.keys():
            for dset_name in self.energies_names:
                extract_data('img_data', dset_name, data_dict, 'energies')
        if 'times' in data_dict.keys():
            for dset_name in self.times_names:
                extract_data('img_data', dset_name, data_dict, 'times')
        if 'eventids' in data_dict.keys():
            extract_data('event_data', 'eventids', data_dict, None, 'uint64')
        if 'planecodes' in data_dict.keys():
            extract_data('event_data', 'planecodes', data_dict, None, 'uint16')
        if 'segments' in data_dict.keys():
            extract_data('event_data', 'segments', data_dict, None, 'uint8')
        if 'zs' in data_dict.keys():
            extract_data('event_data', 'zs', data_dict, None)

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


def make_plots(data_dict, max_events, normed_img):
    """
    cases:
    * 'energies+times',
    * 'energies' and 'times' separately,
    * or 'energies' or 'times'
    If 2-deep tensor, assume energy is index 0, time is index 1
    """
    target_plane_codes = {9: 1, 18: 2, 27: 3, 44: 4, 49: 5}
    pkeys = []
    for k in data_dict.keys():
        if len(data_dict[k]) > 0:
            pkeys.append(k)
    print('Data dictionary present keys: {}'.format(pkeys))

    def combine_et(e_tensor, t_tensor):
        base_shape = e_tensor.shape
        base_shape[1] = 2
        new_tensor = pylab.zeros(base_shape)
        new_tensor[:, 0, :, :] = e_tensor[:, 0, :, :]
        new_tensor[:, 1, :, :] = t_tensor[:, 1, :, :]
        return new_tensor

    types = ['energy', 'time']
    views = ['x', 'u', 'v']   # TODO? build dynamically?

    plotting_two_tensors = False
    if data_dict['energies+times']:
        plotting_two_tensors = True
    elif data_dict['energies'] or data_dict['times']:
        if data_dict['energies'] and data_dict['times']:
            plotting_two_tensors = True
            for view in views:
                data_dict['energies+times'][view] = combine_et(
                    data_dict['energies'][view], data_dict['times'][view]
                )
            data_dict['energies'] = {}
            data_dict['times'] = {}
        else:
            data_dict['oned'] = {}
            data_dict['oned']['type'] = None
            if data_dict['energies']:
                data_dict['oned']['type'] = 'energies'
                types = ['energy']
            else:
                data_dict['oned']['type'] = 'times'
                types = ['time']
            for view in views:
                if data_dict['energies']:
                    data_dict['oned'][view] = data_dict['energies'][view]
                elif data_dict['times']:
                    data_dict['oned'][view] = data_dict['times'][view]
                else:
                    raise ValueError('Mal-formed values tensor!')

    print('  Plotting 2D tensors? {}'.format(plotting_two_tensors))

    evt_plotted = 0
    for counter in range(len(data_dict['eventids'])):
        evtid = data_dict['eventids'][counter]
        segment = data_dict['segments'][counter] \
            if len(data_dict['segments']) > 0 else -1
        planecode = data_dict['planecodes'][counter] \
            if len(data_dict['planecodes']) > 0 else -1
        (run, subrun, gate, phys_evt) = decode_eventid(evtid)
        if evt_plotted > max_events:
            break
        print('Plotting entry {}: {}: {} - {} - {} - {} for segment {} / planecode {}'.format(
                  counter, evtid, run, subrun, gate, phys_evt, segment, planecode
              ))

        # run, subrun, gate, phys_evt = decode_eventid(evtid)
        fig_wid = 9
        fig_height = 6 if plotting_two_tensors else 3
        grid_height = 2 if plotting_two_tensors else 1
        fig = pylab.figure(figsize=(fig_wid, fig_height))
        if planecode in target_plane_codes.keys():
            fig.suptitle('{}/{}/{}/{}: seg {} / pcode {} / targ {}'.format(
                run, subrun, gate, phys_evt,
                segment, planecode, target_plane_codes[planecode[0]]
            ))
        else:
            fig.suptitle('{}/{}/{}/{}: seg {} / pcode {}'.format(
                run, subrun, gate, phys_evt, segment, planecode
            ))
        gs = pylab.GridSpec(grid_height, 3)

        for i, t in enumerate(types):
            if plotting_two_tensors:
                datatyp = 'energies+times'
            else:
                datatyp = 'energies' if t == 'energy' else 'times'
            # set the bounds on the color scale
            if normed_img:
                minv = 0 if t == 'energy' else -1
                maxv = 1
            else:
                maxes = []
                mins = []
                for v in views:
                    maxes.append(
                        np.abs(np.max(data_dict[datatyp][v][counter, i, :, :]))
                    )
                    mins.append(
                        np.abs(np.max(data_dict[datatyp][v][counter, i, :, :]))
                    )
                minv = np.max(mins)
                maxv = np.max(maxes)
                maxex = maxv if maxv > minv else minv
                minv = 0 if minv < 0.0001 else 0 if t == 'energy' else -maxv
                maxv = maxex
            for j, view in enumerate(views):
                gs_pos = i * 3 + j
                ax = pylab.subplot(gs[gs_pos])
                ax.axis('on')
                ax.xaxis.set_major_locator(pylab.NullLocator())
                ax.yaxis.set_major_locator(pylab.NullLocator())
                cmap = 'jet' if t == 'energy' else 'bwr'
                cbt = 'energy' if t == 'energy' else 'times'
                datap = data_dict[datatyp][view][counter, i, :, :]
                # make the plot
                im = ax.imshow(
                    datap,
                    cmap=pylab.get_cmap(cmap),
                    interpolation='nearest',
                    vmin=minv, vmax=maxv
                )
                cbar = pylab.colorbar(im, fraction=0.04)
                cbar.set_label(cbt, size=9)
                cbar.ax.tick_params(labelsize=6)
                pylab.title(t + ' - ' + view, fontsize=12)
                pylab.xlabel('plane', fontsize=10)
                pylab.ylabel('strip', fontsize=10)
        figname = 'evt_%d.pdf' % (counter)
        pylab.savefig(figname, bbox_inches='tight')
        pylab.close()
        evt_plotted += 1


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-f', '--file', dest='filename',
                      help='Dset file name', metavar='FILENAME',
                      default=None, type='string')
    parser.add_option('-n', '--nevents', dest='n_events', default=10,
                      help='Number of events', metavar='N_EVENTS',
                      type='int')
    parser.add_option('--imgw_x', dest='imgw_x', default=50,
                      help='Image width (x)', metavar='IMG_WIDTHX',
                      type='int')
    parser.add_option('--imgw_uv', dest='imgw_uv', default=25,
                      help='Image width (uv)', metavar='IMG_WIDTHUV',
                      type='int')
    parser.add_option('--n_planecodes', dest='n_planecodes', default=67,
                      help='Number of planecodes (onehot)',
                      metavar='N_PLANECODES', type='int')
    parser.add_option('--normed_img', dest='normed_img', default=False,
                      help='Image from normalized source',
                      metavar='NORMED_IMG', action='store_true')

    (options, args) = parser.parse_args()

    if not options.filename:
        print("\nSpecify file (-f):\n\n")
        print(__doc__)
        sys.exit(1)

    img_sizes = (options.imgw_x, options.imgw_uv)
    reader = MnvDataReader(
        filename=options.filename,
        n_events=options.n_events,
        img_sizes=img_sizes,
        n_planecodes=options.n_planecodes
    )
    dd = reader.read_data()

    make_plots(dd, options.n_events, options.normed_img)
