#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import h5py
import logging

LOGGER = logging.getLogger(__name__)


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


class MnvDataReaderVertexST:
    """
    Minerva Data Reader for (target) vertex-finder "SpaceTime" data

    name values are usually, e.g., 'train' or 'validation', etc.

    allowed compression values:
    * `tf.python_io.TFRecordCompressionType.ZLIB`
    * `tf.python_io.TFRecordCompressionType.GZIP`
    """
    def __init__(
            self, filenames_list, batch_size=100,
            name='reader', data_format='NHWC', compression=None
    ):
        self.filenames_list = filenames_list
        self.batch_size = batch_size
        self.name = name
        imgdat_names = {}
        imgdat_names['x'] = 'hitimes-x'
        imgdat_names['u'] = 'hitimes-u'
        imgdat_names['v'] = 'hitimes-v'
        self.imgdat_names = imgdat_names
        self.data_format = data_format
        self.compression = tf.python_io.TFRecordCompressionType.NONE
        if compression:
            self.compression = compression

    def _make_mnv_vertex_finder_batch_dict(
            self, eventids_batch,
            hitimesx_batch, hitimesu_batch, hitimesv_batch,
            planecodes_batch, segments_batch, zs_batch
    ):
        batch_dict = {}
        batch_dict['eventids'] = eventids_batch
        batch_dict[self.imgdat_names['x']] = hitimesx_batch
        batch_dict[self.imgdat_names['u']] = hitimesu_batch
        batch_dict[self.imgdat_names['v']] = hitimesv_batch
        batch_dict['planecodes'] = planecodes_batch
        batch_dict['segments'] = segments_batch
        batch_dict['zs'] = zs_batch
        return batch_dict

    def _tfrecord_to_graph_ops_et(self, num_epochs):
        """
        specialize at the function name level ('_et' for 'engy+tm')
        """
        def proces_hitimes(inp, shape):
            """ Start with a (N, C, H, W) structure, -> (N, H, W, C)? """
            tnsr = tf.reshape(tf.decode_raw(inp, tf.float32), shape)
            if self.data_format == 'NCHW':
                return tnsr
            elif self.data_format == 'NHWC':
                return tf.transpose(tnsr, [0, 2, 3, 1])
            else:
                raise ValueError('Invalid data format in data reader!')

        file_queue = tf.train.string_input_producer(
            self.filenames_list,
            name=self.name+'_file_queue',
            num_epochs=num_epochs
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
                self.imgdat_names['x']: tf.FixedLenFeature([], tf.string),
                self.imgdat_names['u']: tf.FixedLenFeature([], tf.string),
                self.imgdat_names['v']: tf.FixedLenFeature([], tf.string),
                'planecodes': tf.FixedLenFeature([], tf.string),
                'segments': tf.FixedLenFeature([], tf.string),
                'zs': tf.FixedLenFeature([], tf.string),
            },
            name=self.name+'_data'
        )
        evtids = tf.decode_raw(tfrecord_features['eventids'], tf.int64)
        hitimesx = proces_hitimes(
            tfrecord_features[self.imgdat_names['x']], [-1, 2, 127, 50]
        )
        hitimesu = proces_hitimes(
            tfrecord_features[self.imgdat_names['u']], [-1, 2, 127, 25]
        )
        hitimesv = proces_hitimes(
            tfrecord_features[self.imgdat_names['v']], [-1, 2, 127, 25]
        )
        pcodes = tf.decode_raw(tfrecord_features['planecodes'], tf.int16)
        pcodes = tf.cast(pcodes, tf.int32)
        pcodes = tf.one_hot(indices=pcodes, depth=67, on_value=1, off_value=0)
        segs = tf.decode_raw(tfrecord_features['segments'], tf.uint8)
        segs = tf.cast(segs, tf.int32)
        segs = tf.one_hot(indices=segs, depth=11, on_value=1, off_value=0)
        zs = tf.decode_raw(tfrecord_features['zs'], tf.float32)
        return evtids, hitimesx, hitimesu, hitimesv, pcodes, segs, zs

    def batch_generator(self, num_epochs=1):
        es, x, u, v, ps, sg, zs = self._tfrecord_to_graph_ops_et(num_epochs)
        capacity = 10 * self.batch_size
        es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b = tf.train.batch(
            [es, x, u, v, ps, sg, zs],
            batch_size=self.batch_size,
            capacity=capacity,
            enqueue_many=True,
            allow_smaller_final_batch=True,
            name=self.name+'_batch'
        )
        return self._make_mnv_vertex_finder_batch_dict(
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b
        )

    def shuffle_batch_generator(self, num_epochs=1):
        es, x, u, v, ps, sg, zs = self._tfrecord_to_graph_ops_et(num_epochs)
        min_after_dequeue = 3 * self.batch_size
        capacity = 10 * self.batch_size
        es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b = tf.train.shuffle_batch(
            [es, x, u, v, ps, sg, zs],
            batch_size=self.batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True,
            allow_smaller_final_batch=True,
            name=self.name+'_shuffle_batch'
        )
        return self._make_mnv_vertex_finder_batch_dict(
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b
        )


class MnvDataReaderVertexSTHDF5:
    def __init__(self, filename, views=['x', 'u', 'v']):
        self.filename = filename
        self.views = views
        self.filetype = filename.split('.')[-1]
        self.hdf5_extensions = ['hdf5', 'h5']

        if self.filetype not in self.hdf5_extensions:
            msg = 'Invalid file type extension! '
            msg += 'Valid extensions: ' + ','.join(self.all_extensions)
            raise ValueError(msg)
        self._f = None

        self.times_names = ['times-x', 'times-u', 'times-v']
        self.energies_names = ['hits-x', 'hits-u', 'hits-v']
        self.energiestimes_names = ['hitimes-x', 'hitimes-u', 'hitimes-v']

    def open(self):
        LOGGER.info("Opening hdf5 file {}".format(self.file))
        self._f = h5py.File(self.file, 'r')
        for name in self._f:
            LOGGER.info('{:>12}: {:>8}: shape = {}'.format(
                name, np.dtype(self._f[name]), np.shape(self._f[name])
            ))

    def close(self):
        try:
            self._f.close()
        except AttributeError:
            LOGGER.info('hdf5 file is not open yet.')

    def get_nevents(self):
        sizes = [self._f[d].shape[0] for d in self._f]
        if min(sizes) != max(sizes):
            msg = "All dsets must have the same size!"
            LOGGER.error(msg)
            raise ValueError(msg)
        return sizes[0]

    def _read_hdf5(self, start_idx, stop_idx):
        """
        possibilities: energy tensors, time tensors, energy+time tensors
        (2-deep). get everything there into a dictionary keyed by type,
        and then by view.
        """
        n_events = stop_idx - start_idx

        def extract_data(dset_name, data_dict, tensor_type, dtype='f'):
            view = dset_name[-1]
            try:
                shp = np.shape(self._f[dset_name])
            except KeyError:
                print("'{}' does not exist.".format(dset_name))
                shp = None
            if shp is not None:
                if len(shp) == 4:
                    shp = (n_events, shp[1], shp[2], shp[3])
                    data_dict[tensor_type][view] = \
                        np.zeros(shp, dtype=dtype)
                    data_dict[tensor_type][view] = \
                        self._f[dset_name][start_idx:stop_idx]
                elif len(shp) == 1:
                    shp = (n_events,)
                    data_dict[dset_name] = np.zeros(shp, dtype=dtype)
                    data_dict[dset_name] = \
                        self._f[dset_name][start_idx:stop_idx]
                else:
                    raise ValueError('Data shape has a bad length!')

        self._f = h5py.File(self.filename, 'r')
        
        data_dict = make_mnv_vertex_finder_data_dict()

        if 'energies+times' in data_dict.keys():
            for dset_name in self.energiestimes_names:
                extract_data(dset_name, data_dict, 'energies+times')
        if 'energies' in data_dict.keys():
            for dset_name in self.energies_names:
                extract_data(dset_name, data_dict, 'energies')
        if 'times' in data_dict.keys():
            for dset_name in self.times_names:
                extract_data(dset_name, data_dict, 'times')
        if 'eventids' in data_dict.keys():
            extract_data('eventids', data_dict, None, 'uint64')
        if 'planecodes' in data_dict.keys():
            extract_data('planecodes', data_dict, None, 'uint16')
        if 'segments' in data_dict.keys():
            extract_data('segments', data_dict, None, 'uint8')
        if 'zs' in data_dict.keys():
            extract_data('zs', data_dict, None)
        
        return data_dict
