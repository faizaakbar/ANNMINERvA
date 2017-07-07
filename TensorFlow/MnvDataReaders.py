#!/usr/bin/env python
import tensorflow as tf


class MnvDataReaderVertexST:
    """
    Minerva Data Reader for (target) vertex-finder "SpaceTime" data
    """
    def __init__(self, filenames_list, batch_size=100, views=['x', 'u', 'v']):
        self.filenames_list = filenames_list
        self.batch_size = batch_size
        self.views = views
        self.energiestimes_names = ['hitimes-x', 'hitimes-u', 'hitimes-v']
        self._f = None

    def _make_mnv_vertex_finder_batch_dict(
            eventids_batch, hitimesx_batch, hitimesu_batch, hitimesv_batch,
            planecodes_batch, segments_batch, zs_batch
    ):
        batch_dict = {}
        batch_dict['eventids'] = eventids_batch
        batch_dict['hitimes-x'] = hitimesx_batch
        batch_dict['hitimes-u'] = hitimesu_batch
        batch_dict['hitimes-v'] = hitimesv_batch
        batch_dict['planecodes'] = planecodes_batch
        batch_dict['segments'] = segments_batch
        batch_dict['zs'] = zs_batch
        return batch_dict

    def _tfrecord_to_graph_ops_et(self):
        """
        specialize at the function name level ('_et' for 'engy+tm')
        """
        def proces_hitimes(inp, shape):
            """ Keep (N, C, H, W) structure """
            return tf.reshape(tf.decode_raw(inp, tf.float32), shape)

        file_queue = tf.train.string_input_producer(
            self.filenames_list, name='file_queue'
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
                'segments': tf.FixedLenFeature([], tf.string),
                'zs': tf.FixedLenFeature([], tf.string),
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
        segs = tf.decode_raw(tfrecord_features['segments'], tf.uint8)
        zs = tf.decode_raw(tfrecord_features['zs'], tf.float32)
        return evtids, hitimesx, hitimesu, hitimesv, pcodes, segs, zs

    def batch_generator(self):
        es, hx, hu, hv, ps, sg, zs = self._tfrecord_to_graph_ops_et()
        capacity = 10 * self.batch_size
        es_b, hx_b, hu_b, hv_b, ps_b, sg_b, zs_b = tf.train.batch(
            [es, hx, hu, hv, ps, sg, zs],
            batch_size=self.n_events,
            capacity=capacity,
            enqueue_many=True
        )
        return _make_mnv_vertex_finder_data_dict(
            es_b, hx_b, hu_b, hv_b, ps_b, sg_b, zs_b
        )

    def shuffle_batch_generator(self):
        es, hx, hu, hv, ps, sg, zs = self._tfrecord_to_graph_ops_et()
        min_after_dequeue = 3 * self.batch_size
        capacity = 10 * self.batch_size
        es_b, hx_b, hu_b, hv_b, ps_b, sg_b, zs_b = tf.train.shuffle_batch(
            [es, hx, hu, hv, ps, sg, zs],
            batch_size=self.batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True
        )
        return _make_mnv_vertex_finder_data_dict(
            es_b, hx_b, hu_b, hv_b, ps_b, sg_b, zs_b
        )
