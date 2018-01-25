#!/usr/bin/env python
import tensorflow as tf

from MnvDataConstants import EVENT_DATA
from MnvDataConstants import EVENTIDS, PLANECODES, SEGMENTS, ZS
from MnvDataConstants import IMG_DATA
from MnvDataConstants import HITIMESU, HITIMESV, HITIMESX


class MnvTFRecordReaderBase:
    """
    Minerva Data Reader for TFRecord files.

    name values are usually, e.g., 'train' or 'validation', etc.

    allowed compression values:
    * `tf.python_io.TFRecordCompressionType.ZLIB`
    * `tf.python_io.TFRecordCompressionType.GZIP`
    """
    def __init__(self, args_dict):
        self.filenames_list = args_dict['FILENAMES_LIST']
        self.batch_size = args_dict['BATCH_SIZE']
        self.name = args_dict['NAME']
        self.img_shp = args_dict['IMG_SHP']
        self.n_planecodes = args_dict['N_PLANECODES']
        self.data_format = args_dict['DATA_FORMAT']
        self.compression = args_dict['FILE_COMPRESSION']

    def _process_hitimes(self, inp, shape):
        """ Start with a (N, C, H, W) structure, -> (N, H, W, C)? """
        tnsr = tf.reshape(tf.decode_raw(inp, tf.float32), shape)
        if self.data_format == 'NCHW':
            return tnsr
        elif self.data_format == 'NHWC':
            return tf.transpose(tnsr, [0, 2, 3, 1])
        else:
            raise ValueError('Invalid data format in data reader!')

    def _get_tfrecord_filequeue_and_reader(self):
        file_queue = tf.train.string_input_producer(
            self.filenames_list, name='file_queue', num_epochs=1
        )
        reader = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(
                compression_type=self.compression
            )
        )
        _, tfrecord = reader.read(file_queue)
        return tfrecord

    def _decode_eventids(self, tfrecord_features):
        evtids = tf.decode_raw(tfrecord_features[EVENTIDS], tf.int64)
        return evtids

    def _decode_planecodes(self, tfrecord_features):
        pcodes = tf.decode_raw(
            tfrecord_features[PLANECODES], tf.int32
        )
        pcodes = tf.one_hot(
            indices=pcodes, depth=self.n_planecodes, on_value=1, off_value=0
        )
        return pcodes

    def _decode_segments(self, tfrecord_features):
        segs = tf.decode_raw(tfrecord_features[SEGMENTS], tf.uint8)
        segs = tf.one_hot(indices=segs, depth=11, on_value=1, off_value=0)
        return segs

    def _decode_zs(self, tfrecord_features):
        zs = tf.decode_raw(tfrecord_features[ZS], tf.float32)
        return zs

    def _decode_hitimesx(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESX],
            [-1, self.img_shp[3], self.img_shp[0], self.img_shp[1]]
        )

    def _decode_hitimesu(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESU],
            [-1, self.img_shp[3], self.img_shp[0], self.img_shp[2]]
        )

    def _decode_hitimesv(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESV],
            [-1, self.img_shp[3], self.img_shp[0], self.img_shp[2]]
        )


class MnvDataReaderVertexST(MnvTFRecordReaderBase):
    """
    Minerva Data Reader for (target) vertex-finder "SpaceTime" data
    """
    def __init__(self, args_dict):
        """
        img_shp = (imgh, imgw_x, imgw_uv, img_depth)
        TODO - get the img depth into this call also...
        """
        MnvTFRecordReaderBase.__init__(self, args_dict)

    def _make_mnv_vertex_finder_batch_dict(
            self, eventids_batch,
            hitimesx_batch, hitimesu_batch, hitimesv_batch,
            planecodes_batch, segments_batch, zs_batch
    ):
        batch_dict = {}
        batch_dict[EVENTIDS] = eventids_batch
        batch_dict[HITIMESX] = hitimesx_batch
        batch_dict[HITIMESU] = hitimesu_batch
        batch_dict[HITIMESV] = hitimesv_batch
        batch_dict[PLANECODES] = planecodes_batch
        batch_dict[SEGMENTS] = segments_batch
        batch_dict[ZS] = zs_batch
        return batch_dict

    def _tfrecord_to_graph_ops_et(self, num_epochs):
        """
        specialize at the function name level ('_et' for 'engy+tm')
        """
        with tf.variable_scope(self.name + '_tfrec_to_graph_ops'):
            tfrecord = self._get_tfrecord_filequeue_and_reader()
            tfrecord_features = tf.parse_single_example(
                tfrecord,
                features={
                    EVENTIDS: tf.FixedLenFeature([], tf.string),
                    HITIMESX: tf.FixedLenFeature([], tf.string),
                    HITIMESU: tf.FixedLenFeature([], tf.string),
                    HITIMESV: tf.FixedLenFeature([], tf.string),
                    PLANECODES: tf.FixedLenFeature([], tf.string),
                    SEGMENTS: tf.FixedLenFeature([], tf.string),
                    ZS: tf.FixedLenFeature([], tf.string),
                },
                name=self.name+'_data'
            )
            with tf.variable_scope(self.name + '_' + EVENT_DATA):
                evtids = self._decode_eventids(tfrecord_features)
                pcodes = self._decode_planecodes(tfrecord_features)
                segs = self._decode_segments(tfrecord_features)
                zs = self._decode_zs(tfrecord_features)
            with tf.variable_scope(self.name + '_' + IMG_DATA):
                hitimesx = self._decode_hitimesx(tfrecord_features)
                hitimesu = self._decode_hitimesu(tfrecord_features)
                hitimesv = self._decode_hitimesv(tfrecord_features)
        return evtids, hitimesx, hitimesu, hitimesv, pcodes, segs, zs

    def batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_batch_gen'):
            es, x, u, v, ps, sg, zs = \
                self._tfrecord_to_graph_ops_et(num_epochs)
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
        with tf.variable_scope(self.name+'_shufflebatch_gen'):
            es, x, u, v, ps, sg, zs = \
                self._tfrecord_to_graph_ops_et(num_epochs)
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


class MnvDataReaderHamultKineST(MnvTFRecordReaderBase):
    """
    Minerva Data Reader for hadmult-kine "SpaceTime" data
    """
    def __init__(self, args_dict):
        """
        img_shp = (imgh, imgw_x, imgw_uv, img_depth)
        TODO - get the img depth into this call also...
        """
        MnvTFRecordReaderBase.__init__(self, args_dict)

    def _make_mnv_hadmultkine_batch_dict(
            self, eventids_batch,
            hitimesx_batch, hitimesu_batch, hitimesv_batch,
            planecodes_batch, segments_batch, zs_batch
    ):
        batch_dict = {}
        batch_dict[EVENTIDS] = eventids_batch
        batch_dict[HITIMESX] = hitimesx_batch
        batch_dict[HITIMESU] = hitimesu_batch
        batch_dict[HITIMESV] = hitimesv_batch
        batch_dict[PLANECODES] = planecodes_batch
        batch_dict[SEGMENTS] = segments_batch
        batch_dict[ZS] = zs_batch
        return batch_dict

    def _tfrecord_to_graph_ops_et(self, num_epochs):
        """
        specialize at the function name level ('_et' for 'engy+tm')
        """
        with tf.variable_scope(self.name + '_tfrec_to_graph_ops'):
            tfrecord = self._get_tfrecord_filequeue_and_reader()
            tfrecord_features = tf.parse_single_example(
                tfrecord,
                features={
                    EVENTIDS: tf.FixedLenFeature([], tf.string),
                    HITIMESX: tf.FixedLenFeature([], tf.string),
                    HITIMESU: tf.FixedLenFeature([], tf.string),
                    HITIMESV: tf.FixedLenFeature([], tf.string),
                    PLANECODES: tf.FixedLenFeature([], tf.string),
                    SEGMENTS: tf.FixedLenFeature([], tf.string),
                    ZS: tf.FixedLenFeature([], tf.string),
                },
                name=self.name+'_data'
            )
            with tf.variable_scope(self.name + '_' + EVENT_DATA):
                evtids = self._decode_eventids(tfrecord_features)
                pcodes = self._decode_planecodes(tfrecord_features)
                segs = self._decode_segments(tfrecord_features)
                zs = self._decode_zs(tfrecord_features)
            with tf.variable_scope(self.name + '_' + IMG_DATA):
                hitimesx = self._decode_hitimesx(tfrecord_features)
                hitimesu = self._decode_hitimesu(tfrecord_features)
                hitimesv = self._decode_hitimesv(tfrecord_features)
        return evtids, hitimesx, hitimesu, hitimesv, pcodes, segs, zs

    def batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_batch_gen'):
            es, x, u, v, ps, sg, zs = \
                self._tfrecord_to_graph_ops_et(num_epochs)
            capacity = 10 * self.batch_size
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b = tf.train.batch(
                [es, x, u, v, ps, sg, zs],
                batch_size=self.batch_size,
                capacity=capacity,
                enqueue_many=True,
                allow_smaller_final_batch=True,
                name=self.name+'_batch'
            )
        return self._make_mnv_hadmultkine_batch_dict(
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b
        )

    def shuffle_batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_shufflebatch_gen'):
            es, x, u, v, ps, sg, zs = \
                self._tfrecord_to_graph_ops_et(num_epochs)
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
        return self._make_mnv_hadmultkine_batch_dict(
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b
        )
