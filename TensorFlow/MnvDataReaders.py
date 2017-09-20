#!/usr/bin/env python
import tensorflow as tf


class MnvDataReaderVertexST:
    """
    Minerva Data Reader for (target) vertex-finder "SpaceTime" data

    name values are usually, e.g., 'train' or 'validation', etc.

    allowed compression values:
    * `tf.python_io.TFRecordCompressionType.ZLIB`
    * `tf.python_io.TFRecordCompressionType.GZIP`
    """
    def __init__(self, args_dict):
        """
        img_shp = (imgh, imgw_x, imgw_uv, img_depth)
        TODO - get the img depth into this call also...
        """
        self.filenames_list = args_dict['FILENAMES_LIST']
        self.batch_size = args_dict['BATCH_SIZE']
        self.name = args_dict['NAME']
        self.img_shp = args_dict['IMG_SHP']
        self.n_planecodes = args_dict['N_PLANECODES']
        imgdat_names = {}
        imgdat_names['x'] = 'hitimes-x'
        imgdat_names['u'] = 'hitimes-u'
        imgdat_names['v'] = 'hitimes-v'
        self.imgdat_names = imgdat_names
        self.data_format = args_dict['DATA_FORMAT']
        self.compression = args_dict['FILE_COMPRESSION']

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

        with tf.variable_scope(self.name + '_tfrec_to_graph_ops'):
            file_queue = tf.train.string_input_producer(
                self.filenames_list,
                name=self.name+'_file_queue',
                num_epochs=num_epochs
            )
            reader = tf.TFRecordReader(
                options=tf.python_io.TFRecordOptions(
                    compression_type=self.compression
                ),
                name=self.name+'_tfrecordreader'
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
            with tf.variable_scope(self.name + '_eventids'):
                evtids = tf.decode_raw(tfrecord_features['eventids'], tf.int64)
            with tf.variable_scope(self.name + '_hitimes'):
                hitimesx = proces_hitimes(
                    tfrecord_features[self.imgdat_names['x']],
                    [-1, self.img_shp[3], self.img_shp[0], self.img_shp[1]]
                )
                hitimesu = proces_hitimes(
                    tfrecord_features[self.imgdat_names['u']],
                    [-1, self.img_shp[3], self.img_shp[0], self.img_shp[2]]
                )
                hitimesv = proces_hitimes(
                    tfrecord_features[self.imgdat_names['v']],
                    [-1, self.img_shp[3], self.img_shp[0], self.img_shp[2]]
                )
            with tf.variable_scope(self.name + '_planecodes'):
                pcodes = tf.decode_raw(
                    tfrecord_features['planecodes'], tf.int32
                )
                pcodes = tf.one_hot(
                    indices=pcodes, depth=self.n_planecodes,
                    on_value=1, off_value=0
                )
            with tf.variable_scope(self.name + '_segments'):
                segs = tf.decode_raw(tfrecord_features['segments'], tf.uint8)
                segs = tf.one_hot(
                    indices=segs, depth=11, on_value=1, off_value=0
                )
            with tf.variable_scope(self.name + '_zs'):
                zs = tf.decode_raw(tfrecord_features['zs'], tf.float32)
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
