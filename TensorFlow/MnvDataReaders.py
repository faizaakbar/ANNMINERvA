#!/usr/bin/env python
from collections import OrderedDict
import tensorflow as tf

from MnvDataConstants import EVENTIDS, PLANECODES, SEGMENTS, ZS
from MnvDataConstants import HITIMESU, HITIMESV, HITIMESX
from MnvDataConstants import QSQRD, WINV, XBJ, YBJ, CURRENT, INT_TYPE, TARGETZ
from MnvDataConstants import ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS
from MnvDataConstants import ESUM_NEUTPIONS, ESUM_NEUTRONS
from MnvDataConstants import ESUM_OTHERS, ESUM_PROTONS
from MnvDataConstants import N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS
from MnvDataConstants import N_NEUTPIONS, N_NEUTRONS
from MnvDataConstants import N_OTHERS, N_PROTONS
from MnvDataConstants import ESUM_ELECTRONS, ESUM_MUONS, ESUM_TAUS
from MnvDataConstants import N_ELECTRONS, N_MUONS, N_TAUS


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
        self.data_fields = None
        self._basic_int32_fields = set([
            CURRENT, INT_TYPE, TARGETZ,
            N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS,
            N_NEUTPIONS, N_NEUTRONS, N_OTHERS, N_PROTONS,
            N_ELECTRONS, N_MUONS, N_TAUS,
        ])
        self._basic_float32_fields = set([
            ZS, QSQRD, WINV, XBJ, YBJ,
            ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS,
            ESUM_NEUTPIONS, ESUM_NEUTRONS, ESUM_OTHERS, ESUM_PROTONS,
            ESUM_ELECTRONS, ESUM_MUONS, ESUM_TAUS,
        ])

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

    def _decode_basic(self, tfrecord_features, field, tf_dtype):
        return tf.decode_raw(tfrecord_features[field], tf_dtype)

    def _decode_tfrecord_feature(self, tfrecord_features, field):
        if field == EVENTIDS:
            return self._decode_basic(tfrecord_features, field, tf.int64)
        elif field == HITIMESU:
            return self._decode_hitimesu(tfrecord_features)
        elif field == HITIMESV:
            return self._decode_hitimesv(tfrecord_features)
        elif field == HITIMESX:
            return self._decode_hitimesx(tfrecord_features)
        elif field == PLANECODES:
            return self._decode_planecodes(tfrecord_features)
        elif field == SEGMENTS:
            return self._decode_segments(tfrecord_features)
        elif field in self._basic_float32_fields:
            return self._decode_basic(tfrecord_features, field, tf.float32)
        elif field in self._basic_int32_fields:
            return self._decode_basic(tfrecord_features, field, tf.int32)

    def _tfrecord_to_graph_ops(self, num_epochs):
        od = OrderedDict()
        with tf.variable_scope(self.name + '_tfrec_to_graph_ops'):
            tfrecord = self._get_tfrecord_filequeue_and_reader()
            tfrecord_features = tf.parse_single_example(
                tfrecord,
                features=self._features_dict,
                name=self.name+'_data'
            )
            with tf.variable_scope(self.name + '_input_data'):
                for field in self.data_fields:
                    od[field] = self._decode_tfrecord_feature(
                        tfrecord_features, field
                    )
        return od

    def batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_batch_gen'):
            od = self._tfrecord_to_graph_ops(num_epochs)
            capacity = 10 * self.batch_size
            batch_values = tf.train.batch(
                od.values(),
                batch_size=self.batch_size,
                capacity=capacity,
                enqueue_many=True,
                allow_smaller_final_batch=True,
                name=self.name+'_batch'
            )
        rd = OrderedDict(zip(od.keys(), batch_values))
        return rd

    def shuffle_batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_shufflebatch_gen'):
            od = self._tfrecord_to_graph_ops(num_epochs)
            min_after_dequeue = 3 * self.batch_size
            capacity = 10 * self.batch_size
            batch_values = tf.train.shuffle_batch(
                od.values(),
                batch_size=self.batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                enqueue_many=True,
                allow_smaller_final_batch=True,
                name=self.name+'_shuffle_batch'
            )
        rd = OrderedDict(zip(od.keys(), batch_values))
        return rd


class MnvDataReaderImageST(MnvTFRecordReaderBase):
    """
    Minerva Data Reader for plain image "SpaceTime" data
    """
    def __init__(self, args_dict):
        """
        img_shp = (imgh, imgw_x, imgw_uv, img_depth)
        TODO - get the img depth into this call also...
        """
        MnvTFRecordReaderBase.__init__(self, args_dict)
        self.data_fields = sorted([
            EVENTIDS, HITIMESU, HITIMESV, HITIMESX
        ])
        self._features_dict = {
            f: tf.FixedLenFeature([], tf.string) for f in self.data_fields
        }


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
        self.data_fields = sorted([
            EVENTIDS, PLANECODES, SEGMENTS, ZS,
            HITIMESU, HITIMESV, HITIMESX
        ])
        self._features_dict = {
            f: tf.FixedLenFeature([], tf.string) for f in self.data_fields
        }


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
        self.data_fields = sorted([
            EVENTIDS, PLANECODES, SEGMENTS, ZS,
            HITIMESU, HITIMESV, HITIMESX,
            QSQRD, WINV, XBJ, YBJ, CURRENT, INT_TYPE, TARGETZ,
            ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS,
            ESUM_NEUTPIONS, ESUM_NEUTRONS, ESUM_OTHERS, ESUM_PROTONS,
            N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS,
            N_NEUTPIONS, N_NEUTRONS, N_OTHERS, N_PROTONS,
            ESUM_ELECTRONS, ESUM_MUONS, ESUM_TAUS,
            N_ELECTRONS, N_MUONS, N_TAUS,
        ])
        self._features_dict = {
            f: tf.FixedLenFeature([], tf.string) for f in self.data_fields
        }

