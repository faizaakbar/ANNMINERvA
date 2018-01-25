#!/usr/bin/env python
import tensorflow as tf

from MnvDataConstants import EVENT_DATA
from MnvDataConstants import EVENTIDS, PLANECODES, SEGMENTS, ZS
from MnvDataConstants import IMG_DATA
from MnvDataConstants import HITIMESU, HITIMESV, HITIMESX
from MnvDataConstants import GEN_DATA
from MnvDataConstants import QSQRD, WINV, XBJ, YBJ, CURRENT, INT_TYPE, TARGETZ
from MnvDataConstants import HADRO_DATA
from MnvDataConstants import ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS
from MnvDataConstants import ESUM_NEUTPIONS, ESUM_NEUTRONS
from MnvDataConstants import ESUM_OTHERS, ESUM_PROTONS
from MnvDataConstants import N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS
from MnvDataConstants import N_NEUTPIONS, N_NEUTRONS
from MnvDataConstants import N_OTHERS, N_PROTONS
from MnvDataConstants import LEPTO_DATA
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
            planecodes_batch, segments_batch, zs_batch,
            qsqrd, w, xbj, ybj, current, int_type, targetZ,
            esum_chgdkaons, esum_chgdpions, esum_hadmultmeas,
            esum_neutpions, esum_neutrons, esum_others, esum_protons,
            n_chgdkaons, n_chgdpions, n_hadmultmeas,
            n_neutpions, n_neutrons, n_others, n_protons
    ):
        batch_dict = {}
        batch_dict[EVENTIDS] = eventids_batch
        batch_dict[HITIMESX] = hitimesx_batch
        batch_dict[HITIMESU] = hitimesu_batch
        batch_dict[HITIMESV] = hitimesv_batch
        batch_dict[PLANECODES] = planecodes_batch
        batch_dict[SEGMENTS] = segments_batch
        batch_dict[ZS] = zs_batch
        batch_dict[QSQRD] = qsqrd
        batch_dict[WINV] = w
        batch_dict[XBJ] = xbj
        batch_dict[YBJ] = ybj
        batch_dict[CURRENT] = current
        batch_dict[INT_TYPE] = int_type
        batch_dict[TARGETZ] = targetZ
        batch_dict[ESUM_CHGDKAONS] = esum_chgdkaons
        batch_dict[ESUM_CHGDPIONS] = esum_chgdpions
        batch_dict[ESUM_HADMULTMEAS] = esum_hadmultmeas
        batch_dict[ESUM_NEUTPIONS] = esum_neutpions
        batch_dict[ESUM_NEUTRONS] = esum_neutrons
        batch_dict[ESUM_OTHERS] = esum_others
        batch_dict[ESUM_PROTONS] = esum_protons
        batch_dict[N_CHGDKAONS] = n_chgdkaons
        batch_dict[N_CHGDPIONS] = n_chgdpions
        batch_dict[N_HADMULTMEAS] = n_hadmultmeas
        batch_dict[N_NEUTPIONS] = n_neutpions
        batch_dict[N_NEUTRONS] = n_neutrons
        batch_dict[N_OTHERS] = n_others
        batch_dict[N_PROTONS] = n_protons
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
                    QSQRD: tf.FixedLenFeature([], tf.string),
                    WINV: tf.FixedLenFeature([], tf.string),
                    XBJ: tf.FixedLenFeature([], tf.string),
                    YBJ: tf.FixedLenFeature([], tf.string),
                    CURRENT: tf.FixedLenFeature([], tf.string),
                    INT_TYPE: tf.FixedLenFeature([], tf.string),
                    TARGETZ: tf.FixedLenFeature([], tf.string),
                    ESUM_CHGDKAONS: tf.FixedLenFeature([], tf.string),
                    ESUM_CHGDPIONS: tf.FixedLenFeature([], tf.string),
                    ESUM_HADMULTMEAS: tf.FixedLenFeature([], tf.string),
                    ESUM_NEUTPIONS: tf.FixedLenFeature([], tf.string),
                    ESUM_NEUTRONS: tf.FixedLenFeature([], tf.string),
                    ESUM_OTHERS: tf.FixedLenFeature([], tf.string),
                    ESUM_PROTONS: tf.FixedLenFeature([], tf.string),
                    N_CHGDKAONS: tf.FixedLenFeature([], tf.string),
                    N_CHGDPIONS: tf.FixedLenFeature([], tf.string),
                    N_HADMULTMEAS: tf.FixedLenFeature([], tf.string),
                    N_NEUTPIONS: tf.FixedLenFeature([], tf.string),
                    N_NEUTRONS: tf.FixedLenFeature([], tf.string),
                    N_OTHERS: tf.FixedLenFeature([], tf.string),
                    N_PROTONS: tf.FixedLenFeature([], tf.string),
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
            with tf.variable_scope(self.name + '_' + GEN_DATA):
                qsqrd = tf.decode_raw(tfrecord_features[QSQRD], tf.float32)
                w = tf.decode_raw(tfrecord_features[WINV], tf.float32)
                xbj = tf.decode_raw(tfrecord_features[XBJ], tf.float32)
                ybj = tf.decode_raw(tfrecord_features[YBJ], tf.float32)
                current = tf.decode_raw(tfrecord_features[CURRENT], tf.int32)
                int_type = tf.decode_raw(tfrecord_features[INT_TYPE], tf.int32)
                targetZ = tf.decode_raw(tfrecord_features[TARGETZ], tf.int32)
            with tf.variable_scope(self.name + '_' + HADRO_DATA):
                esum_chgdkaons = tf.decode_raw(
                    tfrecord_features[ESUM_CHGDKAONS], tf.float32
                )
                esum_chgdpions = tf.decode_raw(
                    tfrecord_features[ESUM_CHGDPIONS], tf.float32
                )
                esum_hadmultmeas = tf.decode_raw(
                    tfrecord_features[ESUM_HADMULTMEAS], tf.float32
                )
                esum_neutpions = tf.decode_raw(
                    tfrecord_features[ESUM_NEUTPIONS], tf.float32
                )
                esum_neutrons = tf.decode_raw(
                    tfrecord_features[ESUM_NEUTRONS], tf.float32
                )
                esum_others = tf.decode_raw(
                    tfrecord_features[ESUM_OTHERS], tf.float32
                )
                esum_protons = tf.decode_raw(
                    tfrecord_features[ESUM_PROTONS], tf.float32
                )
                n_chgdkaons = tf.decode_raw(
                    tfrecord_features[N_CHGDKAONS], tf.int32
                )
                n_chgdpions = tf.decode_raw(
                    tfrecord_features[N_CHGDPIONS], tf.int32
                )
                n_hadmultmeas = tf.decode_raw(
                    tfrecord_features[N_HADMULTMEAS], tf.int32
                )
                n_neutpions = tf.decode_raw(
                    tfrecord_features[N_NEUTPIONS], tf.int32
                )
                n_neutrons = tf.decode_raw(
                    tfrecord_features[N_NEUTRONS], tf.int32
                )
                n_others = tf.decode_raw(
                    tfrecord_features[N_OTHERS], tf.int32
                )
                n_protons = tf.decode_raw(
                    tfrecord_features[N_PROTONS], tf.int32
                )
        return evtids, hitimesx, hitimesu, hitimesv, pcodes, segs, zs, \
            qsqrd, w, xbj, ybj, current, int_type, targetZ, \
            esum_chgdkaons, esum_chgdpions, esum_hadmultmeas, esum_neutpions, \
            esum_neutrons, esum_others, esum_protons, \
            n_chgdkaons, n_chgdpions, n_hadmultmeas, n_neutpions, \
            n_neutrons, n_others, n_protons

    def batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_batch_gen'):
            es, x, u, v, ps, sg, zs, qsqrd, w, xbj, ybj, current, int_type, \
                targetZ, esum_chgdkaons, esum_chgdpions, esum_hadmultmeas, \
                esum_neutpions, esum_neutrons, esum_others, esum_protons, \
                n_chgdkaons, n_chgdpions, n_hadmultmeas, \
                n_neutpions, n_neutrons, n_others, n_protons = \
                self._tfrecord_to_graph_ops_et(num_epochs)
            capacity = 10 * self.batch_size
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b, qsqrd_b, w_b, \
                xbj_b, ybj_b, current_b, int_type_b, targetZ_b, \
                esum_chgdkaons_b, esum_chgdpions_b, esum_hadmultmeas_b, \
                esum_neutpions_b, esum_neutrons_b, esum_others_b, \
                esum_protons_b, \
                n_chgdkaons_b, n_chgdpions_b, n_hadmultmeas_b, \
                n_neutpions_b, n_neutrons_b, n_others_b, \
                n_protons_b = \
                tf.train.batch(
                    [es, x, u, v, ps, sg, zs, qsqrd, w, xbj, ybj, current,
                     int_type, targetZ, esum_chgdkaons, esum_chgdpions,
                     esum_hadmultmeas, esum_neutpions, esum_neutrons,
                     esum_others, esum_protons,
                     n_chgdkaons, n_chgdpions,
                     n_hadmultmeas, n_neutpions, n_neutrons,
                     n_others, n_protons],
                    batch_size=self.batch_size,
                    capacity=capacity,
                    enqueue_many=True,
                    allow_smaller_final_batch=True,
                    name=self.name+'_batch'
                )
        return self._make_mnv_hadmultkine_batch_dict(
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b, qsqrd_b, w_b, xbj_b, ybj_b,
            current_b, int_type_b, targetZ_b, esum_chgdkaons_b,
            esum_chgdpions_b, esum_hadmultmeas_b, esum_neutpions_b,
            esum_neutrons_b, esum_others_b, esum_protons_b,
            n_chgdkaons_b, n_chgdpions_b, n_hadmultmeas_b,
            n_neutpions_b, n_neutrons_b, n_others_b, n_protons_b
        )

    def shuffle_batch_generator(self, num_epochs=1):
        with tf.variable_scope(self.name+'_shufflebatch_gen'):
            es, x, u, v, ps, sg, zs, qsqrd, w, xbj, ybj, current, int_type, \
                targetZ, esum_chgdkaons, esum_chgdpions, esum_hadmultmeas, \
                esum_neutpions, esum_neutrons, esum_others, esum_protons, \
                n_chgdkaons, n_chgdpions, n_hadmultmeas, \
                n_neutpions, n_neutrons, n_others, n_protons = \
                self._tfrecord_to_graph_ops_et(num_epochs)
            min_after_dequeue = 3 * self.batch_size
            capacity = 10 * self.batch_size
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b, qsqrd_b, w_b, \
                xbj_b, ybj_b, current_b, int_type_b, targetZ_b, \
                esum_chgdkaons_b, esum_chgdpions_b, esum_hadmultmeas_b, \
                esum_neutpions_b, esum_neutrons_b, esum_others_b, \
                esum_protons_b, \
                n_chgdkaons_b, n_chgdpions_b, n_hadmultmeas_b, \
                n_neutpions_b, n_neutrons_b, n_others_b, \
                n_protons_b = \
                tf.train.shuffle_batch(
                    [es, x, u, v, ps, sg, zs, qsqrd, w, xbj, ybj, current,
                     int_type, targetZ, esum_chgdkaons, esum_chgdpions,
                     esum_hadmultmeas, esum_neutpions, esum_neutrons,
                     esum_others, esum_protons,
                     n_chgdkaons, n_chgdpions,
                     n_hadmultmeas, n_neutpions, n_neutrons,
                     n_others, n_protons],
                    batch_size=self.batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=True,
                    allow_smaller_final_batch=True,
                    name=self.name+'_shuffle_batch'
                )
        return self._make_mnv_hadmultkine_batch_dict(
            es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b, qsqrd_b, w_b, xbj_b, ybj_b,
            current_b, int_type_b, targetZ_b, esum_chgdkaons_b,
            esum_chgdpions_b, esum_hadmultmeas_b, esum_neutpions_b,
            esum_neutrons_b, esum_others_b, esum_protons_b,
            n_chgdkaons_b, n_chgdpions_b, n_hadmultmeas_b,
            n_neutpions_b, n_neutrons_b, n_others_b, n_protons_b
        )

