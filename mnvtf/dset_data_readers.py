#!/usr/bin/env python
import tensorflow as tf

from data_constants import HITIMESU, HITIMESV, HITIMESX
from data_constants import EVENTIDS, PLANECODES
from hdf5_readers import MnvHDF5Reader as HDF5Reader


class DsetMnvTFRecReaderBase(object):
    """
    Minerva data reader for TFRecord files using the dataset API.

    Argument comments:
    * name is typically 'train/test/valid/etc.'
    * n_planecodes is needed to one-hot encode the planecodes
    * compression should be 'GZIP' (or 'ZLIB?')
    * img shape is (img_h, img_w_x, img_w_uv, img_depth)
    """

    def __init__(self, args_dict):
        self.filenames_list = args_dict['FILENAMES_LIST']
        self.batch_size = args_dict['BATCH_SIZE']
        self.name = args_dict['NAME']
        self.img_shp = args_dict['IMG_SHP']
        self.n_planecodes = args_dict['N_PLANECODES']
        self.data_format = args_dict['DATA_FORMAT']
        # TODO - passed in compression is the wrong type (int)
        self.compression = args_dict['FILE_COMPRESSION']
        self.compression = 'GZIP'
        self._features_dict = None

    def _process_hitimes(self, inp, shape):
        """ Start with a (N, C, H, W) structure, -> (N, H, W, C) """
        tnsr = tf.reshape(tf.decode_raw(inp, tf.float32), shape)
        if self.data_format == 'NCHW':
            return tnsr
        elif self.data_format == 'NHWC':
            return tf.transpose(tnsr, [1, 2, 0])
        else:
            raise ValueError('Invalid data format in data reader!')

    def _decode_hitimesx(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESX],
            [self.img_shp[3], self.img_shp[0], self.img_shp[1]]
        )

    def _decode_hitimesu(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESU],
            [self.img_shp[3], self.img_shp[0], self.img_shp[2]]
        )

    def _decode_hitimesv(self, tfrecord_features):
        return self._process_hitimes(
            tfrecord_features[HITIMESV],
            [self.img_shp[3], self.img_shp[0], self.img_shp[2]]
        )

    def _decode_basic(self, tfrecord_features, field, tf_dtype):
        return tf.decode_raw(tfrecord_features[field], tf_dtype)

    def _decode_onehot(
            self, tfrecord_features, field, depth, tf_dtype=tf.int32
    ):
        v = tf.decode_raw(tfrecord_features[field], tf_dtype)
        v = tf.one_hot(indices=v, depth=depth, on_value=1, off_value=0)
        v = tf.reshape(v, [depth])
        return v

    def _decode_onehot_capped(
            self, tfrecord_features, field, depth, tf_dtype=tf.int32
    ):
        """depth becomes the cap value - should be ~ range(depth)"""
        v = tf.decode_raw(tfrecord_features[field], tf_dtype)
        m = (depth - 1) * tf.ones_like(v)
        z = tf.where(v < depth, x=v, y=m)
        z = tf.one_hot(indices=z, depth=depth, on_value=1, off_value=0)
        z = tf.reshape(z, [depth])
        return z


class DsetMnvTFRecReaderPlanecodes(DsetMnvTFRecReaderBase):

    def __init__(self, args_dict):
        super(DsetMnvTFRecReaderPlanecodes, self).__init__(args_dict)
        self.data_fields = [
            HITIMESU, HITIMESV, HITIMESX, EVENTIDS, PLANECODES
        ]
        self._features_dict = {
            f: tf.FixedLenFeature([], tf.string) for f in self.data_fields
        }

    def make_dataset(self, num_epochs=1, shuffle=False):
        def parse_fn(tfrecord):
            tfr_features = tf.parse_single_example(
                tfrecord,
                features=self._features_dict,
                name=self.name+'_data'
            )
            hitimesx = self._decode_hitimesx(tfr_features)
            hitimesu = self._decode_hitimesu(tfr_features)
            hitimesv = self._decode_hitimesv(tfr_features)
            eventids = self._decode_basic(tfr_features, EVENTIDS, tf.int64)
            planecodes = self._decode_onehot(
                tfr_features, PLANECODES, self.n_planecodes
            )
            return hitimesx, hitimesu, hitimesv, eventids, planecodes
        dataset = tf.data.TFRecordDataset(
            self.filenames_list, compression_type=self.compression
        )
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(parse_fn).prefetch(self.batch_size*10)
        dataset = dataset.batch(self.batch_size)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.batch_size*10)
        return dataset

    def batch_generator(self, num_epochs=1, shuffle=False):
        ds = self.make_dataset(num_epochs, shuffle)
        iterator = ds.make_one_shot_iterator()
        hitimesx, hitimesu, hitimesv, eventids, planecodes = \
            iterator.get_next()
        return hitimesx, hitimesu, hitimesv, eventids, planecodes

    def shuffle_batch_generator(self, num_epochs=1):
        return self.batch_generator(num_epochs, shuffle=True)


class DsetMnvHDF5ReaderBase(object):
    '''
    Minerva data reader for HDF5 files using the dataset API

    Notes:
    * initially, this API will only fully support _inference_ - the changes
    needed in the overall structure to better support training (and validation)
    are too extensive to be pursued.
    * right now, the filenames_list must be only one file long.
    '''

    def __init__(self, args_dict):
        self.filenames_list = args_dict['FILENAMES_LIST']
        self.batch_size = args_dict['BATCH_SIZE']
        self.name = args_dict['NAME']
        if len(self.filenames_list) != 1:
            raise ValueError('hdf5 datasets work with filelist of length 1.')
        self.filename = self.filenames_list[0]


class DsetMnvHDF5ReaderPlanecodes(DsetMnvHDF5ReaderBase):
    '''
    Minerva data reader for the vertex finding in the targets problem.
    '''

    def __init__(self, args_dict):
        super(DsetMnvHDF5ReaderPlanecodes, self).__init__(args_dict)

    def _make_generator_fn(self):
        '''
        make a generator function that we can query for batches
        '''
        reader = HDF5Reader(self.filename)
        reader.open()
        nevents = reader.get_nevents(group='event_data')

        def example_generator_fn():
            start_idx, stop_idx = 0, self.batch_size
            while True:
                if start_idx >= nevents:
                    reader.close()
                    return
                yield reader.get_vtxfinder_data(start_idx, stop_idx)
                start_idx, stop_idx = \
                    start_idx + self.batch_size, stop_idx + self.batch_size

        return example_generator_fn

    def make_dataset(self, num_epochs=1, shuffle=False):
        # make a generator function
        dgen = self._make_generator_fn()
        x_shape = [None, 127, 94, 2]
        uv_shape = [None, 127, 47, 2]
        evtid_shape = [None, 1]
        pcode_shape = [None, 174]
        ds = tf.data.Dataset.from_generator(
            dgen,
            (tf.float32, tf.float32, tf.float32, tf.int64, tf.int32),
            (tf.TensorShape(x_shape),
             tf.TensorShape(uv_shape),
             tf.TensorShape(uv_shape),
             tf.TensorShape(evtid_shape),
             tf.TensorShape(pcode_shape))
        )
        # we are grabbing an entire "batch", so don't call `batch()`, etc.
        # also, note, there are issues with doing more than one epoch for
        # `from_generator` - so do just one epoch at a time for now.
        ds = ds.prefetch(10)
        if shuffle:
            ds = ds.shuffle(10)

        return ds

    def batch_generator(self, num_epochs=1, shuffle=False):
        ds = self.make_dataset(num_epochs, shuffle)
        iterator = ds.make_one_shot_iterator()
        hitimesx, hitimesu, hitimesv, eventids, planecodes = \
            iterator.get_next()
        return hitimesx, hitimesu, hitimesv, eventids, planecodes

    def shuffle_batch_generator(self, num_epochs=1):
        return self.batch_generator(num_epochs, shuffle=True)
