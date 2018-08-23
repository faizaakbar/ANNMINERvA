#!/usr/bin/env python
import logging
import argparse

import tensorflow as tf
import mnvtf.utils as utils


logfilename = 'log_' + __file__.split('.')[0] + '.txt'
logging.basicConfig(
    filename=logfilename, level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Starting...")
logger.info(__file__)


def get_test_hdf5():
    '''find a "modern" vertex-finding hdf5 file'''
    import os
    DDIR = os.environ['HOME'] + '/Dropbox/Data/RandomData/hdf5'
    TFILE = DDIR + '/hadmultkineimgs_mnvvtx.hdf5'
    return TFILE


def get_test_tfrec():
    '''find a "modern" vertex-finding tfrecord file'''
    import os
    DDIR = os.environ['HOME'] + '/Dropbox/Data/RandomData/TensorFlow'
    TFILE = DDIR + '/hadmultkineimgs_mnvvtx.tfrecord.gz'
    return TFILE


def datareader_dict(filenames_list, name='test'):
    img_shp = (127, 94, 47, 2)  # h, w_x, w_uv, depth
    dd = utils.make_data_reader_dict(
        filenames_list=filenames_list,
        batch_size=10,
        name=name,
        compression='gz',
        img_shp=img_shp,
        data_format='NHWC',
        n_planecodes=174
    )
    return dd


def test_graph_one_shot_iterator(generator):
    X, U, V, eventids, targets = generator(num_epochs=1)
    with tf.Session() as sess:
        counter = 0
        try:
            while True:
                x, u, v, evts, tgs = sess.run([
                    X, U, V, eventids, targets
                ])
                logger.info('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                    x.shape, x.dtype,
                    u.shape, u.dtype,
                    v.shape, v.dtype,
                    evts.shape, evts.dtype,
                    tgs.shape, tgs.dtype,
                ))
                counter += 1
                if counter > 100:
                    break
        except tf.errors.OutOfRangeError:
            print('end of dataset at counter = {}'.format(counter))
        except Exception as e:
            print(e)


def test_hdf5(batch_size):
    logger.info('hdf5s')
    from mnvtf.dset_data_readers import DsetMnvHDF5ReaderPlanecodes

    filenames_list = [get_test_hdf5()]
    dd = datareader_dict(filenames_list)
    reader = DsetMnvHDF5ReaderPlanecodes(dd)
    test_graph_one_shot_iterator(reader.batch_generator)


def test_tfrecord(batch_size):
    logger.info('checking tfrecords')
    from mnvtf.dset_data_readers import DsetMnvTFRecReaderPlanecodes

    filenames_list = [get_test_tfrec()]
    dd = datareader_dict(filenames_list)
    reader = DsetMnvTFRecReaderPlanecodes(dd)
    test_graph_one_shot_iterator(reader.batch_generator)


def main(hdf5, batch_size):
    if hdf5:
        test_hdf5(batch_size)
    else:
        test_tfrecord(batch_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hdf5', default=False, action='store_true',
        help='Use HDF5 instead of TFRecord'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Batch size'
    )
    args = parser.parse_args()

    main(**vars(args))
