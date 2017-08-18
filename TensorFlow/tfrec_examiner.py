"""
"""
from __future__ import print_function
from six.moves import range
import tensorflow as tf
import numpy as np
import sys
import os
import logging
import glob

LOGGER = logging.getLogger(__name__)


def make_mnv_vertex_finder_batch_dict(
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


def tfrecord_to_graph_ops_xtxutuvtv(filenames, compressed):
    def proces_hitimes(inp, shape):
        """
        *Note* - Minerva HDF5's are packed (N, C, H, W), so we must transpose
        them to (N, H, W, C) here.
        """
        hitimes = tf.decode_raw(inp, tf.float32)
        hitimes = tf.reshape(hitimes, shape)
        hitimes = tf.transpose(hitimes, [0, 2, 3, 1])
        return hitimes

    file_queue = tf.train.string_input_producer(
        filenames, name='file_queue', num_epochs=1
    )
    if compressed:
        compression_type = tf.python_io.TFRecordCompressionType.GZIP
    else:
        compression_type = tf.python_io.TFRecordCompressionType.NONE
    reader = tf.TFRecordReader(
        options=tf.python_io.TFRecordOptions(
            compression_type=compression_type
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
    segs = tf.cast(segs, tf.int32)
    segs = tf.one_hot(indices=segs, depth=11, on_value=1, off_value=0)
    zs = tf.decode_raw(tfrecord_features['zs'], tf.float32)
    return evtids, hitimesx, hitimesu, hitimesv, pcodes, segs, zs


def batch_generator(tfrecord_filelist, compressed, num_epochs=1):
    es, x, u, v, ps, sg, zs = tfrecord_to_graph_ops_xtxutuvtv(
        tfrecord_filelist, compressed
    )
    capacity = 10 * 100
    es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b = tf.train.batch(
        [es, x, u, v, ps, sg, zs],
        batch_size=100,
        capacity=capacity,
        enqueue_many=True,
        allow_smaller_final_batch=True,
        name='generator_batch'
    )
    return make_mnv_vertex_finder_batch_dict(
        es_b, x_b, u_b, v_b, ps_b, sg_b, zs_b
    )


def test_read_tfrecord_all_evtids(tfrecord_file, outfile, compressed):
    LOGGER.info('opening {} for reading'.format(tfrecord_file))
    batch_dict = batch_generator([tfrecord_file], compressed)
    with tf.Session() as sess:
        # have to run local variable init for `string_input_producer`
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            with open(outfile, 'a') as f:
                LOGGER.info(' opened {} for eventids'.format(outfile))
                for batch_num in range(1000000):
                    evtids, segs = sess.run([
                        batch_dict['eventids'],
                        batch_dict['segments']
                    ])
                    decoded_segments = np.argmax(segs, axis=1)
                    zppd = zip(evtids, decoded_segments)
                    for record in zppd:
                        f.write('{}, {}\n'.format(record[0], record[1]))
        except tf.errors.OutOfRangeError:
            LOGGER.info('Reading stopped - queue is empty.')
        except Exception as e:
            LOGGER.info(e)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    def arg_list_split(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-p', '--file_pattern', dest='file_pattern',
                      help='File pattern', metavar='FILEPATTERN',
                      default=None, type='string')
    parser.add_option('-d', '--dir', dest='dir',
                      help='Directory (for file patterns)',
                      metavar='DIR', default=None, type='string')
    parser.add_option('-c', '--compressed_to_gz', dest='compressed_to_gz',
                      default=False, help='Gzip compressed',
                      metavar='COMPRESSED_TO_GZ', action='store_true')
    parser.add_option('-g', '--logfile', dest='logfilename',
                      help='Log file name', metavar='LOGFILENAME',
                      default=None, type='string')

    (options, args) = parser.parse_args()

    if not options.file_pattern:
        print("\nSpecify file list or file pattern:\n\n")
        print(__doc__)
        sys.exit(1)

    logfilename = options.logfilename or \
        'log_tfrec_examiner.txt'
    logging.basicConfig(
        filename=logfilename, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    ext = '*.tfrecord.gz' if options.compressed_to_gz else '*.tfrecord'
    files = glob.glob(
        options.dir + '/' + options.file_pattern + ext
    )
    # kill any repeats
    files = list(set(files))
    files.sort()

    LOGGER.info("Datasets:")
    dataset_statsinfo = 0
    for f in files:
        fsize = os.stat(f).st_size
        dataset_statsinfo += os.stat(f).st_size
        LOGGER.info(" {}, size = {}".format(f, fsize))
    LOGGER.info("Total dataset size: {}".format(dataset_statsinfo))

    out_file_pat = options.dir + '/' + options.file_pattern + "%06d" + '.txt'
    for i, f in enumerate(files):
        out_file = out_file_pat % i
        test_read_tfrecord_all_evtids(f, out_file, options.compressed_to_gz)
