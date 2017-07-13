"""
convert an hdf5 file to tfrecords (train, valid, test) - assumes the two-deep
minerva "spacetime" hdf5 file format.

Usage:
    python hdf5_to_tfrec_minerva_xtxutuvtv.py -f hdf5_file -n n_events \
         [-t train fraction (0.83)] [-v valid fraction (0.08)] [-r]
"""
from __future__ import print_function
from six.moves import range
import h5py
import tensorflow as tf
import numpy as np
import sys
import os


class minerva_hdf5_reader:
    """
    the `minerva_hdf5_reader` will return numpy ndarrays of data for given
    ranges. user should call `open()` and `close()` to start/finish.
    """
    def __init__(self, hdf5_file):
        self.file = hdf5_file
        self._f = None

    def open(self):
        self._f = h5py.File(self.file, 'r')

    def close(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_data(self, name, start_idx, stop_idx):
        return self._f[name][start_idx: stop_idx]

    def get_nevents(self):
        sizes = [self._f[d].shape[0] for d in self._f]
        if min(sizes) != max(sizes):
            raise ValueError("All dsets must have the same size!")
        return sizes[0]


def make_mnv_data_dict():
    # eventids are really uint64, planecodes are really uint16
    data_list = [
        ('eventids', tf.int64),
        ('hitimes-u', tf.float32),
        ('hitimes-v', tf.float32),
        ('hitimes-x', tf.float32),
        ('planecodes', tf.int16),
        ('segments', tf.uint8),
        ('zs', tf.float32)
    ]
    mnv_data = {}
    for datum in data_list:
        mnv_data[datum[0]] = {}
        mnv_data[datum[0]]['dtype'] = datum[1]
        mnv_data[datum[0]]['byte_data'] = None
    
    return mnv_data


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


def get_binary_data(reader, name, start_idx, stop_idx):
    """
    * reader - hdf5_reader
    * name of dset in the hdf5 file
    * indices
    returns byte data
    """
    dta = reader.get_data(name, start_idx, stop_idx)
    return dta.tobytes()


def write_to_tfrecord(data_dict, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    features_dict = {}
    for k in data_dict.keys():
        features_dict[k] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data_dict[k]['byte_data']])
        )
    example = tf.train.Example(
        features=tf.train.Features(feature=features_dict)
    )
    writer.write(example.SerializeToString())
    writer.close()


def write_tfrecord(reader, data_dict, start_idx, stop_idx, tfrecord_file):
    for k in data_dict:
        data_dict[k]['byte_data'] = get_binary_data(
            reader, k, start_idx, stop_idx
        )
    write_to_tfrecord(data_dict, tfrecord_file)


def tfrecord_to_graph_ops_xtxutuvtv(filenames):
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
    segs = tf.cast(segs, tf.int32)
    segs = tf.one_hot(indices=segs, depth=11, on_value=1, off_value=0)
    zs = tf.decode_raw(tfrecord_features['zs'], tf.float32)
    return evtids, hitimesx, hitimesu, hitimesv, pcodes, segs, zs


def batch_generator(tfrecord_filelist, num_epochs=1):
    es, x, u, v, ps, sg, zs = tfrecord_to_graph_ops_xtxutuvtv(
        tfrecord_filelist
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


def test_read_tfrecord(tfrecord_file):
    batch_dict = batch_generator([tfrecord_file])
    with tf.Session() as sess:
        # have to run local variable init for `string_input_producer`
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for batch_num in range(1000000):
                evtids, hitsx, hitsu, hitsv, pcodes, segs, zs = sess.run([
                    batch_dict['eventids'],
                    batch_dict['hitimes-x'],
                    batch_dict['hitimes-u'],
                    batch_dict['hitimes-v'],
                    batch_dict['planecodes'],
                    batch_dict['segments'],
                    batch_dict['zs'],
                ])
                print('batch =', batch_num)
                print('evtids shape =', evtids.shape)
                print('hitimes x shape =', hitsx.shape)
                print('hitimes u shape =', hitsu.shape)
                print('hitimes v shape =', hitsv.shape)
                print('planecodes shape =', pcodes.shape)
                print('  planecodes =', np.argmax(pcodes, axis=1))
                print('segments shape =', segs.shape)
                print('  segments =', np.argmax(segs, axis=1))
                print('zs shape =', zs.shape)
        except tf.errors.OutOfRangeError:
            print('Reading stopped - queue is empty.')
        except Exception as e:
            print(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def write_all(
        n_events, hdf5_file, train_file, valid_file, test_file,
        train_fraction, valid_fraction
):
    m = minerva_hdf5_reader(hdf5_file)
    m.open()
    if n_events <= 0:
        n_total = m.get_nevents()
    else:
        n_total = n_events
    n_train = int(n_total * train_fraction)
    n_valid = int(n_total * valid_fraction)
    n_test = n_total - n_train - n_valid
    print("{} total events".format(n_total))
    print("{} train events".format(n_train))
    print("{} valid events".format(n_valid))
    print("{} test events".format(n_test))

    data_dict = make_mnv_data_dict()
    # events included are [start, stop)
    if n_train > 0:
        print('creating train file...')
        write_tfrecord(m, data_dict, 0, n_train, train_file)
    if n_valid > 0:
        print('creating valid file...')
        write_tfrecord(m, data_dict, n_train, n_train + n_valid, valid_file)
    if n_test > 0:
        print('creating test file...')
        write_tfrecord(m, data_dict, n_train + n_valid, n_total, test_file)

    m.close()


def read_all(train_file, valid_file, test_file):
    print('reading train file...')
    test_read_tfrecord(train_file)
    print('reading valid file...')
    test_read_tfrecord(valid_file)
    print('reading test file...')
    test_read_tfrecord(test_file)


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-f', '--file', dest='filename',
                      help='Dset file name', metavar='FILENAME',
                      default=None, type='string')
    parser.add_option('-n', '--nevents', dest='n_events', default=0,
                      help='Number of events', metavar='N_EVENTS',
                      type='int')
    parser.add_option('-r', '--test_read', dest='do_test', default=False,
                      help='Test read', metavar='DO_TEST',
                      action='store_true')
    parser.add_option('-t', '--train_fraction', dest='train_fraction',
                      default=0.88, help='Train fraction',
                      metavar='TRAIN_FRAC', type='float')
    parser.add_option('-v', '--valid_fraction', dest='valid_fraction',
                      default=0.09, help='Valid fraction',
                      metavar='VALID_FRAC', type='float')

    (options, args) = parser.parse_args()

    if not options.filename:
        print("\nSpecify file (-f):\n\n")
        print(__doc__)
        sys.exit(1)

    if (options.train_fraction + options.valid_fraction) > 1.001:
        print("\nTraining and validation fractions sum > 1!")
        print(__doc__)
        sys.exit(1)

    hdf5_file = options.filename
    base_name = hdf5_file.split('.')[0]
    train_file = base_name + '_train.tfrecord'
    valid_file = base_name + '_valid.tfrecord'
    test_file = base_name + '_test.tfrecord'
    for filename in [train_file, valid_file, test_file]:
        if os.path.isfile(filename):
            print('found existing tfrecord file {}, removing...'.format(
                filename
            ))
            os.remove(filename)

    write_all(
        options.n_events,
        hdf5_file, train_file, valid_file, test_file,
        options.train_fraction, options.valid_fraction
    )
    if options.do_test:
        read_all(train_file, valid_file, test_file)
