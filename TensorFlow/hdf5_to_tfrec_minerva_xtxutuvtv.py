"""
convert a list of hdf5s file to a list of tfrecords files of the basic types
(train, valid, test) - assumes the two-deep minerva "spacetime" hdf5 file
format.

Usage:
    python hdf5_to_tfrec_minerva_xtxutuvtv.py -l hdf5_file_list \
         -n n_events_per_tfrecord_triplet \
         -m max_number_of_tfrecord_triplets \
         [-t train fraction (0.83)] [-v valid fraction (0.08)] \
         [-r (do a test read - default is False)] \
         [-g logfilename]
         [-d dry_run_for_write (default is False)]
         [-c compress_to_gz (default is False)]
"""
from __future__ import print_function
from six.moves import range
import h5py
import tensorflow as tf
import numpy as np
import sys
import os
import logging
import gzip
import shutil
import glob


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


def slices_maker(n, slice_size=100000):
    """
    make "slices" of size `slice_size` from a file of `n` events
    (so, [0, slice_size), [slice_size, 2 * slice_size), etc.)
    """
    if n < slice_size:
        return [(0, n)]

    remainder = n % slice_size
    n = n - remainder
    nblocks = n // slice_size
    counter = 0
    slices = []
    for i in range(nblocks):
        end = counter + slice_size
        slices.append((counter, end))
        counter += slice_size

    if remainder != 0:
        slices.append((counter, counter + remainder))

    return slices


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


def gz_compress(infile):
    outfile = infile + '.gz'
    with open(infile, 'rb') as f_in, gzip.open(outfile, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    if os.path.isfile(outfile) and (os.stat(outfile).st_size > 0):
        os.remove(infile)
    else:
        raise IOError('Compressed file not produced!')


def write_to_tfrecord(data_dict, tfrecord_file, compress_to_gz):
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

    if compress_to_gz:
        gz_compress(tfrecord_file)


def write_tfrecord(
        reader, data_dict, start_idx, stop_idx, tfrecord_file, compress_to_gz
):
    for k in data_dict:
        data_dict[k]['byte_data'] = get_binary_data(
            reader, k, start_idx, stop_idx
        )
    write_to_tfrecord(data_dict, tfrecord_file, compress_to_gz)


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


def test_read_tfrecord(tfrecord_file, compressed):
    logger.info('opening {} for reading'.format(tfrecord_file))
    batch_dict = batch_generator([tfrecord_file], compressed)
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
                logger.info('batch = {}'.format(batch_num))
                logger.info('evtids shape = {}'.format(evtids.shape))
                logger.info('hitimes x shape = {}'.format(hitsx.shape))
                logger.info('hitimes u shape = {}'.format(hitsu.shape))
                logger.info('hitimes v shape = {}'.format(hitsv.shape))
                logger.info('planecodes shape = {}'.format(pcodes.shape))
                logger.info('  planecodes = {}'.format(
                    np.argmax(pcodes, axis=1)
                ))
                logger.info('segments shape = {}'.format(segs.shape))
                logger.info('  segments = {}'.format(
                    np.argmax(segs, axis=1)
                ))
                logger.info('zs shape = {}'.format(zs.shape))
        except tf.errors.OutOfRangeError:
            logger.info('Reading stopped - queue is empty.')
        except Exception as e:
            logger.info(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def write_all(
        n_events_per_tfrecord_triplet, max_triplets, file_num_start,
        hdf5_file, train_file_pat, valid_file_pat, test_file_pat,
        train_fraction, valid_fraction, dry_run, compress_to_gz
):
    # todo, make this a while loop that keeps making tf record files
    # until we run out of events in the hdf5, then pass back the
    # file number we stopped on
    logger.info('opening hdf5 file {} for file start number {}'.format(
        hdf5_file, file_num_start
    ))
    m = minerva_hdf5_reader(hdf5_file)
    m.open()
    n_total = m.get_nevents()
    slcs = slices_maker(n_total, n_events_per_tfrecord_triplet)
    n_processed = 0
    new_files = []

    for i, slc in enumerate(slcs):
        file_num = i + file_num_start
        if (max_triplets > 0) and ((file_num + 1) > max_triplets):
            break
        n_slc = slc[-1] - slc[0]
        n_train = int(n_slc * train_fraction)
        n_valid = int(n_slc * valid_fraction)
        n_test = n_slc - n_train - n_valid
        train_start, train_stop = n_processed, n_processed + n_train
        valid_start, valid_stop = train_stop, train_stop + n_valid
        test_start, test_stop = valid_stop, valid_stop + n_test
        train_file = train_file_pat % file_num
        valid_file = valid_file_pat % file_num
        test_file = test_file_pat % file_num
        logger.info("slice {}, {} total events".format(i, n_slc))
        logger.info(
            "slice {}, {} train events, [{}-{}): {}".format(
                i, n_train, train_start, train_stop, train_file)
        )
        logger.info(
            "slice {}, {} valid events, [{}-{}): {}".format(
                i, n_valid, valid_start, valid_stop, valid_file)
        )
        logger.info(
            "slice {}, {} test events, [{}-{}): {}".format(
                i, n_test, test_start, test_stop, test_file)
        )

        for filename in [train_file, valid_file, test_file]:
            if os.path.isfile(filename):
                logger.info(
                    'found existing tfrecord file {}, removing...'.format(
                        filename
                    )
                )
                os.remove(filename)

        if not dry_run:
            data_dict = make_mnv_data_dict()
            # events included are [start, stop)
            if n_train > 0:
                logger.info('creating train file...')
                write_tfrecord(
                    m, data_dict, train_start, train_stop,
                    train_file, compress_to_gz
                )
                new_files.append(
                    train_file + '.gz' if compress_to_gz else train_file
                )
            if n_valid > 0:
                logger.info('creating valid file...')
                write_tfrecord(
                    m, data_dict, valid_start, valid_stop,
                    valid_file, compress_to_gz
                )
                new_files.append(
                    valid_file + '.gz' if compress_to_gz else valid_file
                )
            if n_test > 0:
                logger.info('creating test file...')
                write_tfrecord(
                    m, data_dict, test_start, test_stop,
                    test_file, compress_to_gz
                )
                new_files.append(
                    test_file + '.gz' if compress_to_gz else test_file
                )
        n_processed += n_slc

    logger.info("Processed {} events, finished with file number {}".format(
        n_processed, (file_num - 1)
    ))
    m.close()
    return file_num, new_files


def read_all(files_written, dry_run, compressed):
    logger.info('reading files...')
        
    for filename in files_written:
        if os.path.isfile(filename):
            logger.info(
                'found existing tfrecord file {} with size {}...'.format(
                    filename, os.stat(filename).st_size
                )
            )
            if not dry_run:
                test_read_tfrecord(filename, compressed)


if __name__ == '__main__':

    def arg_list_split(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-l', '--file_list', dest='file_list',
                      help='HDF5 file list (csv)', metavar='FILELIST',
                      type='string', action='callback',
                      callback=arg_list_split)
    parser.add_option('-p', '--file_pattern', dest='file_pattern',
                      help='File pattern', metavar='FILEPATTERN',
                      default=None, type='string')
    parser.add_option('-n', '--nevents', dest='n_events', default=0,
                      help='Number of events per file', metavar='N_EVENTS',
                      type='int')
    parser.add_option('-m', '--max_triplets', dest='max_triplets', default=0,
                      help='Max number of each file type',
                      metavar='MAX_TRIPLETS', type='int')
    parser.add_option('-r', '--test_read', dest='do_test', default=False,
                      help='Test read', metavar='DO_TEST',
                      action='store_true')
    parser.add_option('-d', '--dry_run', dest='dry_run', default=False,
                      help='Dry run for write', metavar='DRY_RUN',
                      action='store_true')
    parser.add_option('-c', '--compress_to_gz', dest='compress_to_gz',
                      default=False, help='Gzip compression',
                      metavar='COMPRESS_TO_GZ', action='store_true')
    parser.add_option('-t', '--train_fraction', dest='train_fraction',
                      default=0.88, help='Train fraction',
                      metavar='TRAIN_FRAC', type='float')
    parser.add_option('-v', '--valid_fraction', dest='valid_fraction',
                      default=0.09, help='Valid fraction',
                      metavar='VALID_FRAC', type='float')
    parser.add_option('-g', '--logfile', dest='logfilename',
                      help='Log file name', metavar='LOGFILENAME',
                      default=None, type='string')

    (options, args) = parser.parse_args()

    # TODO - add `file_pattern` option (make it work)
    if (not options.file_list) and (not options.file_pattern):
        print("\nSpecify file list or file pattern:\n\n")
        print(__doc__)
        sys.exit(1)

    if (options.train_fraction + options.valid_fraction) > 1.001:
        print("\nTraining and validation fractions sum > 1!")
        print(__doc__)
        sys.exit(1)

    logfilename = options.logfilename or 'hdf5_to_tfrec_minerva_xtxutuvtv.log'
    logging.basicConfig(
        filename=logfilename, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting...")
    logger.info(__file__)

    files = options.file_list or []
    extra_files = glob.glob(options.file_pattern + '*.hdf5')
    files.extend(extra_files)
    extra_files = glob.glob(options.file_pattern + '*.h5')
    files.extend(extra_files)
    # kill any repeats
    files = list(set(files))
    files.sort()

    logger.info("Datasets:")
    dataset_statsinfo = 0
    for hdf5_file in files:
        fsize = os.stat(hdf5_file).st_size
        dataset_statsinfo += os.stat(hdf5_file).st_size
        logger.info(" {}, size = {}".format(hdf5_file, fsize))
    logger.info("Total dataset size: {}".format(dataset_statsinfo))

    # loop over list of hdf5 files (glob for patterns?), for each file, create
    # tfrecord files of specified size, putting remainders in new files.
    file_num = 0
    for i, hdf5_file in enumerate(files):
        base_name = hdf5_file.split('.')[0]
        # create file patterns to fill tfrecord files by number
        train_file_pat = base_name + '_%06d' + '_train.tfrecord'
        valid_file_pat = base_name + '_%06d' + '_valid.tfrecord'
        test_file_pat = base_name + '_%06d' + '_test.tfrecord'

        out_num, files_written = write_all(
            n_events_per_tfrecord_triplet=options.n_events,
            max_triplets=options.max_triplets, file_num_start=file_num,
            hdf5_file=hdf5_file, train_file_pat=train_file_pat,
            valid_file_pat=valid_file_pat, test_file_pat=test_file_pat,
            train_fraction=options.train_fraction,
            valid_fraction=options.valid_fraction,
            dry_run=options.dry_run, compress_to_gz=options.compress_to_gz
        )
        file_num = out_num

        if options.do_test:
            read_all(files_written, options.dry_run, options.compress_to_gz)
