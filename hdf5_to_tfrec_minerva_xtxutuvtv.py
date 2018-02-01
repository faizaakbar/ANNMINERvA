"""
Convert a list of hdf5s file to a list of tfrecords files of the basic types
(train, valid, test) - assumes the two-deep minerva "spacetime" hdf5 file
format.
"""
from __future__ import print_function
from collections import OrderedDict
from six.moves import range
import tensorflow as tf
import numpy as np
import sys
import os
import logging
import glob

import mnvtf.mnv_utils as mnv_utils
from mnvtf.MnvHDF5 import MnvHDF5Reader
from mnvtf.MnvDataConstants import make_mnv_data_dict
from mnvtf.MnvDataConstants import EVENT_DATA
from mnvtf.MnvDataConstants import PLANECODES, SEGMENTS

LOGGER = logging.getLogger(__name__)


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


def get_binary_data(reader, name, start_idx, stop_idx):
    """
    * reader - hdf5_reader
    * name of dset in the hdf5 file
    * indices
    returns byte data

    NOTE: we must treat the 'planecodes' dataset as special - TF has some
    issues with 16bit dtypes as of TF 1.2, so we must cast planecodes as 32-bit
    _prior_ to byte conversion.

    Note: syntax to cap a numpy array: `b[np.where(b > 5)] = 5`
    """
    dta = reader.get_data(name, start_idx, stop_idx)
    if name == PLANECODES:
        # we must cast the 16 bit values into 32 bit values
        dta = dta.astype(np.int32)
    return dta.tobytes()


def write_tfrecord(
        reader, data_dict, start_idx, stop_idx, tfrecord_file, compress_to_gz
):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    features_dict = {}
    for idx in range(start_idx, stop_idx):
        for k in data_dict:
            data_dict[k]['byte_data'] = get_binary_data(
                reader, k, idx, idx + 1
            )
            if len(data_dict[k]['byte_data']) > 0:
                features_dict[k] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[data_dict[k]['byte_data']]
                    )
                )
        example = tf.train.Example(
            features=tf.train.Features(feature=features_dict)
        )
        writer.write(example.SerializeToString())
    writer.close()

    if compress_to_gz:
        mnv_utils.gz_compress(tfrecord_file)


def test_read_tfrecord(
        tfrecord_file, hdf5_type, compression,
        img_h, imgw_x, imgw_uv, img_depth,
        n_planecodes, data_format
):
    tf.reset_default_graph()
    LOGGER.info('opening {} for reading'.format(tfrecord_file))

    dd = mnv_utils.make_data_reader_dict(
        filenames_list=[tfrecord_file],
        batch_size=64,
        name='test_read',
        compression=compression,
        img_shp=(img_h, imgw_x, imgw_uv, img_depth),
        data_format=data_format,
        n_planecodes=n_planecodes
    )
    reader_class = mnv_utils.get_reader_class(hdf5_type)
    reader = reader_class(dd)
    # get an ordered dict
    batch_dict = reader.batch_generator()

    with tf.Session() as sess:
        # have to run local variable init for `string_input_producer`
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for batch_num in range(10):
                tensor_list = sess.run(batch_dict.values())
                results = OrderedDict(zip(batch_dict.keys(), tensor_list))
                LOGGER.info('batch = {}'.format(batch_num))
                for k, v in results.items():
                    LOGGER.info('{} shape = {}'.format(k, v.shape))
                    if 'hitimes' not in k:
                        if k == PLANECODES or k == SEGMENTS:
                            LOGGER.info('  {} = {}'.format(
                                k, np.argmax(v, axis=1)
                            ))
                        else:
                            LOGGER.info('  {} = {}'.format(k, v))
        except tf.errors.OutOfRangeError:
            LOGGER.info('Reading stopped - queue is empty.')
        except Exception as e:
            LOGGER.info(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def write_all(
        n_events_per_tfrecord_triplet, max_triplets, file_num_start,
        hdf5_file, train_file_pat, valid_file_pat, test_file_pat,
        train_fraction, valid_fraction, dry_run, compress_to_gz,
        file_num_start_write, hdf5_type
):
    # todo, make this a while loop that keeps making tf record files
    # until we run out of events in the hdf5, then pass back the
    # file number we stopped on
    LOGGER.info('opening hdf5 file {} for file start number {}'.format(
        hdf5_file, file_num_start
    ))
    m = MnvHDF5Reader(hdf5_file)
    m.open()
    n_total = m.get_nevents(group=EVENT_DATA)
    slcs = slices_maker(n_total, n_events_per_tfrecord_triplet)
    n_processed = 0
    new_files = []

    for i, slc in enumerate(slcs):
        file_num = i + file_num_start
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
        LOGGER.info("slice {}, {} total events".format(i, n_slc))
        LOGGER.info(
            "slice {}, {} train events, [{}-{}): {}".format(
                i, n_train, train_start, train_stop, train_file)
        )
        LOGGER.info(
            "slice {}, {} valid events, [{}-{}): {}".format(
                i, n_valid, valid_start, valid_stop, valid_file)
        )
        LOGGER.info(
            "slice {}, {} test events, [{}-{}): {}".format(
                i, n_test, test_start, test_stop, test_file)
        )

        for filename in [train_file, valid_file, test_file]:
            if os.path.isfile(filename):
                LOGGER.info(
                    'found existing tfrecord file {}, removing...'.format(
                        filename
                    )
                )
                os.remove(filename)

        if not dry_run and file_num >= file_num_start_write:
            list_of_groups = mnv_utils.get_groups_list(hdf5_type)
            data_dict = make_mnv_data_dict(list_of_groups=list_of_groups)
            # events included are [start, stop)
            if n_train > 0:
                LOGGER.info('creating train file...')
                write_tfrecord(
                    m, data_dict, train_start, train_stop,
                    train_file, compress_to_gz
                )
                new_files.append(
                    train_file + '.gz' if compress_to_gz else train_file
                )
            if n_valid > 0:
                LOGGER.info('creating valid file...')
                write_tfrecord(
                    m, data_dict, valid_start, valid_stop,
                    valid_file, compress_to_gz
                )
                new_files.append(
                    valid_file + '.gz' if compress_to_gz else valid_file
                )
            if n_test > 0:
                LOGGER.info('creating test file...')
                write_tfrecord(
                    m, data_dict, test_start, test_stop,
                    test_file, compress_to_gz
                )
                new_files.append(
                    test_file + '.gz' if compress_to_gz else test_file
                )
        if (max_triplets > 0) and (len(new_files) / 3 >= max_triplets):
            break

        n_processed += n_slc

    LOGGER.info("Processed {} events, finished with file number {}".format(
        n_processed, (file_num - 1)
    ))
    m.close()
    return file_num, new_files


def read_all(
        files_written, hdf5_type, dry_run, compressed, imgw_x, imgw_uv,
        n_planecodes, img_h=127, img_depth=2, data_format='NHWC'
):
    LOGGER.info('reading files...')

    for filename in files_written:
        if os.path.isfile(filename):
            LOGGER.info(
                'found existing tfrecord file {} with size {}...'.format(
                    filename, os.stat(filename).st_size
                )
            )
            if not dry_run:
                compression = 'gz' if compressed else ''
                test_read_tfrecord(
                    filename, hdf5_type, compression,
                    img_h, imgw_x, imgw_uv, img_depth,
                    n_planecodes, data_format
                )


if __name__ == '__main__':

    def arg_list_split(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-l', '--file_list', dest='file_list',
                      help='HDF5 file list (csv, full paths)',
                      metavar='FILELIST', type='string', action='callback',
                      callback=arg_list_split)
    parser.add_option('-p', '--file_pattern', dest='file_pattern',
                      help='File pattern', metavar='FILEPATTERN',
                      default=None, type='string')
    parser.add_option('-i', '--in_dir', dest='in_dir',
                      help='In directory (for file patterns)',
                      metavar='IN_DIR', default=None, type='string')
    parser.add_option('-o', '--out_dir', dest='out_dir',
                      help='Out directory', metavar='OUT_DIR', default=None,
                      type='string')
    parser.add_option('-n', '--nevents', dest='n_events', default=0,
                      help='Number of events per file', metavar='N_EVENTS',
                      type='int')
    parser.add_option('-s', '--start_idx', dest='start_idx', default=0,
                      help='Start writing to disk at index value',
                      metavar='START_IDX', type='int')
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
    parser.add_option('--imgw_x', dest='imgw_x', default=94,
                      help='Image width-x', metavar='IMGWX',
                      type='int')
    parser.add_option('--imgw_uv', dest='imgw_uv', default=47,
                      help='Image width-uv', metavar='IMGWUV',
                      type='int')
    parser.add_option('--n_planecodes', dest='n_planecodes', default=173,
                      help='Number (count) of planecodes', metavar='NPCODES',
                      type='int')
    parser.add_option('--hdf5_type', dest='hdf5_type',
                      default=None, help='HDF5 groups set',
                      metavar='HDF5TYPE', type='string')

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

    logfilename = options.logfilename or \
        'log_hdf5_to_tfrec_minerva_xtxutuvtv.txt'
    logging.basicConfig(
        filename=logfilename, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    files = options.file_list or []
    for ext in ['*.hdf5', '*.h5']:
        extra_files = glob.glob(
                options.in_dir + '/' + options.file_pattern + ext
        )
        files.extend(extra_files)
    # kill any repeats
    files = list(set(files))
    files.sort()

    LOGGER.info("Datasets:")
    dataset_statsinfo = 0
    for hdf5_file in files:
        fsize = os.stat(hdf5_file).st_size
        dataset_statsinfo += os.stat(hdf5_file).st_size
        LOGGER.info(" {}, size = {}".format(hdf5_file, fsize))
    LOGGER.info("Total dataset size: {}".format(dataset_statsinfo))

    # loop over list of hdf5 files (glob for patterns?), for each file, create
    # tfrecord files of specified size, putting remainders in new files.
    file_num = 0
    for i, hdf5_file in enumerate(files):
        # NOTE - this includes the path, so don't put '.' or '..' in the dirs!
        base_name = hdf5_file.split('/')[-1]
        base_name = options.out_dir + '/' + base_name.split('.')[0]
        # create file patterns to fill tfrecord files by number
        train_file_pat = base_name + '_%06d_train.tfrecord'
        valid_file_pat = base_name + '_%06d_valid.tfrecord'
        test_file_pat = base_name + '_%06d_test.tfrecord'

        # TODO: add a 'mask' that properly trims imgw_x/uv, caps the pcodes
        out_num, files_written = write_all(
            n_events_per_tfrecord_triplet=options.n_events,
            max_triplets=options.max_triplets, file_num_start=file_num,
            hdf5_file=hdf5_file, train_file_pat=train_file_pat,
            valid_file_pat=valid_file_pat, test_file_pat=test_file_pat,
            train_fraction=options.train_fraction,
            valid_fraction=options.valid_fraction,
            dry_run=options.dry_run, compress_to_gz=options.compress_to_gz,
            file_num_start_write=options.start_idx,
            hdf5_type=options.hdf5_type
        )
        file_num = out_num

        if options.do_test:
            read_all(
                files_written,
                options.hdf5_type,
                options.dry_run,
                options.compress_to_gz,
                options.imgw_x,
                options.imgw_uv,
                options.n_planecodes
            )

        if (options.max_triplets > 0) and \
           (len(files_written) / 3 > options.max_triplets):
            break
