"""
count up the total number of events in a set of TFRecord files
"""
from __future__ import print_function
from six.moves import range
import tensorflow as tf
import logging

import mnvtf.utils as utils

LOGGER = logging.getLogger(__name__)
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('data_dir', '/tmp/data',
                           """Directory where data is stored.""")
tf.app.flags.DEFINE_string('file_root', 'mnv_data_',
                           """File basename.""")
tf.app.flags.DEFINE_string('compression', '',
                           """pigz (zz) or gzip (gz).""")
tf.app.flags.DEFINE_string('data_format', 'NHWC',
                           """Tensor packing structure.""")
tf.app.flags.DEFINE_string('log_name', 'temp_log.txt',
                           """Logfile name.""")
tf.app.flags.DEFINE_string('out_pattern', 'temp_out',
                           """Logfile name.""")
tf.app.flags.DEFINE_string('tfrec_type', 'hadmultkineimgs',
                           """TFRecord file type.""")
tf.app.flags.DEFINE_string('field', 'eventids',
                           """Recorded data field.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('imgh', 127,
                            """Img height.""")
tf.app.flags.DEFINE_integer('imgw_x', 50,
                            """X-view img width.""")
tf.app.flags.DEFINE_integer('imgw_uv', 25,
                            """U/V-view img width.""")
tf.app.flags.DEFINE_integer('img_depth', 2,
                            """Img depth.""")
tf.app.flags.DEFINE_integer('n_planecodes', 67,
                            """Number of planecodes.""")


def read_all_field(datareader_dict, typ, tfrec_type, field):
    LOGGER.info('read all {} for {}...'.format(field, typ))
    out_file = FLAGS.out_pattern + typ + '_' + field + '.txt'
    tf.reset_default_graph()
    n_evt = 0

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            reader_class = utils.get_reader_class(tfrec_type)
            reader = reader_class(datareader_dict)
            # get an ordered dict
            batch_dict = reader.batch_generator(num_epochs=1)
            vals = batch_dict[field]

            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                with open(out_file, 'ab+') as f:
                    for batch_num in range(1000000):
                        vs = sess.run(vals)
                        n_evt += len(vs)
                        for v in vs:
                            f.write('{}\n'.format(v))
            except tf.errors.OutOfRangeError:
                LOGGER.info('Reading stopped - queue is empty.')
            except Exception as e:
                LOGGER.info(e)
            finally:
                coord.request_stop()
                coord.join(threads)

    LOGGER.info('found {} {} events'.format(n_evt, typ))
    utils.gz_compress(out_file)

    return n_evt


def main(argv=None):
    logfilename = FLAGS.log_name
    logging.basicConfig(
        filename=logfilename, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    runpars_dict = utils.make_run_params_dict()
    train_list, valid_list, test_list = \
        utils.get_trainvalidtest_file_lists(
            FLAGS.data_dir, FLAGS.file_root, FLAGS.compression
        )
    flist_dict = {}
    read_types_list = []
    if len(train_list) > 0:
        flist_dict['train'] = train_list
        read_types_list.append('train')
    if len(valid_list) > 0:
        flist_dict['valid'] = valid_list
        read_types_list.append('valid')
    if len(test_list) > 0:
        flist_dict['test'] = test_list
        read_types_list.append('test')

    def datareader_dict(filenames_list, name):
        img_shp = (FLAGS.imgh, FLAGS.imgw_x, FLAGS.imgw_uv, FLAGS.img_depth)
        dd = utils.make_data_reader_dict(
            filenames_list=filenames_list,
            batch_size=FLAGS.batch_size,
            name=name,
            compression=FLAGS.compression,
            img_shp=img_shp,
            data_format=FLAGS.data_format,
            n_planecodes=FLAGS.n_planecodes
        )
        return dd

    LOGGER.debug(' run_params_dict = {}'.format(repr(runpars_dict)))

    n_total = 0
    for typ in read_types_list:
        dd = datareader_dict(flist_dict[typ], typ)
        LOGGER.info(' data reader dict for {} = {}'.format(
            typ, repr(dd)
        ))
        n_total += read_all_field(dd, typ, FLAGS.tfrec_type, FLAGS.field)

    LOGGER.info('Total events = {}'.format(n_total))


if __name__ == '__main__':
    tf.app.run()
