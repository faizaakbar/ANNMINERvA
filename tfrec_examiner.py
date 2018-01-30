"""
"""
from __future__ import print_function
from six.moves import range
import tensorflow as tf
import logging

import mnvtf.mnv_utils as mnv_utils
from mnvtf.MnvDataConstants import EVENTIDS

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


def read_all_evtids(datareader_dict, typ, tfrec_type):
    LOGGER.info('read all eventids for {}...'.format(typ))
    out_file = FLAGS.out_pattern + typ + '.txt'
    tf.reset_default_graph()
    n_evt = 0

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            reader_class = mnv_utils.get_reader_class(tfrec_type)
            reader = reader_class(datareader_dict)
            # get an ordered dict
            batch_dict = reader.batch_generator(num_epochs=1)
            eventids = batch_dict[EVENTIDS]

            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                with open(out_file, 'ab+') as f:
                    for batch_num in range(1000000):
                        evtids = sess.run(eventids)
                        n_evt += len(evtids)
                        for evtid in evtids:
                            f.write('{}\n'.format(evtid))
            except tf.errors.OutOfRangeError:
                LOGGER.info('Reading stopped - queue is empty.')
            except Exception as e:
                LOGGER.info(e)
            finally:
                coord.request_stop()
                coord.join(threads)

    LOGGER.info('found {} {} events'.format(n_evt, typ))
    mnv_utils.gz_compress(out_file)


def main(argv=None):
    logfilename = FLAGS.log_name
    logging.basicConfig(
        filename=logfilename, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    runpars_dict = mnv_utils.make_run_params_dict()
    train_list, valid_list, test_list = \
        mnv_utils.get_trainvalidtest_file_lists(
            FLAGS.data_dir, FLAGS.file_root, FLAGS.compression
        )
    flist_dict = {}
    flist_dict['train'] = train_list
    flist_dict['valid'] = valid_list
    flist_dict['test'] = test_list

    def datareader_dict(filenames_list, name):
        img_shp = (FLAGS.imgh, FLAGS.imgw_x, FLAGS.imgw_uv, FLAGS.img_depth)
        dd = mnv_utils.make_data_reader_dict(
            filenames_list=filenames_list,
            batch_size=FLAGS.batch_size,
            name=name,
            compression=FLAGS.compression,
            img_shp=img_shp,
            data_format=FLAGS.data_format,
            n_planecodes=FLAGS.n_planecodes
        )
        return dd

    LOGGER.info(' run_params_dict = {}'.format(repr(runpars_dict)))

    for typ in ['train', 'valid', 'test']:
        dd = datareader_dict(flist_dict[typ], typ)
        LOGGER.info(' data reader dict for {} = {}'.format(
            typ, repr(dd)
        ))
        read_all_evtids(dd, typ, FLAGS.tfrec_type)


if __name__ == '__main__':
    tf.app.run()