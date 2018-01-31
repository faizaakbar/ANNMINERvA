"""
minerva test. NOTE, this script assumes data is available in the form of
TFRecords, with groups of files with extensions like `..._train.tfrecord`,
`..._valid.tfrecord`, and `..._test.tfrecord` (possibly with .zz or .gz
compression extension markers).
"""
from __future__ import print_function

import tensorflow as tf

from mnvtf.MnvModelsTricolumnar import TriColSTEpsilon
from mnvtf.MnvModelsTricolumnar import make_default_convpooldict
from mnvtf.MnvTFRunners import MnvTFRunnerCategorical
from mnvtf.MnvDataReaders import MnvDataReaderVertexST
import mnvtf.mnv_utils as mnv_utils

MNV_TYPE = 'st_epsilon'
FLAGS = tf.app.flags.FLAGS

#
# input data specification
# NOTE - data_dir and file_root can be commma-separated lists and still be
# handled correctly here, etc.
#
tf.app.flags.DEFINE_string('data_dir', '/tmp/data',
                           """Directory where data is stored.""")
tf.app.flags.DEFINE_string('file_root', 'mnv_data_',
                           """File basename.""")
tf.app.flags.DEFINE_string('compression', '',
                           """pigz (zz) or gzip (gz).""")
tf.app.flags.DEFINE_string('data_format', 'NHWC',
                           """Tensor packing structure.""")
tf.app.flags.DEFINE_string('tfrec_type', 'hadmultkineimgs',
                           """TFRecord file type.""")
#
# logs and special outputs (models, predictions, etc.)
#
tf.app.flags.DEFINE_string('log_name', 'temp_log.txt',
                           """Logfile name.""")
tf.app.flags.DEFINE_string('log_level', 'INFO',
                           """Logging level (INFO/DEBUG/etc.).""")
tf.app.flags.DEFINE_boolean('do_log_devices', False,
                            """Log device placement.""")
tf.app.flags.DEFINE_string('model_dir', '/tmp/minerva/models',
                           """Directory where models are stored.""")
tf.app.flags.DEFINE_string('pred_store_name', 'temp_store',
                           """Predictions store name.""")
#
# training description
#
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('save_every_n_batch', 500,
                            """Save every N batches.""")
tf.app.flags.DEFINE_string('strategy', 'Adam',
                           """Optimizer strategy.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          """Learning rate.""")
tf.app.flags.DEFINE_boolean('do_batch_norm', False,
                            """Do batch normalization.""")
# TODO - add l2 reg params, etc.
#
# classification goal specification
#
tf.app.flags.DEFINE_string('targets_label', 'planecodes',
                           """Name of targets tensor.""")
tf.app.flags.DEFINE_integer('n_classes', 173,
                            """Name of target classes.""")
#
# img and data features
#
tf.app.flags.DEFINE_integer('imgh', 127,
                            """Img height.""")
tf.app.flags.DEFINE_integer('imgw_x', 94,
                            """X-view img width.""")
tf.app.flags.DEFINE_integer('imgw_uv', 47,
                            """U/V-view img width.""")
tf.app.flags.DEFINE_integer('img_depth', 2,
                            """Img depth.""")
tf.app.flags.DEFINE_integer('n_planecodes', 173,
                            """Number of planecodes.""")
#
# action flags
#
tf.app.flags.DEFINE_boolean('do_training', True,
                            """Perform training ops.""")
tf.app.flags.DEFINE_boolean('do_validation', True,
                            """Perform validation ops.""")
tf.app.flags.DEFINE_boolean('do_testing', True,
                            """Perform testing ops.""")
tf.app.flags.DEFINE_boolean('do_prediction', False,
                            """Perform prediction ops.""")
#
# debugging
#
tf.app.flags.DEFINE_boolean('be_verbose', False,
                            """Log extra debugging output.""")
tf.app.flags.DEFINE_boolean('do_a_short_run', False,
                            """Do a very short run.""")
#
# special logic: using the 'test' set for part of training (if no intnention
# to run a separate test round); use _all_ the data for testing/prediction
# (if no itnentions to use any of the data for training); use the 'validation'
# dataset for testing also (if intending to keep the test set 'in reserve'
# until the model is finalized.
tf.app.flags.DEFINE_boolean('do_use_test_for_train', False,
                            """Use 'test' data for training also.""")
tf.app.flags.DEFINE_boolean('do_use_all_for_test', False,
                            """Use all available data for testing/pred.""")
tf.app.flags.DEFINE_boolean('do_use_valid_for_test', False,
                            """Use validation data for testing/pred.""")


def main(argv=None):
    # check flag logic - can we run?
    if FLAGS.do_use_test_for_train and \
       FLAGS.do_use_all_for_test and \
       FLAGS.do_use_valid_for_test:
        print('Impossible set of special run flags!')
        return

    do_training = FLAGS.do_training
    do_validation = FLAGS.do_validation
    do_testing = FLAGS.do_testing
    do_prediction = FLAGS.do_prediction

    # set up logger
    import logging
    logfilename = FLAGS.log_name
    logging_level = mnv_utils.get_logging_level(FLAGS.log_level)
    logging.basicConfig(
        filename=logfilename, level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting...")
    logger.info(__file__)

    # set up features parameters
    feature_targ_dict = mnv_utils.make_feature_targ_dict(MNV_TYPE, FLAGS)
    feature_targ_dict['BUILD_KBD_FUNCTION'] = make_default_convpooldict

    # set up run parameters
    runpars_dict = mnv_utils.make_run_params_dict(MNV_TYPE, FLAGS)
    reader_class = mnv_utils.get_reader_class(FLAGS.tfrec_type)
    runpars_dict['DATA_READER_CLASS'] = reader_class

    # do a short test run?
    short = FLAGS.do_a_short_run

    # set up file lists - part of run parameters; assume our data_dir and
    # file_root are both comma-separated lists - we will make every possible
    # combinaton of them, so be careful, etc.
    train_list, valid_list, test_list = \
        mnv_utils.get_trainvalidtest_file_lists(
            FLAGS.data_dir, FLAGS.file_root, FLAGS.compression
        )
    # fix lists if there are special options
    if FLAGS.do_use_test_for_train:
        train_list.extend(test_list)
        test_list = valid_list
    if FLAGS.do_use_all_for_test:
        do_training = False   # just in case, turn this off
        test_list.extend(train_list)
        test_list.extend(valid_list)
        train_list = []
        valid_list = []
    if FLAGS.do_use_valid_for_test:
        test_list = valid_list

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

    runpars_dict['TRAIN_READER_ARGS'] = datareader_dict(train_list, 'train')
    runpars_dict['VALID_READER_ARGS'] = datareader_dict(valid_list, 'valid')
    runpars_dict['TEST_READER_ARGS'] = datareader_dict(test_list, 'data')

    # set up training parameters
    train_params_dict = mnv_utils.make_train_params_dict(MNV_TYPE, FLAGS)

    logger.info(' run_params_dict = {}'.format(repr(runpars_dict)))
    logger.info(' feature_targ_dict = {}'.format(repr(feature_targ_dict)))
    logger.info(' train_params_dict = {}'.format(repr(train_params_dict)))
    logger.info('  Final file list lengths:')
    for typ in ['train', 'valid', 'test']:
        dkey = '%s_READER_ARGS' % typ.upper()
        logger.info('   N {} = {}'.format(
            typ, len(runpars_dict[dkey]['FILENAMES_LIST'])
        ))
    model = TriColSTEpsilon(
        n_classes=FLAGS.n_classes,
        data_format=FLAGS.data_format,
        use_batch_norm=FLAGS.do_batch_norm
    )
    runner = MnvTFRunnerCategorical(
        model,
        run_params_dict=runpars_dict,
        feature_targ_dict=feature_targ_dict,
        train_params_dict=train_params_dict
    )
    if do_training:
        runner.run_training(do_validation=do_validation, short=short)
    if do_testing:
        runner.run_testing(short=short)
    if do_prediction:
        runner.run_prediction(short=short, log_freq=10)


if __name__ == '__main__':
    tf.app.run()
