"""
minerva test
"""
from __future__ import print_function
import glob
import sys

import tensorflow as tf

from MnvModelsTricolumnar import TriColSTEpsilon
from MnvModelsTricolumnar import make_default_convpooldict
from MnvTFRunners import MnvTFRunnerCategorical
import mnv_utils

MNV_TYPE = 'st_epsilon'
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('data_dir', '/tmp/data',
                           """Directory where data is stored.""")
tf.app.flags.DEFINE_string('file_root', 'mnv_data_',
                           """File basename.""")
tf.app.flags.DEFINE_string('compression', '',
                           """pigz (zz) or gzip (gz).""")
tf.app.flags.DEFINE_string('log_name', 'temp_log.txt',
                           """Logfile name.""")
tf.app.flags.DEFINE_string('log_level', 'INFO',
                           """Logging level (INFO/DEBUG/etc.).""")
tf.app.flags.DEFINE_string('targets_label', 'segments',
                           """Name of targets tensor.""")
tf.app.flags.DEFINE_integer('n_classes', 11,
                            """Name of target classes.""")
tf.app.flags.DEFINE_boolean('do_training', True,
                            """Perform training ops.""")
tf.app.flags.DEFINE_boolean('do_validation', True,
                            """Perform validation ops.""")
tf.app.flags.DEFINE_boolean('do_testing', True,
                            """Perform testing ops.""")
# TODO - do_prediction


def main(argv=None):
    # set up logger
    import logging
    logfilename = FLAGS.log_name
    if FLAGS.log_level == 'DEBUG':
        logging_level = logging.DEBUG
    elif FLAGS.log_level == 'INFO':
        logging_level = logging.INFO
    else:
        print('Only accepting "DEBUG" and "INFO" log levels right now.')
        logging_level = logging.INFO

    logging.basicConfig(
        filename=logfilename, level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting...")
    logger.info(__file__)

    # set up run parameters
    run_params_dict = mnv_utils.make_default_run_params_dict(MNV_TYPE)
    # set up file lists
    comp_ext = ''
    if FLAGS.compression == 'zz':
        comp_ext = '.zz'
        run_params_dict['COMPRESSION'] = mnv_utils.ZLIB_COMP
    elif FLAGS.compression == 'gz':
        comp_ext = '.gz'
        run_params_dict['COMPRESSION'] = mnv_utils.GZIP_COMP
    train_list = glob.glob(FLAGS.data_dir + '/' + FLAGS.file_root +
                           '*_train.tfrecord' + comp_ext)
    valid_list = glob.glob(FLAGS.data_dir + '/' + FLAGS.file_root +
                           '*_valid.tfrecord' + comp_ext)
    test_list = glob.glob(FLAGS.data_dir + '/' + FLAGS.file_root +
                          '*_test.tfrecord' + comp_ext)
    logger.info('training file list =', train_list)
    logger.info('validation file list =', valid_list)
    logger.info('testing file list =', test_list)
    if len(train_list) == 0 and len(valid_list) == 0 and len(test_list) == 0:
        logger.error('No files found at specified path!')
        sys.exit(1)
    run_params_dict['TRAIN_FILE_LIST'] = train_list
    run_params_dict['VALID_FILE_LIST'] = valid_list
    run_params_dict['TEST_FILE_LIST'] = test_list

    # set up features parameters
    feature_targ_dict = mnv_utils.make_default_feature_targ_dict(MNV_TYPE)
    feature_targ_dict['BUILD_KBD_FUNCTION'] = make_default_convpooldict
    feature_targ_dict['TARGETS_LABEL'] = FLAGS.target_label
    model = TriColSTEpsilon(n_classes=FLAGS.n_classes)

    # set up training parameters
    train_params_dict = mnv_utils.make_default_train_params_dict(MNV_TYPE)

    # set up image parameters
    img_params_dict = mnv_utils.make_default_img_params_dict(MNV_TYPE)

    runner = MnvTFRunnerCategorical(
        model,
        run_params_dict=run_params_dict,
        feature_targ_dict=feature_targ_dict,
        train_params_dict=train_params_dict,
        img_params_dict=img_params_dict
    )
    if FLAGS.do_training:
        runner.run_training(do_validation=FLAGS.do_validation, short=True)
    if FLAGS.do_testing:
        runner.run_testing(short=True)


if __name__ == '__main__':
    tf.app.run()
