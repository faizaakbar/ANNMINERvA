"""
minerva test. NOTE, this script assumes data is available in the form of
TFRecords, with groups of files with extensions like `..._train.tfrecord`,
`..._valid.tfrecord`, and `..._test.tfrecord` (possibly with .zz or .gz
compression extension markers).
"""
from __future__ import print_function

import tensorflow as tf

from MnvModelsTricolumnar import TriColSTEpsilon
from MnvModelsTricolumnar import make_default_convpooldict
from MnvTFRunners import MnvTFRunnerCategorical
import mnv_utils

MNV_TYPE = 'st_epsilon'
FLAGS = tf.app.flags.FLAGS

#
# input data specification
#
tf.app.flags.DEFINE_string('data_dir', '/tmp/data',
                           """Directory where data is stored.""")
tf.app.flags.DEFINE_string('file_root', 'mnv_data_',
                           """File basename.""")
tf.app.flags.DEFINE_string('compression', '',
                           """pigz (zz) or gzip (gz).""")
#
# logs and special outputs (models, predictions, etc.)
#
tf.app.flags.DEFINE_string('log_name', 'temp_log.txt',
                           """Logfile name.""")
tf.app.flags.DEFINE_string('log_level', 'INFO',
                           """Logging level (INFO/DEBUG/etc.).""")
tf.app.flags.DEFINE_string('model_dir', '/tmp/minerva/models',
                           """Directory where models are stored.""")
tf.app.flags.DEFINE_string('pred_store_name', 'temp_store',
                           """Predictions store name.""")
#
# classification goal specification
#
tf.app.flags.DEFINE_string('targets_label', 'segments',
                           """Name of targets tensor.""")
tf.app.flags.DEFINE_integer('n_classes', 11,
                            """Name of target classes.""")
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
tf.app.flags.DEFINE_boolean('use_test_for_train', False,
                            """Use 'test' data for training also.""")
tf.app.flags.DEFINE_boolean('use_all_for_test', False,
                            """Use all available data for testing/pred.""")
tf.app.flags.DEFINE_boolean('use_valid_for_test', False,
                            """Use validation data for testing/pred.""")


def main(argv=None):
    # check flag logic - can we run?
    if FLAGS.use_test_for_train and \
       FLAGS.use_all_for_test and \
       FLAGS.use_valid_for_test:
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

    # set up run parameters
    run_params_dict = mnv_utils.make_default_run_params_dict(MNV_TYPE)
    run_params_dict['MODEL_DIR'] = FLAGS.model_dir
    run_params_dict['PREDICTION_STORE_NAME'] = FLAGS.pred_store_name
    run_params_dict['BE_VERBOSE'] = FLAGS.be_verbose

    # set up file lists - part of run parameters
    comp_ext = ''
    if FLAGS.compression == 'zz':
        comp_ext = '.zz'
        run_params_dict['COMPRESSION'] = mnv_utils.ZLIB_COMP
    elif FLAGS.compression == 'gz':
        comp_ext = '.gz'
        run_params_dict['COMPRESSION'] = mnv_utils.GZIP_COMP
    train_list, valid_list, test_list = \
        mnv_utils.get_trainvalidtest_file_lists(
            FLAGS.data_dir, FLAGS.file_root, comp_ext
        )
    # fix lists if there are special options
    if FLAGS.use_test_for_train:
        train_list.extend(test_list)
        test_list = valid_list
    if FLAGS.use_all_for_test:
        do_training = False   # just in case, turn this off
        test_list.extend(train_list)
        test_list.extend(valid_list)
        test_list = []
        valid_list = []
    if FLAGS.use_valid_for_test:
        test_list = valid_list
    run_params_dict['TRAIN_FILE_LIST'] = train_list
    run_params_dict['VALID_FILE_LIST'] = valid_list
    run_params_dict['TEST_FILE_LIST'] = test_list

    # set up features parameters
    feature_targ_dict = mnv_utils.make_default_feature_targ_dict(MNV_TYPE)
    feature_targ_dict['BUILD_KBD_FUNCTION'] = make_default_convpooldict
    feature_targ_dict['TARGETS_LABEL'] = FLAGS.targets_label
    model = TriColSTEpsilon(n_classes=FLAGS.n_classes)

    # TODO - pass some training params in on the command line
    # set up training parameters
    train_params_dict = mnv_utils.make_default_train_params_dict(MNV_TYPE)

    # set up image parameters
    img_params_dict = mnv_utils.make_default_img_params_dict(MNV_TYPE)

    # tweak operating parameters for very short runs
    short = FLAGS.do_a_short_run
    if short:
        run_params_dict['SAVE_EVRY_N_BATCHES'] = 1
        train_params_dict['BATCH_SIZE'] = 64

    logger.info(' run_params_dict = {}'.format(repr(run_params_dict)))
    logger.info(' feature_targ_dict = {}'.format(repr(feature_targ_dict)))
    logger.info(' train_params_dict = {}'.format(repr(train_params_dict)))
    logger.info(' img_params_dict = {}'.format(repr(img_params_dict)))
    runner = MnvTFRunnerCategorical(
        model,
        run_params_dict=run_params_dict,
        feature_targ_dict=feature_targ_dict,
        train_params_dict=train_params_dict,
        img_params_dict=img_params_dict
    )
    if do_training:
        runner.run_training(do_validation=do_validation, short=short)
    if do_testing:
        runner.run_testing(short=short)
    if do_prediction:
        runner.run_prediction(short=short, log_freq=10)


if __name__ == '__main__':
    tf.app.run()
