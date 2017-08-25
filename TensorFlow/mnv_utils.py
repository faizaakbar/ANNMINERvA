#!/usr/bin/env python
import tensorflow as tf
import logging

LOGGER = logging.getLogger(__name__)

BATCH_SIZE = 128
ZLIB_COMP = tf.python_io.TFRecordCompressionType.ZLIB
GZIP_COMP = tf.python_io.TFRecordCompressionType.GZIP


def make_default_run_params_dict(mnv_type='st_epsilon'):
    run_params_dict = {}
    run_params_dict['TRAIN_FILE_LIST'] = None
    run_params_dict['VALID_FILE_LIST'] = None
    run_params_dict['TEST_FILE_LIST'] = None
    run_params_dict['COMPRESSION'] = None
    run_params_dict['MODEL_DIR'] = '/tmp/minerva'
    run_params_dict['LOAD_SAVED_MODEL'] = True
    run_params_dict['SAVE_EVRY_N_BATCHES'] = 20
    run_params_dict['BE_VERBOSE'] = False
    run_params_dict['PREDICTION_STORE_NAME'] = 'preds'
    return run_params_dict


def make_default_feature_targ_dict(mnv_type='st_epsilon'):
    feature_targ_dict = {}
    if mnv_type == 'st_epsilon':
        feature_targ_dict['FEATURE_STR_DICT'] = {}
        feature_targ_dict['FEATURE_STR_DICT']['x'] = 'hitimes-x'
        feature_targ_dict['FEATURE_STR_DICT']['u'] = 'hitimes-u'
        feature_targ_dict['FEATURE_STR_DICT']['v'] = 'hitimes-v'
        feature_targ_dict['TARGETS_LABEL'] = None
        feature_targ_dict['IMG_DEPTH'] = 2
        feature_targ_dict['BUILD_KBD_FUNCTION'] = None
    return feature_targ_dict


def make_default_train_params_dict(mnv_type='st_epsilon'):
    train_params_dict = {}
    train_params_dict['LEARNING_RATE'] = 0.001
    train_params_dict['BATCH_SIZE'] = BATCH_SIZE
    train_params_dict['NUM_EPOCHS'] = 1
    train_params_dict['MOMENTUM'] = 0.9
    train_params_dict['STRATEGY'] = tf.train.AdamOptimizer
    train_params_dict['DROPOUT_KEEP_PROB'] = 0.5
    return train_params_dict


def make_default_img_params_dict(mnv_type='st_epsilon'):
    img_params_dict = {}
    if mnv_type == 'st_epsilon':
        img_params_dict['IMG_DEPTH'] = 2
    return img_params_dict


def get_logging_level(log_level):
    logging_level = logging.INFO
    if log_level == 'DEBUG':
        logging_level = logging.DEBUG
    elif log_level == 'INFO':
        logging_level = logging.INFO
    elif log_level == 'WARNING':
        logging_level = logging.WARNING
    elif log_level == 'ERROR':
        logging_level = logging.ERROR
    elif log_level == 'CRITICAL':
        logging_level = logging.CRITICAL
    else:
        print('Unknown or unset logging level. Using INFO')

    return logging_level


def get_trainvalidtest_file_lists(data_dir, file_root, comp_ext):
    """
    Assume we are looking for three sets of files in the form of
    TFRecords, with groups of files with extensions like
    `..._train.tfrecord`, `..._valid.tfrecord`, and
    `..._test.tfrecord` (possibly with .zz or .gz compression
    extension markers).
    """
    import glob
    train_list = glob.glob(data_dir + '/' + file_root +
                           '*_train.tfrecord' + comp_ext)
    valid_list = glob.glob(data_dir + '/' + file_root +
                           '*_valid.tfrecord' + comp_ext)
    test_list = glob.glob(data_dir + '/' + file_root +
                          '*_test.tfrecord' + comp_ext)
    for t, l in zip(['training', 'validation', 'test'],
                    [train_list, valid_list, test_list]):
        LOGGER.info('{} file list ='.format(t))
        for filename in l:
            LOGGER.info('  {}'.format(filename))
    if len(train_list) == 0 and \
       len(valid_list) == 0 and \
       len(test_list) == 0:
        LOGGER.error('No files found at specified path!')
        return None, None, None
    return train_list, valid_list, test_list


def get_number_of_trainable_parameters():
    """ use default graph """
    # https://stackoverflow.com/questions/38160940/ ...
    LOGGER.debug('Now compute total number of trainable params...')
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        name = variable.name
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        LOGGER.debug(' layer name = {}, shape = {}, n_params = {}'.format(
            name, shape, variable_parameters
        ))
        total_parameters += variable_parameters
    LOGGER.debug('Total parameters = %d' % total_parameters)
    return total_parameters


def decode_eventid(eventid):
    """
    assume "standard" encoding
    """
    evtid = str(eventid)
    phys_evt = evtid[-2:]
    evtid = evtid[:-2]
    gate = evtid[-4:]
    evtid = evtid[:-4]
    subrun = evtid[-4:]
    evtid = evtid[:-4]
    run = evtid
    return (run, subrun, gate, phys_evt)


def freeze_graph(
        model_dir, output_nodes_list, output_graph_name='frozen_model.pb'
):
    """
    reduce a saved model and metadata down to a deployable file; following
    https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

    output_node_names = e.g., 'softmax_linear/logits'
    """
    from tensorflow.python.framework import graph_util

    LOGGER.info('Attempting to freeze graph at {}'.format(model_dir))
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    if input_checkpoint is None:
        LOGGER.error('Cannot load checkpoint at {}'.format(model_dir))
        return None

    # specify the output graph
    absolute_model_dir = '/'.join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + '/' + output_graph_name

    # import the meta graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)
    # get the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # convert variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_nodes_list
        )
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        LOGGER.info('Froze graph with {} ops'.format(
            len(output_graph_def.node)
        ))

    return output_graph
