#!/usr/bin/env python
import tensorflow as tf

BATCH_SIZE = 128
ZLIB_COMP = tf.python_io.TFRecordCompressionType.ZLIB
GZIP_COMP = tf.python_io.TFRecordCompressionType.GZIP


def make_default_run_params_dict(mnv_type='st_epsilon'):
    run_params_dict = {}
    run_params_dict['TRAIN_FILE_LIST'] = None
    run_params_dict['VALID_FILE_LIST'] = None
    run_params_dict['TEST_FILE_LIST'] = None
    run_params_dict['COMPRESSION'] = None
    run_params_dict['MODEL_DIR'] = None
    run_params_dict['LOAD_SAVED_MODEL'] = True
    run_params_dict['SAVE_EVRY_N_EVTS'] = BATCH_SIZE * 5
    run_params_dict['DEBUG_PRINT'] = True
    run_params_dict['BE_VERBOSE'] = False
    run_params_dict['WRITE_DB'] = True
    
    if run_params_dict['COMPRESSION'] == ZLIB_COMP:
        run_params_dict['COMP_EXT'] = '.zz'
    elif run_params_dict['COMPRESSION'] == GZIP_COMP:
        run_params_dict['COMP_EXT'] = '.gz'

    return run_params_dict


def make_default_feature_targ_dict(mnv_type='st_epsilon'):
    feature_targ_dict = {}
    if mnv_type == 'st_epsilon':
        feature_targ_dict['FEATURE_STR_DICT'] = {}
        feature_targ_dict['FEATURE_STR_DICT']['x'] = 'hitimes-x'
        feature_targ_dict['FEATURE_STR_DICT']['u'] = 'hitimes-u'
        feature_targ_dict['FEATURE_STR_DICT']['v'] = 'hitimes-v'
        feature_targ_dict['TARGETS_LABEL'] = 'segments'
    return feature_targ_dict


def make_default_train_params_dict(mnv_type='st_epsilon'):
    train_params_dict = {}
    train_params_dict['LEARNING_RATE'] = 0.001
    train_params_dict['BATCH_SIZE'] = BATCH_SIZE
    train_params_dict['NUM_EPOCHS'] = 1
    train_params_dict['MOMENTUM'] = 0.9
    train_params_dict['STRATEGY'] = tf.train.AdamOptimizer
    train_params_dict['DROPOUT_KEEP_PROB'] = 0.75
    return train_params_dict


def make_default_img_params_dict(mnv_type='st_epsilon'):
    img_params_dict = {}
    img_params_dict['IMG_DEPTH'] = 2
    return img_params_dict
