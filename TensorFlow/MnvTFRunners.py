#!/usr/bin/env python
"""
Run TF
"""
from __future__ import print_function
import os
import time
import logging

import tensorflow as tf

# TODO - shouldn't need to import models here... pass them in
from MnvModelsTricolumnar import TriColSTEpsilon
from MnvModelsTricolumnar import make_default_convpooldict
from MnvDataReaders import MnvDataReaderVertexST

logger = logging.getLogger(__name__)


class MnvTFRunnerCategorical:
    """
    Minerva runner class for categorical classification
    (not sure we need to make this distinction here)
    """
    def __init__(
            self,
            run_params_dict,
            feature_targ_dict=dict(),
            train_params_dict=dict(),
            img_params_dict=dict(),
    ):
        try:
            self.train_file_list = run_params_dict['TRAIN_FILE_LIST']
            self.valid_file_list = run_params_dict['VALID_FILE_LIST']
            self.test_file_list = run_params_dict['TEST_FILE_LIST']
            self.file_compression = run_params_dict['COMPRESSION']
            self.save_model_directory = run_params_dict['MODEL_DIR']
            self.load_saved_model = run_params_dict['LOAD_SAVED_MODEL']
            self.save_freq = run_params_dict['SAVE_EVRY_N_EVTS']
            self.debug_print = run_params_dict['DEBUG_PRINT']
            self.be_verbose = run_params_dict['BE_VERBOSE']
            self.write_db = run_params_dict['WRITE_DB']
        except KeyError as e:
            print(e)

        self.features = feature_targ_dict.get(
            'FEATURE_STR_DICT',
            dict([('x', 'hitimes-x'),
                  ('u', 'hitimes-u'),
                  ('v', 'hitimes-v')])
        )
        self.targets_label = feature_targ_dict.get(
            'TARGETS_LABEL', 'segments'
        )

        self.learning_rate = train_params_dict.get('LEARNING_RATE', 0.001)
        self.batch_size = train_params_dict.get('BATCH_SIZE', 128)
        self.num_epochs = train_params_dict.get('NUM_EPOCHS', 1)
        self.momentum = train_params_dict.get('MOMENTUM', 0.9)
        self.strategy = train_params_dict.get(
            'STRATEGY', tf.train.AdamOptimizer
        )

        self.img_depth = img_params_dict.get('IMG_DEPTH', 2)

        self.views = ['x', 'u', 'v']

    def run_training(
            self, target_label='segments', do_validation=False, short=False
    ):
        """
        run training (TRAIN file list) and optionally run a validation pass
        (on the VALID file list)
        """
        logger.info('staring run_training...')
        tf.reset_default_graph()
        initial_step = 0
        ckpt_dir = self.save_model_directory + '/checkpoints'
        logger.info('tensorboard command:')
        logger.info('\ttensorboard --logdir {}'.format(
            self.save_model_directory
        ))

        with tf.Graph().as_default() as g:
            img_depth = self.img_depth
            train_reader = MnvDataReaderVertexST(
                filenames_list=self.train_file_list,
                batch_size=self.batch_size,
                name='train',
                compression=self.file_compression
            )
            batch_dict_train = train_reader.shuffle_batch_generator(
                num_epochs=self.num_epochs
            )
            X_train = batch_dict_train[self.features['x']]
            U_train = batch_dict_train[self.features['u']]
            V_train = batch_dict_train[self.features['v']]
            targ_train = batch_dict_train[self.targets_label]
            f_train = [X_train, U_train, V_train]

            valid_reader = MnvDataReaderVertexST(
                filenames_list=self.valid_file_list,
                batch_size=self.batch_size,
                name='valid',
                compression=self.file_compression
            )
            batch_dict_valid = valid_reader.batch_generator(num_epochs=1)
            X_valid = batch_dict_valid[self.features['x']]
            U_valid = batch_dict_valid[self.features['u']]
            V_valid = batch_dict_valid[self.features['v']]
            targ_valid = batch_dict_valid[self.targets_label]
            f_valid = [X_valid, U_valid, V_valid]

            d = make_default_convpooldict(img_depth=img_depth)
            model = TriColSTEpsilon(
                learning_rate=self.learning_rate,
                n_classes=11
            )
            model.prepare_for_inference(f_train, d)
            model.prepare_for_training(targ_train)

            writer = tf.summary.FileWriter(self.save_model_directory)
            saver = tf.train.Saver()

            n_steps = 5 if short else 200
            skip_step = 1 if short else self.save_freq
            print(' Processing {} steps...'.format(n_steps))

            init = tf.global_variables_initializer()
            # TODO, continue with `start_time`...

    def run_testing(self):
        """
        run a test pass (not "validation"!), based on the TEST file list.
        """
        pass

    def run_prediction(self):
        """
        make predictions based on the TEST file list
        """
        pass
