#!/usr/bin/env python
"""
Run TF
"""
from __future__ import print_function
import os
import time
import logging

import tensorflow as tf
from six.moves import range

from MnvDataReaders import MnvDataReaderVertexST
import mnv_utils

LOGGER = logging.getLogger(__name__)


class MnvTFRunnerCategorical:
    """
    Minerva runner class for categorical classification
    (not sure we need to make this distinction here)
    """
    def __init__(
            self,
            model,
            run_params_dict,
            feature_targ_dict=None,
            train_params_dict=None,
            img_params_dict=None,
    ):
        if feature_targ_dict is None:
            feature_targ_dict = dict()
        if train_params_dict is None:
            train_params_dict = dict()
        if img_params_dict is None:
            img_params_dict = dict()

        try:
            self.train_file_list = run_params_dict['TRAIN_FILE_LIST']
            self.valid_file_list = run_params_dict['VALID_FILE_LIST']
            self.test_file_list = run_params_dict['TEST_FILE_LIST']
            self.file_compression = run_params_dict['COMPRESSION']
            self.save_model_directory = run_params_dict['MODEL_DIR']
            self.save_model_name = run_params_dict['MODEL_NAME']
            self.load_saved_model = run_params_dict['LOAD_SAVED_MODEL']
            self.save_freq = run_params_dict['SAVE_EVRY_N_BATCHES']
            self.be_verbose = run_params_dict['BE_VERBOSE']
            self.pred_store_name = run_params_dict['PREDICTION_STORE_NAME']
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
        try:
            self.build_kbd_function = feature_targ_dict['BUILD_KBD_FUNCTION']
        except KeyError as e:
            print(e)

        self.learning_rate = train_params_dict.get('LEARNING_RATE', 0.001)
        self.batch_size = train_params_dict.get('BATCH_SIZE', 128)
        self.num_epochs = train_params_dict.get('NUM_EPOCHS', 1)
        self.momentum = train_params_dict.get('MOMENTUM', 0.9)
        self.dropout_keep_prob = train_params_dict.get(
            'DROPOUT_KEEP_PROB', 0.75
        )
        self.strategy = train_params_dict.get(
            'STRATEGY', tf.train.AdamOptimizer
        )

        self.model = model
        self.img_depth = img_params_dict.get('IMG_DEPTH', 2)
        self.views = ['x', 'u', 'v']

        self.data_recorder = None
        try:
            from MnvRecorderSQLite import MnvCategoricalSQLiteRecorder
            self.data_recorder = MnvCategoricalSQLiteRecorder(
                self.model.n_classes, self.pred_store_name
            )
        except ImportError as e:
            LOGGER.error('Cannot store prediction in sqlite: {}'.format(e))
            from MnvRecorderText import MnvCategoricalTextRecorder
            self.data_recorder = MnvCategoricalTextRecorder(
                self.pred_store_name
            )

    def run_training(
            self, do_validation=False, short=False
    ):
        """
        run training (TRAIN file list) and optionally run a validation pass
        (on the VALID file list)
        """
        LOGGER.info('staring run_training...')
        tf.reset_default_graph()
        initial_batch = 0
        ckpt_dir = self.save_model_directory + '/checkpoints'
        run_dest_dir = self.save_model_directory + '/%d' % time.time()
        LOGGER.info('tensorboard command:')
        LOGGER.info('\ttensorboard --logdir {}'.format(
            self.save_model_directory
        ))

        with tf.Graph().as_default() as g:

            # n_batches: control this with num_epochs
            n_batches = 5 if short else int(1e9)
            save_every_n_batch = 1 if short else self.save_freq
            LOGGER.info(' Processing {} batches, saving every {}...'.format(
                n_batches, save_every_n_batch
            ))

            with tf.Session(graph=g) as sess:

                train_reader = MnvDataReaderVertexST(
                    filenames_list=self.train_file_list,
                    batch_size=self.batch_size,
                    name='data',
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

                d = self.build_kbd_function(img_depth=self.img_depth)
                self.model.prepare_for_inference(f_train, d)
                self.model.prepare_for_training(
                    targ_train, learning_rate=self.learning_rate
                )
                LOGGER.info('Preparing to train model with %d parameters' %
                            mnv_utils.get_number_of_trainable_parameters())
                # for op in g.get_operations():
                #     LOGGER.debug('op name = {}'.format(op.name))
                for tnsr in g.as_graph_def().node:
                    LOGGER.debug('tnsr name = {}'.format(tnsr.name))

                writer = tf.summary.FileWriter(run_dest_dir)
                saver = tf.train.Saver()

                start_time = time.time()
                sess.run(tf.global_variables_initializer())
                # have to run local variable init for `string_input_producer`
                sess.run(tf.local_variables_initializer())

                ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    LOGGER.info('Restored session from {}'.format(ckpt_dir))

                writer.add_graph(sess.graph)
                initial_batch = self.model.global_step.eval()
                LOGGER.info('initial step = %d' % initial_batch)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # NOTE: specifically catch `tf.errors.OutOfRangeError` or we
                # won't handle the exception correctly.
                try:
                    # TODO - also grab eventids (for a while) to test and
                    # be sure we're seeing different events appropriately
                    for b_num in range(
                            initial_batch, initial_batch + n_batches
                    ):
                        LOGGER.debug('  processing batch {}'.format(b_num))
                        _, loss_batch, logits_batch, summary = sess.run(
                            [self.model.optimizer,
                             self.model.loss,
                             self.model.logits,
                             self.model.train_summary_op],
                            feed_dict={
                                self.model.dropout_keep_prob:
                                self.dropout_keep_prob
                            }
                        )
                        writer.add_summary(summary, global_step=b_num)
                        LOGGER.debug(' {} - x-tower; conv2 kernels = {}'.format(
                            b_num,
                            g.get_tensor_by_name('x_tower/conv2/kernels:0').eval()[0,0,0,:]
                        ))
                        if (b_num + 1) % save_every_n_batch == 0:
                            LOGGER.info(
                                '  Loss at batch+1 {}: {:5.1f}'.format(
                                    b_num + 1, loss_batch
                                )
                            )
                            LOGGER.info('   Elapsed time = {}'.format(
                                time.time() - start_time
                            ))
                            saver.save(sess, ckpt_dir, b_num)
                            LOGGER.info('     saved at iter %d' % b_num)
                            if do_validation:
                                # try validation
                                self.model.reassign_features(f_valid)
                                self.model.reassign_targets(targ_valid)
                                loss_valid, logits_valid, summary = sess.run(
                                    [self.model.loss,
                                     self.model.logits,
                                     self.model.valid_summary_op],
                                    feed_dict={
                                        self.model.dropout_keep_prob: 1.0
                                    }
                                )
                                writer.add_summary(summary, global_step=b_num)
                                LOGGER.info('   Valid loss = %f' % loss_valid)
                                # reset for training
                                self.model.reassign_features(f_train)
                                self.model.reassign_targets(targ_train)
                except tf.errors.OutOfRangeError:
                    LOGGER.info('Training stopped - queue is empty.')
                    LOGGER.info(
                        'Executing final save at batch {}'.format(b_num)
                    )
                    saver.save(sess, ckpt_dir, b_num)
                except Exception as e:
                    LOGGER.error(e)
                finally:
                    coord.request_stop()
                    coord.join(threads)

            writer.close()

        out_graph = mnv_utils.freeze_graph(
            self.save_model_directory,
            self.model.get_output_nodes(),
            self.save_model_name
        )
        LOGGER.info(' Saved graph {}'.format(out_graph))
        LOGGER.info('Finished training...')

    def run_testing(self, short=False):
        """
        run a test pass (not "validation"!), based on the TEST file list.
        """
        LOGGER.info('Starting testing...')
        tf.reset_default_graph()
        ckpt_dir = self.save_model_directory + '/checkpoints'

        # with tf.Graph().as_default() as g:
        for dummy_var in range(1):

            grph = mnv_utils.load_frozen_graph(
                self.save_model_directory + '/' + self.save_model_name
            )

            n_batches = 2 if short else int(1e9)
            LOGGER.info(' Processing {} batches...'.format(n_batches))

            with tf.Session(graph=grph) as sess:
                # grph = mnv_utils.load_frozen_graph(
                #     self.save_model_directory + '/' + self.save_model_name
                # )
                data_reader = MnvDataReaderVertexST(
                    filenames_list=self.test_file_list,
                    batch_size=self.batch_size,
                    name='data',
                    compression=self.file_compression
                )
                batch_dict = data_reader.batch_generator()
                # for op in grph.get_operations():
                #     LOGGER.debug('op name = {}'.format(op.name))
                for tnsr in grph.as_graph_def().node:
                    LOGGER.debug('tnsr name = {}'.format(tnsr.name))
                targ = batch_dict[self.targets_label]
                # targ = grph.get_tensor_by_name('targets:0')
                lgts = grph.get_tensor_by_name('softmax_linear/logits:0')
                # lss = grph.get_tensor_by_name('loss/loss:0')
                drpt = grph.get_tensor_by_name('dropout_keep_prob:0')

                # data_reader = MnvDataReaderVertexST(
                #     filenames_list=self.test_file_list,
                #     batch_size=self.batch_size,
                #     name='data',
                #     compression=self.file_compression
                # )
                # batch_dict = data_reader.batch_generator()
                # X = batch_dict[self.features['x']]
                # U = batch_dict[self.features['u']]
                # V = batch_dict[self.features['v']]
                # targ = batch_dict[self.targets_label]
                # f = [X, U, V]

                # d = self.build_kbd_function(img_depth=self.img_depth)
                # self.model.prepare_for_inference(f, d)
                # self.model.prepare_for_loss_computation(targ)
                LOGGER.info('Preparing to test model with %d parameters' %
                            mnv_utils.get_number_of_trainable_parameters())

                # saver = tf.train.Saver()
                start_time = time.time()

                sess.run(tf.global_variables_initializer())
                LOGGER.debug('global vars inited')
                # have to run local variable init for `string_input_producer`
                sess.run(tf.local_variables_initializer())
                LOGGER.debug('local vars inited')

                # ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
                # if ckpt and ckpt.model_checkpoint_path:
                #     saver.restore(sess, ckpt.model_checkpoint_path)
                #     LOGGER.info('Restored session from {}'.format(ckpt_dir))

                # final_step = self.model.global_step.eval()
                # LOGGER.info('evaluation after {} steps.'.format(final_step))
                average_loss = 0.0
                total_correct_preds = 0

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # NOTE: specifically catch `tf.errors.OutOfRangeError` or we
                # won't handle the exception correctly.
                n_processed = 0
                try:
                    for i in range(n_batches):
                        logits_batch, Y_batch = sess.run(
                            [lgts, targ], feed_dict={drpt: 1.0}
                            # [self.model.loss, self.model.logits, targ],
                            # feed_dict={
                            #     self.model.dropout_keep_prob: 1.0
                            # }
                        )
                        batch_sz = logits_batch.shape[0]
                        n_processed += batch_sz
                        # average_loss += loss_batch
                        preds = tf.nn.softmax(logits_batch)
                        correct_preds = tf.equal(
                            tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                        )
                        LOGGER.debug(' {} - x-tower; conv2 kernels = {}'.format(
                            i,
                            g.get_tensor_by_name('x_tower/conv2/kernels:0').eval()[0,0,0,:]
                        ))
                        if self.be_verbose:
                            LOGGER.debug('   preds   = \n{}'.format(
                                tf.argmax(preds, 1).eval()
                            ))
                            LOGGER.debug('   Y_batch = \n{}'.format(
                                tf.argmax(Y_batch, 1).eval()
                            ))
                        accuracy = tf.reduce_sum(
                            tf.cast(correct_preds, tf.float32)
                        )
                        total_correct_preds += sess.run(accuracy)
                        # if self.be_verbose:
                        #     LOGGER.debug(
                        #         '  batch {} loss = {} for size = {}'.format(
                        #             i, loss_batch, batch_sz
                        #         )
                        #     )

                except tf.errors.OutOfRangeError:
                    LOGGER.info('Testing stopped - queue is empty.')
                except Exception as e:
                    LOGGER.error(e)
                finally:
                    if n_processed > 0:
                        LOGGER.info("n_processed = {}".format(n_processed))
                        LOGGER.info(
                            " Total correct preds = {}".format(
                                total_correct_preds
                            )
                        )
                        LOGGER.info("  Accuracy: {}".format(
                            total_correct_preds / n_processed
                        ))
                        LOGGER.info('  Average loss: {}'.format(
                            average_loss / n_processed
                        ))
                    coord.request_stop()
                    coord.join(threads)

            LOGGER.info('  Elapsed time = {}'.format(time.time() - start_time))

        LOGGER.info('Finished testing...')

    def run_prediction(self, short=False, log_freq=10):
        """
        make predictions based on the TEST file list

        TODO - look for sqlalchemy and put the predictions into a sqlite db.
        If sqlalchemy is not available, look for h5py and store the predictions
        in an HDF5 file. If neither is available, store predictions in plain
        text.
        """
        LOGGER.info("Starting prediction...")
        tf.reset_default_graph()
        ckpt_dir = self.save_model_directory + '/checkpoints'

        with tf.Graph().as_default() as g:

            n_batches = 2 if short else int(1e9)
            LOGGER.info(' Processing {} batches...'.format(n_batches))

            with tf.Session(graph=g) as sess:
                data_reader = MnvDataReaderVertexST(
                    filenames_list=self.test_file_list,
                    batch_size=self.batch_size,
                    name='prediction',
                    compression=self.file_compression
                )
                batch_dict = data_reader.batch_generator()
                X = batch_dict[self.features['x']]
                U = batch_dict[self.features['u']]
                V = batch_dict[self.features['v']]
                evtids = batch_dict['eventids']
                f = [X, U, V]

                d = self.build_kbd_function(img_depth=self.img_depth)
                self.model.prepare_for_inference(f, d)
                LOGGER.info('Predictions with model with %d parameters' %
                            mnv_utils.get_number_of_trainable_parameters())

                saver = tf.train.Saver()

                start_time = time.time()

                sess.run(tf.global_variables_initializer())
                # have to run local variable init for `string_input_producer`
                sess.run(tf.local_variables_initializer())

                ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    LOGGER.info('Restored session from {}'.format(ckpt_dir))

                final_step = self.model.global_step.eval()
                LOGGER.info('evaluation after {} steps.'.format(final_step))

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # NOTE: specifically catch `tf.errors.OutOfRangeError` or we
                # won't handle the exception correctly.
                n_processed = 0
                try:
                    for i in range(n_batches):
                        printlog = True if (i + 1) % log_freq == 0 else False
                        if printlog:
                            LOGGER.debug('batch {}'.format(i))
                        logits_batch, eventids = sess.run(
                            [self.model.logits, evtids],
                            feed_dict={
                                self.model.dropout_keep_prob: 1.0
                            }
                        )
                        batch_sz = logits_batch.shape[0]
                        n_processed += batch_sz
                        probs = tf.nn.softmax(logits_batch).eval()
                        preds = tf.argmax(probs, 1).eval()
                        if printlog:
                            LOGGER.debug('   preds   = \n{}'.format(
                                preds
                            ))
                            LOGGER.info("  batch size = %d" % batch_sz)
                            LOGGER.info("  tot processed = %d" % n_processed)
                        for i, evtid in enumerate(eventids):
                            if printlog:
                                LOGGER.debug(' {} = {}, pred = {}'.format(
                                    i, evtid, preds[i]
                                ))
                                LOGGER.debug('          probs = {}'.format(
                                    probs[i]
                                ))
                            self.data_recorder.write_data(
                                evtid, preds[i], probs[i]
                            )
                except tf.errors.OutOfRangeError:
                    LOGGER.info('Predictions stopped - queue is empty.')
                except Exception as e:
                    LOGGER.error(e)
                finally:
                    self.data_recorder.close()
                    coord.request_stop()
                    coord.join(threads)

            LOGGER.info('  Elapsed time = {}'.format(time.time() - start_time))

        LOGGER.info("Finished prediction...")

    def get_weights_and_biases_values(self):
        """ inspect model weights """
        LOGGER.info('Starting weights and biases retrieval...')
        tf.reset_default_graph()
        ckpt_dir = self.save_model_directory + '/checkpoints'

        # note - don't need input data here, we just want to load the saved
        # model to inspect the weights
        X = tf.placeholder(
            tf.float32, shape=[None, 127, 50, self.img_depth], name='X'
        )
        U = tf.placeholder(
            tf.float32, shape=[None, 127, 25, self.img_depth], name='U'
        )
        V = tf.placeholder(
            tf.float32, shape=[None, 127, 25, self.img_depth], name='V'
        )
        targ = tf.placeholder(
            tf.float32, shape=[None, self.model.n_classes], name='targ'
        )
        f = [X, U, V]
        d = self.build_kbd_function(img_depth=self.img_depth)
        self.model.prepare_for_inference(f, d)
        self.model.prepare_for_loss_computation(targ)
        LOGGER.info('Preparing to check model with %d parameters' %
                    mnv_utils.get_number_of_trainable_parameters())

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # have to run local variable init for `string_input_producer`
            sess.run(tf.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restored session from {}'.format(ckpt_dir))

            final_step = self.model.global_step.eval()
            LOGGER.info('model after {} steps.'.format(final_step))

            k = self.model.weights_biases['x_tower']['conv1']['kernels'].eval()
            LOGGER.info(
                'first x-tower convolutional kernel shape = {}'.format(k.shape)
            )
            LOGGER.info('  k[0, 0, 0, :] = {}'.format(k[0, 0, 0, :]))
