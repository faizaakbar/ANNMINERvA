#!/usr/bin/env python
"""
Run TF
"""
from __future__ import print_function
import os
import time
import logging

import tensorflow as tf
import numpy as np
from six.moves import range

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
    ):
        if feature_targ_dict is None:
            feature_targ_dict = dict()
        if train_params_dict is None:
            train_params_dict = dict()

        try:
            self.data_reader = run_params_dict['DATA_READER_CLASS']
            self.train_reader_args = run_params_dict['TRAIN_READER_ARGS']
            self.valid_reader_args = run_params_dict['VALID_READER_ARGS']
            self.test_reader_args = run_params_dict['TEST_READER_ARGS']
            self.save_model_directory = run_params_dict['MODEL_DIR']
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
        self.img_depth = feature_targ_dict.get('IMG_DEPTH', 2)
        try:
            self.build_kbd_function = feature_targ_dict['BUILD_KBD_FUNCTION']
        except KeyError as e:
            print(e)

        self.learning_rate = train_params_dict.get('LEARNING_RATE', 0.001)
        self.num_epochs = train_params_dict.get('NUM_EPOCHS', 1)
        self.momentum = train_params_dict.get('MOMENTUM', 0.9)
        self.dropout_keep_prob = train_params_dict.get(
            'DROPOUT_KEEP_PROB', 0.75
        )
        self.strategy = train_params_dict.get('STRATEGY', 'Adam')

        self.model = model
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
        except ValueError as e:
            LOGGER.error('{}'.format(e))
            from MnvRecorderText import MnvCategoricalTextRecorder
            self.data_recorder = MnvCategoricalTextRecorder(
                self.pred_store_name
            )            

    def _get_img_shp(self):
        img_shp = self.train_reader_args['IMG_SHP']
        return img_shp

    def _prep_targets_and_features_minerva(self, generator, num_epochs=1):
        batch_dict = generator(num_epochs=num_epochs)
        X = batch_dict[self.features['x']]
        U = batch_dict[self.features['u']]
        V = batch_dict[self.features['v']]
        targets = batch_dict[self.targets_label]
        features = [X, U, V]
        eventids = batch_dict['eventids']
        return targets, features, eventids

    def _log_confusion_maxtr(self, conf_mat):
        conf_mat_filename = self.save_model_directory + \
            '/confusion_matrix.npy'
        if os.path.isfile(conf_mat_filename):
            os.remove(conf_mat_filename)
        LOGGER.info('   Saving confusion matrix to {}'.format(
            conf_mat_filename
        ))
        np.save(conf_mat_filename, conf_mat)
        acc_by_class = 100.0 * np.diag(conf_mat) / np.sum(conf_mat, axis=1)
        for i, v in enumerate(acc_by_class):
            LOGGER.info("   class {:03d} accuracy:\t{:0.5f} %".format(
                i, acc_by_class[i]
            ))

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
            n_batches = 20 if short else int(1e9)
            save_every_n_batch = 5 if short else self.save_freq
            LOGGER.info(' Processing {} batches, saving every {}...'.format(
                n_batches, save_every_n_batch
            ))

            with tf.Session(graph=g) as sess:

                train_reader = self.data_reader(self.train_reader_args)
                targets_train, features_train, eventids_train = \
                    self._prep_targets_and_features_minerva(
                        train_reader.shuffle_batch_generator,
                        self.num_epochs
                    )

                valid_reader = self.data_reader(self.valid_reader_args)
                targets_valid, features_valid, eventids_valid = \
                    self._prep_targets_and_features_minerva(
                        valid_reader.batch_generator,
                        1000000
                    )

                def get_features_train():
                    return features_train

                def get_features_valid():
                    return features_valid

                def get_targets_train():
                    return targets_train

                def get_targets_valid():
                    return targets_valid

                def get_eventids_train():
                    return eventids_train

                def get_eventids_valid():
                    return eventids_valid

                with tf.variable_scope('pipeline_control'):
                    use_valid = tf.placeholder(
                        tf.bool, shape=(), name='train_val_batch_logic'
                    )

                features = tf.cond(
                    use_valid,
                    get_features_valid,
                    get_features_train,
                    name='features_selection'
                )
                targets = tf.cond(
                    use_valid,
                    get_targets_valid,
                    get_targets_train,
                    name='targets_selection'
                )
                eventids = tf.cond(
                    use_valid,
                    get_eventids_valid,
                    get_eventids_train,
                    name='eventids_selection'
                )

                d = self.build_kbd_function(img_depth=self.img_depth)
                self.model.prepare_for_inference(features, d)
                self.model.prepare_for_training(
                    targets,
                    learning_rate=self.learning_rate,
                    strategy=self.strategy
                )
                LOGGER.info('Preparing to train model with %d parameters' %
                            mnv_utils.get_number_of_trainable_parameters())

                writer = tf.summary.FileWriter(run_dest_dir)
                saver = tf.train.Saver(
                    save_relative_paths=True
                )

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
                    for b_num in range(
                            initial_batch, initial_batch + n_batches
                    ):
                        LOGGER.debug('  processing batch {}'.format(b_num))
                        _, loss, summary_t = sess.run(
                            [self.model.optimizer,
                             self.model.loss,
                             self.model.train_summary_op],
                            feed_dict={
                                use_valid: False,
                                self.model.dropout_keep_prob:
                                self.dropout_keep_prob
                            }
                        )
                        LOGGER.debug(
                            '  Train loss at batch {}: {:6.5f}'.format(
                                b_num, loss
                            )
                        )
                        if (b_num) % save_every_n_batch == 0:
                            writer.add_summary(summary_t, global_step=b_num)
                        if (b_num + 1) % save_every_n_batch == 0:
                            # validation
                            loss, logits, targs, summary_v, evtids, acc = \
                                sess.run(
                                    [self.model.loss,
                                     self.model.logits,
                                     self.model.targets,
                                     self.model.valid_summary_op,
                                     eventids,
                                     self.model.accuracy],
                                    feed_dict={
                                        use_valid: True,
                                        self.model.dropout_keep_prob: 1.0
                                    }
                                )
                            saver.save(sess, ckpt_dir, b_num)
                            writer.add_summary(summary_v, global_step=b_num)
                            preds = tf.nn.softmax(logits)
                            LOGGER.debug('   preds   = \n{}'.format(
                                tf.argmax(preds, 1).eval()
                            ))
                            LOGGER.debug('   Y_batch = \n{}'.format(
                                np.argmax(targs, 1)
                            ))
                            LOGGER.debug('   eventids[:10] = \n{}'.format(
                                evtids[:10]
                            ))
                            LOGGER.info(
                                '  Valid loss at batch {}: {:6.5f}'.format(
                                    b_num, loss
                                )
                            )
                            LOGGER.info('    accuracy = {}'.format(
                                acc
                            ))
                            LOGGER.info('   Elapsed time = {}'.format(
                                time.time() - start_time
                            ))
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
            self.save_model_directory, self.model.get_output_nodes()
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

        with tf.Graph().as_default() as g:

            n_batches = 2 if short else int(1e9)
            LOGGER.info(' Processing {} batches...'.format(n_batches))

            with tf.Session(graph=g) as sess:
                data_reader = self.data_reader(self.test_reader_args)
                targets, features, _ = \
                    self._prep_targets_and_features_minerva(
                        data_reader.batch_generator
                    )

                d = self.build_kbd_function(img_depth=self.img_depth)
                self.model.prepare_for_inference(features, d)
                self.model.prepare_for_loss_computation(targets)
                LOGGER.info('Preparing to test model with %d parameters' %
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
                else:
                    LOGGER.error('No model found!')
                    return

                final_step = self.model.global_step.eval()
                LOGGER.info('evaluation after {} steps.'.format(final_step))
                average_loss = 0.0
                total_correct_preds = 0

                confusion_matrix = np.zeros(
                    self.model.n_classes * self.model.n_classes,
                    dtype='float32'
                ).reshape(self.model.n_classes, self.model.n_classes)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # NOTE: specifically catch `tf.errors.OutOfRangeError` or we
                # won't handle the exception correctly.
                n_processed = 0
                try:
                    for i in range(n_batches):
                        loss_batch, logits_batch, Y_batch = sess.run(
                            [self.model.loss, self.model.logits, targets],
                            feed_dict={
                                self.model.dropout_keep_prob: 1.0
                            }
                        )
                        batch_sz = logits_batch.shape[0]
                        n_processed += batch_sz
                        average_loss += loss_batch
                        preds = tf.nn.softmax(logits_batch)
                        correct_preds = tf.equal(
                            tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                        )
                        accuracy = tf.reduce_sum(
                            tf.cast(correct_preds, tf.float32)
                        )
                        total_correct_preds += sess.run(accuracy)
                        evald_preds = tf.argmax(preds, 1).eval()
                        evald_ysbatch = tf.argmax(Y_batch, 1).eval()
                        for t, p in zip(evald_ysbatch, evald_preds):
                            confusion_matrix[p][t] += 1.0
                        if self.be_verbose:
                            LOGGER.debug('   preds   = \n{}'.format(
                                evald_preds
                            ))
                            LOGGER.debug('   Y_batch = \n{}'.format(
                                evald_ysbatch
                            ))
                            LOGGER.debug(
                                '  batch {} loss = {} for size = {}'.format(
                                    i, loss_batch, batch_sz
                                )
                            )

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

            self._log_confusion_maxtr(confusion_matrix)
            LOGGER.info('  Elapsed time = {}'.format(time.time() - start_time))

        LOGGER.info('Finished testing...')

    def run_prediction(self, short=False, log_freq=10):
        """
        make predictions based on the TEST file list
        """
        LOGGER.info("Starting prediction...")
        tf.reset_default_graph()
        ckpt_dir = self.save_model_directory + '/checkpoints'

        with tf.Graph().as_default() as g:

            n_batches = 2 if short else int(1e9)
            LOGGER.info(' Processing {} batches...'.format(n_batches))

            with tf.Session(graph=g) as sess:
                data_reader = self.data_reader(self.test_reader_args)
                targets, features, eventids = \
                    self._prep_targets_and_features_minerva(
                        data_reader.batch_generator
                    )

                d = self.build_kbd_function(img_depth=self.img_depth)
                self.model.prepare_for_inference(features, d)
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
                else:
                    LOGGER.error('No model found!')
                    return

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
                        logits_batch, evtids = sess.run(
                            [self.model.logits, eventids],
                            feed_dict={
                                self.model.dropout_keep_prob: 1.0
                            }
                        )
                        batch_sz = logits_batch.shape[0]
                        n_processed += batch_sz
                        probs = tf.nn.softmax(logits_batch).eval()
                        preds = tf.argmax(probs, 1).eval()
                        if printlog:
                            if self.be_verbose:
                                LOGGER.debug("  batch size = %d" % batch_sz)
                                LOGGER.debug('   preds   = \n{}'.format(
                                    preds
                                ))
                            LOGGER.debug("  tot processed = %d" % n_processed)
                        for i, evtid in enumerate(evtids):
                            if printlog and self.be_verbose:
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

        g = tf.Graph()
        with g.as_default():
            # note - don't need input data here, we just want to load the saved
            # model to inspect the weights
            img_shp = self._get_img_shp()  # h, w_x, w_uv, depth
            X = tf.placeholder(
                tf.float32,
                shape=[None, img_shp[0], img_shp[1], img_shp[3]],
                name='X'
            )
            U = tf.placeholder(
                tf.float32,
                shape=[None, img_shp[0], img_shp[2], img_shp[3]],
                name='U'
            )
            V = tf.placeholder(
                tf.float32,
                shape=[None, img_shp[0], img_shp[2], img_shp[3]],
                name='V'
            )
            targ = tf.placeholder(
                tf.float32, shape=[None, self.model.n_classes], name='targ'
            )
            f = [X, U, V]
            d = self.build_kbd_function(img_depth=img_shp[3])
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

                k = g.get_tensor_by_name('x_tower/conv1/kernels:0').eval()
                LOGGER.info(
                    'first x-tower convolutional kernel shape = {}'.format(
                        k.shape
                    )
                )
                LOGGER.info('  k[0, 0, 0, :] = {}'.format(k[0, 0, 0, :]))
