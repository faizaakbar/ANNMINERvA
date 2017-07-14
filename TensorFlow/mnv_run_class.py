"""
minerva test
"""
from __future__ import print_function

import tensorflow as tf
import numpy as np

from model_tricolumnar_st_epsilon import TriColSTEpsilon
from model_tricolumnar_st_epsilon import make_default_convpooldict

from MnvDataReaders import MnvDataReaderVertexST

import os
import time

hparams_dict = dict()
hparams_dict['LEARNING_RATE'] = 0.001
hparams_dict['BATCH_SIZE'] = 4
hparams_dict['DROPOUT_KEEP_PROB'] = 0.75

SKIP_STEP = 20  # how many iters before checkpointing
N_EPOCHS = 25

# TBOARD_DIR = '/tmp/minerva/st_epsilon' + \
#              '/%06d' % np.random.randint(999999)
# TBOARD_DIR = './622446'
TBOARD_DIR = '/tmp/minerva/st_epsilon/219008'

BASE_FILEPAT = 'minosmatch_nukecczdefs_genallzwitht_pcodecap66_'
FILE_SHPTYP = '127x50x25_xtxutuvtv_'
DSAMP = 'me1Amc_'
PATH = './'

COMPRESSION = tf.python_io.TFRecordCompressionType.GZIP
COMPRESSION = tf.python_io.TFRecordCompressionType.ZLIB

COMP_EXT = ''
if COMPRESSION == tf.python_io.TFRecordCompressionType.ZLIB:
    COMP_EXT = '.zz'
elif COMPRESSION == tf.python_io.TFRecordCompressionType.GZIP:
    COMP_EXT = '.gz'

# use zlib compression for TFrecords?
TRAINF = PATH + BASE_FILEPAT + FILE_SHPTYP + DSAMP + \
         '%04d' + '_train.tfrecord' + COMP_EXT
VALIDF = PATH + BASE_FILEPAT + FILE_SHPTYP + DSAMP + \
         '%04d' + '_valid.tfrecord' + COMP_EXT
TESTF = PATH + BASE_FILEPAT + FILE_SHPTYP + DSAMP + \
        '%04d' + '_test.tfrecord' + COMP_EXT


def train(
        hparams_dict,
        tboard_dest_dir='/tmp/minerva/st_epsilon',
        short=False
):
    tf.reset_default_graph()
    print('tensorboard command:')
    print('\ttensorboard --logdir {}'.format(tboard_dest_dir))

    initial_step = 0
    ckpt_dir = tboard_dest_dir + '/checkpoints'
    run_dest_dir = tboard_dest_dir + '/%d' % time.time()

    with tf.Graph().as_default() as g:

        # TODO - modify things so we can run validation also!
        img_depth = 2
        train_list = [TRAINF % i for i in range(1)]
        valid_list = [VALIDF % i for i in range(1)]
        train_reader = MnvDataReaderVertexST(
            filenames_list=train_list,
            batch_size=hparams_dict['BATCH_SIZE'],
            name='train',
            compression=COMPRESSION
        )
        batch_dict_train = train_reader.shuffle_batch_generator()
        X_train = batch_dict_train['hitimes-x']
        U_train = batch_dict_train['hitimes-u']
        V_train = batch_dict_train['hitimes-v']
        targ_train = batch_dict_train['segments']
        f_train = [X_train, U_train, V_train]

        valid_reader = MnvDataReaderVertexST(
            filenames_list=valid_list,
            batch_size=hparams_dict['BATCH_SIZE'],
            name='valid',
            compression=COMPRESSION
        )
        batch_dict_valid = valid_reader.batch_generator(num_epochs=100)
        X_valid = batch_dict_valid['hitimes-x']
        U_valid = batch_dict_valid['hitimes-u']
        V_valid = batch_dict_valid['hitimes-v']
        targ_valid = batch_dict_valid['segments']
        f_valid = [X_valid, U_valid, V_valid]

        d = make_default_convpooldict(img_depth=img_depth)
        model = TriColSTEpsilon(learning_rate=0.001, n_classes=11)
        model.prepare_for_inference(f_train, d)
        model.prepare_for_training(targ_train)

        writer = tf.summary.FileWriter(run_dest_dir)
        saver = tf.train.Saver()

        n_steps = 5 if short else 200
        skip_step = 1 if short else SKIP_STEP
        print(' Processing {} steps...'.format(n_steps))

        init = tf.global_variables_initializer()

        start_time = time.time()

        with tf.Session(graph=g) as sess:
            sess.run(init)
            # have to run local variable init for `string_input_producer`
            sess.run(tf.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restored session from {}'.format(ckpt_dir))

            writer.add_graph(sess.graph)
            initial_step = model.global_step.eval()
            print('initial step =', initial_step)
            average_loss = 0.0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # NOTE: specifically catch `tf.errors.OutOfRangeError` or we won't
            # handle the exception correctly.
            try:
                for b_num in range(initial_step, initial_step + n_steps):
                    _, loss_batch, summary = sess.run(
                        [model.optimizer,
                         model.loss,
                         model.train_summary_op],
                        feed_dict={
                            model.dropout_keep_prob:
                            hparams_dict['DROPOUT_KEEP_PROB']
                        }
                    )
                    writer.add_summary(summary, global_step=b_num)
                    average_loss += loss_batch
                    if (b_num + 1) % skip_step == 0:
                        print('  Avg training loss at step {}: {:5.1f}'.format(
                            b_num + 1, average_loss / skip_step
                        ))
                        print('   Elapsed time = {}'.format(
                            time.time() - start_time
                        ))
                        average_loss = 0.0
                        saver.save(sess, ckpt_dir, b_num)
                        print('     saved at iter %d' % b_num)
                        # try validation
                        model.reassign_features(f_valid)
                        model.reassign_targets(targ_valid)
                        loss_valid, summary = sess.run(
                            [model.loss,
                             model.valid_summary_op],
                            feed_dict={
                                model.dropout_keep_prob: 1.0
                            }
                        )
                        writer.add_summary(summary, global_step=b_num)
                        print('   Valid loss =', loss_valid)
                        # reset for training
                        model.reassign_features(f_train)
                        model.reassign_targets(targ_train)
            except tf.errors.OutOfRangeError:
                print('Training stopped - queue is empty.')
            except Exception as e:
                print(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        writer.close()

    print('Finished training...')


def test(
        hparams_dict,
        tboard_dest_dir='/tmp/minerva/st_epsilon',
        verbose=False,
        short=False
):
    tf.reset_default_graph()
    print('Starting testing...')

    ckpt_dir = tboard_dest_dir + '/checkpoints'

    with tf.Graph().as_default() as g:

        img_depth = 2
        test_list = [TESTF % i for i in range(1)]
        data_reader = MnvDataReaderVertexST(
            filenames_list=test_list,
            batch_size=hparams_dict['BATCH_SIZE'],
            name='train',
            compression=COMPRESSION
        )
        batch_dict = data_reader.batch_generator()
        batch_size = hparams_dict['BATCH_SIZE']
        X = batch_dict['hitimes-x']
        U = batch_dict['hitimes-u']
        V = batch_dict['hitimes-v']
        targ = batch_dict['segments']
        f = [X, U, V]
        d = make_default_convpooldict(img_depth=img_depth)

        model = TriColSTEpsilon(learning_rate=0.001, n_classes=11)
        model.prepare_for_inference(f, d)
        # TODO - this needs restructuring, shouldn't need to call a fn called
        # `prepare_for_training` - here use it to compute loss
        model.prepare_for_loss_computation(targ)

        saver = tf.train.Saver()

        n_batches = 2 if short else 10000
        init = tf.global_variables_initializer()
        print(' Processing {} batches...'.format(n_batches))

        start_time = time.time()

        with tf.Session(graph=g) as sess:
            sess.run(init)
            # have to run local variable init for `string_input_producer`
            sess.run(tf.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restored session from {}'.format(ckpt_dir))
                if verbose and False:
                    print(
                        [op.name for op in
                         tf.get_default_graph().get_operations()]
                    )

            final_step = model.global_step.eval()
            print('evaluation after {} steps.'.format(final_step))
            average_loss = 0.0
            total_correct_preds = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # NOTE: specifically catch `tf.errors.OutOfRangeError` or we won't
            # handle the exception correctly.
            n_processed = 0
            try:
                for i in range(n_batches):
                    loss_batch, logits_batch, Y_batch = sess.run(
                        [model.loss, model.logits, targ],
                        feed_dict={
                            model.dropout_keep_prob: 1.0
                        }
                    )
                    n_processed += batch_size
                    average_loss += loss_batch
                    preds = tf.nn.softmax(logits_batch)
                    correct_preds = tf.equal(
                        tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
                    )
                    if verbose:
                        print('   preds   = \n{}'.format(
                            tf.argmax(preds, 1).eval()
                        ))
                        print('   Y_batch = \n{}'.format(
                            np.argmax(Y_batch, 1)
                        ))
                    accuracy = tf.reduce_sum(
                        tf.cast(correct_preds, tf.float32)
                    )
                    total_correct_preds += sess.run(accuracy)
                    if verbose:
                        print('  batch {} loss = {} for nproc {}'.format(
                            i, loss_batch, n_processed
                        ))

                    print("  total_correct_preds =", total_correct_preds)
                    print("  n_processed =", n_processed)
                    print(" Accuracy {0}".format(
                        total_correct_preds / n_processed
                    ))
                    print(' Average loss: {:5.1f}'.format(
                        average_loss / n_batches
                    ))
            except tf.errors.OutOfRangeError:
                print('Testing stopped - queue is empty.')
            except Exception as e:
                print(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        print('  Elapsed time = {}'.format(time.time() - start_time))

    print('Finished testing...')


def model_check(
        hparams_dict,
        tboard_dest_dir='/tmp/minerva/st_epsilon',
        short=False
):
    """ inspect model weights """
    tf.reset_default_graph()
    print('Starting model check...')

    ckpt_dir = tboard_dest_dir + '/checkpoints'

    # note - don't need input data here, we just want to load the saved
    # model to inspect the weights
    img_depth = 2
    X = tf.placeholder(
        tf.float32, shape=[None, 127, 50, img_depth], name='X'
    )
    U = tf.placeholder(
        tf.float32, shape=[None, 127, 25, img_depth], name='U'
    )
    V = tf.placeholder(
        tf.float32, shape=[None, 127, 25, img_depth], name='V'
    )
    targ = tf.placeholder(tf.float32, shape=[None, 11], name='targ')
    f = [X, U, V]
    d = make_default_convpooldict(img_depth=img_depth)

    model = TriColSTEpsilon(learning_rate=0.001, n_classes=11)
    model.prepare_for_inference(f, d)
    model.prepare_for_loss_computation(targ)

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

        final_step = model.global_step.eval()
        print('model after {} steps.'.format(final_step))

        k = model.weights_biases['x_tower']['conv1']['kernels'].eval()
        print('first x-tower convolutional kernel shape = ', k.shape)
        print('  k[0, 0, 0, :] =', k[0, 0, 0, :])

        # https://stackoverflow.com/questions/38160940/ ...
        print('now compute total number of trainable params...')
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            name = variable.name
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print('name = {}, shape = {}, n_params = {}'.format(
                name, shape, variable_parameters
            ))
            total_parameters += variable_parameters
        print('Total parameters =', total_parameters)

    return model


if __name__ == '__main__':

    short = True
    train(hparams_dict, TBOARD_DIR, short=short)
    test(hparams_dict, TBOARD_DIR, verbose=False, short=short)
    model_check(hparams_dict, TBOARD_DIR, short=short)
