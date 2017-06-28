"""
mnist!
"""
from __future__ import print_function

import tensorflow as tf
import numpy as np

from model_tricolumnar_st_epsilon import TriColSTEpsilon
from model_tricolumnar_st_epsilon import make_default_convpooldict
# from tfrecord_manager import batch_generator
# from tfrecord_manager import DATA_PATH

import os
import time

hparams_dict = dict()
hparams_dict['LEARNING_RATE'] = 0.001
hparams_dict['BATCH_SIZE'] = 128
hparams_dict['DROPOUT_KEEP_PROB'] = 0.75

SKIP_STEP = 20  # how many iters before checkpointing
N_EPOCHS = 25

TBOARD_DIR = '/tmp/minerva/st_epsilon' + \
             '/%06d' % np.random.randint(999999)
# TBOARD_DIR = './622446'
# TBOARD_DIR = '/tmp/minerva/st_epsilon/622446'


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

        model = TriColSTEpsilon(n_classes=11, params=hparams_dict)
        # train_file = DATA_PATH + 'mnv_train.tfrecord'
        # features_batch, targets_batch = batch_generator(
        #     [train_file], batch_size=128, num_epochs=1
        # )
        
        model.prepare_for_inference(f, d)
        model.prepare_for_training(targ)
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
            # try:
            #     for b_num in range(initial_step, initial_step + n_steps):
            #         _, loss_batch, summary = sess.run(
            #             [model.optimizer, model.loss, model.summary_op],
            #             feed_dict={
            #                 model.dropout: hparams_dict['DROPOUT_KEEP_PROB']
            #             }
            #         )
            #         writer.add_summary(summary, global_step=b_num)
            #         average_loss += loss_batch
            #         if (b_num + 1) % skip_step == 0:
            #             print('  Average loss at step {}: {:5.1f}'.format(
            #                 b_num + 1, average_loss / skip_step
            #             ))
            #             print('   Elapsed time = {}'.format(
            #                 time.time() - start_time
            #             ))
            #             average_loss = 0.0
            #             saver.save(sess, ckpt_dir, b_num)
            #             print('     saved at iter %d' % b_num)
            # except tf.errors.OutOfRangeError:
            #     print('Training stopped - queue is empty.')
            # except Exception as e:
            #     print(e)
            # finally:
            #     coord.request_stop()
            #     coord.join(threads)

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

        model = TriColSTEpsilon(n_classes=11, params=hparams_dict)
        model.prepare_for_inference(f, d)
        model.prepare_for_training(targ)

        # test_file = DATA_PATH + 'mnist_test.tfrecord'
        # batch_size = 5 if short else hparams_dict['BATCH_SIZE']
        # features_batch, targets_batch = batch_generator(
        #     [test_file], batch_size=batch_size, num_epochs=1
        # )

        # model.prepare_for_inference(features_batch)
        # model.prepare_for_training(targets_batch)

        saver = tf.train.Saver()

        n_batches = 5 if short else 10000
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
            # average_loss = 0.0
            # total_correct_preds = 0

            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)

            # NOTE: specifically catch `tf.errors.OutOfRangeError` or we won't
            # handle the exception correctly.
            # n_processed = 0
            # try:
            #     for i in range(n_batches):
            #         loss_batch, logits_batch, Y_batch = sess.run(
            #             [model.loss, model.logits, targets_batch],
            #             feed_dict={
            #                 model.dropout: 1.0
            #             }
            #         )
            #         n_processed += batch_size
            #         average_loss += loss_batch
            #         preds = tf.nn.softmax(logits_batch)
            #         correct_preds = tf.equal(
            #             tf.argmax(preds, 1), tf.argmax(Y_batch, 1)
            #         )
            #         if verbose:
            #             print('   preds   = \n{}'.format(
            #                 tf.argmax(preds, 1).eval()
            #             ))
            #             print('   Y_batch = \n{}'.format(
            #                 np.argmax(Y_batch, 1)
            #             ))
            #         accuracy = tf.reduce_sum(
            #             tf.cast(correct_preds, tf.float32)
            #         )
            #         total_correct_preds += sess.run(accuracy)
            #         if verbose:
            #             print('  batch {} loss = {} for nproc {}'.format(
            #                 i, loss_batch, n_processed
            #             ))

            #         print("  total_correct_preds =", total_correct_preds)
            #         print("  n_processed =", n_processed)
            #         print(" Accuracy {0}".format(
            #             total_correct_preds / n_processed
            #         ))
            #         print(' Average loss: {:5.1f}'.format(
            #             average_loss / n_batches
            #         ))
            # except tf.errors.OutOfRangeError:
            #     print('Testing stopped - queue is empty.')
            # except Exception as e:
            #     print(e)
            # finally:
            #     coord.request_stop()
            #     coord.join(threads)

        print('  Elapsed time = {}'.format(time.time() - start_time))

    print('Finished testing...')


def model_check(
        hparams_dict,
        tboard_dest_dir='/tmp/StanfordCS20si/mnist',
        short=False
):
    tf.reset_default_graph()
    ckpt_dir = tboard_dest_dir + '/checkpoints'

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

    model = TriColSTEpsilon(n_classes=11, params=hparams_dict)
    model.prepare_for_inference(f, d)
    model.prepare_for_training(targ)

    # model = MNISTConvNet(params=hparams_dict)
    # test_file = DATA_PATH + 'mnist_test.tfrecord'
    # batch_size = 5 if short else hparams_dict['BATCH_SIZE']
    # features_batch, targets_batch = batch_generator(
    #     [test_file], batch_size=batch_size, num_epochs=1
    # )
    # model.prepare_for_inference(features_batch)
    # model.prepare_for_training(targets_batch)

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
        print(k.shape)
        print(k[0, 0, 0, :])

    return model


if __name__ == '__main__':

    train(hparams_dict, TBOARD_DIR, short=True)
    # test(hparams_dict, TBOARD_DIR, verbose=False, short=True)
