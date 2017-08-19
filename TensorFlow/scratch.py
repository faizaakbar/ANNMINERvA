from __future__ import print_function
import tensorflow as tf


def print_alpha():
    return 1.0


def print_beta():
    return 0.0


save_every_n_batch = 5
cntr = tf.placeholder(
    tf.int32, shape=(), name='batch_counter'
)
pfrq = tf.constant(
    save_every_n_batch,
    dtype=tf.int32,
    name='const_val_mod_nmbr'
)
tfzo = tf.constant(0, dtype=tf.int32, name='const_zero')
pred = tf.equal(
    tf.mod(cntr, pfrq), tfzo, name='train_valid_pred'
)

get_alpha = tf.placeholder(tf.bool, shape=(), name='batch_logic')
ab_testing = tf.cond(
    get_alpha,
    print_alpha,
    print_beta,
    name='do_ab_testing'
)

with tf.Session() as sess:
    for i in range(10):
        print('i={}, pred={}, (i+1)mod-save={}'.format(
            i,
            sess.run([pred], feed_dict={cntr: (i + 1)}),
            (i + 1) % save_every_n_batch
        ))

with tf.Session() as sess:
    for b in [True, False, True, True, False]:
        print(b, ', ', sess.run([ab_testing], feed_dict={get_alpha: b}))
