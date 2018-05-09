import tensorflow as tf
from collections import OrderedDict

FILEDIR = '/Users/perdue/Documents/MINERvA/AI/minerva_tf/tfrec/201804/me1Amc'
FILENAME = 'hadmultkineimgs_127x94_me1Amc_000000_test.tfrecord.gz'
TFRECFILE = FILEDIR + '/' + FILENAME

CURRENT = 'current'
N_ELECTRONS = 'n_electrons'
N_MUONS = 'n_muons'
N_TAUS = 'n_taus'
IS_SIGNAL = 'is_signal'

DATA_FIELDS = [CURRENT, N_ELECTRONS, N_MUONS, N_TAUS]
FEATURES_DICT = {
    f: tf.FixedLenFeature([], tf.string) for f in DATA_FIELDS
}


def get_tfrecord_filequeue_and_reader(num_epochs=1):
    file_queue = tf.train.string_input_producer(
        [TFRECFILE],
        name='file_queue',
        num_epochs=num_epochs
    )
    reader = tf.TFRecordReader(
        options=tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP
        )
    )
    _, tfrecord = reader.read(file_queue)
    return tfrecord


def decode_basic(tfrecord_features, field, tf_dtype):
    decoded_raw = tf.decode_raw(tfrecord_features[field], tf_dtype)
    return decoded_raw


def true_fn():
    return 1


def false_fn():
    return 0


def cast2bool(tnsr):
    return tf.cast(tnsr, tf.bool)


def decode_tfrecord_feature(tfrecord_features, field):
    if field in DATA_FIELDS:
        decoded_basic = decode_basic(tfrecord_features, field, tf.int32)
        return decoded_basic
    elif field == IS_SIGNAL:
        c = decode_basic(tfrecord_features, CURRENT, tf.int32)
        sig = tf.ones_like(c)
        bkg = tf.zeros_like(c)
        c = cast2bool(c)
        e = cast2bool(decode_basic(tfrecord_features, N_ELECTRONS, tf.int32))
        m = cast2bool(decode_basic(tfrecord_features, N_MUONS, tf.int32))
        t = cast2bool(decode_basic(tfrecord_features, N_TAUS, tf.int32))
        s = tf.logical_and(
            c, tf.logical_and(
                e, tf.logical_and(
                    tf.logical_not(m), tf.logical_not(t)
                )
            )
        )
        z = tf.where(s, x=sig, y=bkg)
        # is_c = tf.cond(tf.equal(c, tf.ones_like(c)), true_fn, false_fn)
        # is_e = tf.cond(e, 1, 0)
        # is_m = tf.cond(m, 1, 0)
        # is_t = tf.cond(t, 1, 0)
        # return tf.cond(
        #     c and e and not m and not t, 1, 0
        # )
        return z


def tfrecord_to_graph_ops(num_epochs=1):
    tfrec = get_tfrecord_filequeue_and_reader(num_epochs)
    tfrec_features = tf.parse_single_example(
        tfrec, features=FEATURES_DICT, name='tfrec_feat'
    )
    od = OrderedDict()
    for field in DATA_FIELDS + [IS_SIGNAL]:
        od[field] = decode_tfrecord_feature(
            tfrec_features, field
        )
    return od


def batch_generator(num_epochs=1):
    od = tfrecord_to_graph_ops(num_epochs)
    capacity = 10
    batch_values = tf.train.batch(
        od.values(),
        batch_size=1,
        capacity=capacity,
        enqueue_many=True,
        allow_smaller_final_batch=True,
        name='batch'
    )
    rd = OrderedDict(zip(od.keys(), batch_values))
    return rd


# or, sess = tf.InteractiveSession()
rd = batch_generator()
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    c = rd[CURRENT]
    s = rd[IS_SIGNAL]
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        for _ in range(1):
            current, signal = sess.run([c, s])
            print(current, signal)
    except tf.errors.OutOfRangeError:
        print('out of examples')
    except Exception as e:
        print(e)
    finally:
        coord.request_stop()
        coord.join(threads)

    print('finished session')
    print('closing session')


print('finished')
print('ending program')
