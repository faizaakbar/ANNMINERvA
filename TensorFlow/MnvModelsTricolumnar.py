"""
MINERvA Tri-columnar "spacetime" (energy+time tensors) epsilon network
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init

import logging

from six.moves import range

LOGGER = logging.getLogger(__name__)


def make_default_convpooldict(
        img_depth=1, data_format='NHWC', use_batch_norm=False
):
    """
    return the default network structure dictionary

    conv kernel shape is [filt_h, filt_w, in_nch, out_nch]
    pool kernel shape is [N, H, W, C] for 'NHWC', etc.
    """
    convpooldict = {}

    if data_format == 'NHWC':
        pool_ksize = [1, 2, 1, 1]
        pool_strides = [1, 2, 1, 1]
    elif data_format == 'NCHW':
        pool_ksize = [1, 1, 2, 1]
        pool_strides = [1, 1, 2, 1]
    else:
        raise Exception('Invalid data format!')

    convpooldict['pool_ksize'] = pool_ksize
    convpooldict['pool_strides'] = pool_strides
    convpooldict['use_batch_norm'] = use_batch_norm

    # assume 127x(N) images
    convpooldict_x = {}
    convpooldict_x['conv1'] = {}
    convpooldict_x['conv2'] = {}
    convpooldict_x['conv3'] = {}
    convpooldict_x['conv4'] = {}
    convpooldict_x['conv1']['kernels'] = [8, 3, img_depth, 12]
    convpooldict_x['conv1']['biases'] = [12]
    # after 8x3 filters -> 120x(N-2) image, then maxpool -> 60x(N-2)
    convpooldict_x['conv2']['kernels'] = [7, 3, 12, 20]
    convpooldict_x['conv2']['biases'] = [20]
    # after 7x3 filters -> 54x(N-4) image, then maxpool -> 27x(N-4)
    convpooldict_x['conv3']['kernels'] = [6, 3, 20, 28]
    convpooldict_x['conv3']['biases'] = [28]
    # after 6x3 filters -> 22x(N-6) image, then maxpool -> 11x(N-6)
    convpooldict_x['conv4']['kernels'] = [6, 3, 28, 36]
    convpooldict_x['conv4']['biases'] = [36]
    # after 6x3 filters -> 6x(N-6) image, then maxpool -> 3x(N-6)
    convpooldict['x'] = convpooldict_x

    # assume 127x(N) images
    convpooldict_u = {}
    convpooldict_u['conv1'] = {}
    convpooldict_u['conv2'] = {}
    convpooldict_u['conv3'] = {}
    convpooldict_u['conv4'] = {}
    convpooldict_u['conv1']['kernels'] = [8, 5, img_depth, 12]
    convpooldict_u['conv1']['biases'] = [12]
    # after 8x3 filters -> 120x(N-4) image, then maxpool -> 60x(N-4)
    convpooldict_u['conv2']['kernels'] = [7, 3, 12, 20]
    convpooldict_u['conv2']['biases'] = [20]
    # after 7x3 filters -> 54x(N-6) image, then maxpool -> 27x(N-6)
    convpooldict_u['conv3']['kernels'] = [6, 3, 20, 28]
    convpooldict_u['conv3']['biases'] = [28]
    # after 6x3 filters -> 22x(N-8) image, then maxpool -> 11x(N-8)
    convpooldict_u['conv4']['kernels'] = [6, 3, 28, 36]
    convpooldict_u['conv4']['biases'] = [36]
    # after 6x3 filters -> 6x(N-10) image, then maxpool -> 3x(N-10)
    convpooldict['u'] = convpooldict_u

    # assume 127x(N) images
    convpooldict_v = {}
    convpooldict_v['conv1'] = {}
    convpooldict_v['conv2'] = {}
    convpooldict_v['conv3'] = {}
    convpooldict_v['conv4'] = {}
    convpooldict_v['conv1']['kernels'] = [8, 5, img_depth, 12]
    convpooldict_v['conv1']['biases'] = [12]
    # after 8x3 filters -> 120x(N-4) image, then maxpool -> 60x(N-4)
    convpooldict_v['conv2']['kernels'] = [7, 3, 12, 20]
    convpooldict_v['conv2']['biases'] = [20]
    # after 7x3 filters -> 54x(N-6) image, then maxpool -> 27x(N-6)
    convpooldict_v['conv3']['kernels'] = [6, 3, 20, 28]
    convpooldict_v['conv3']['biases'] = [28]
    # after 6x3 filters -> 22x(N-8) image, then maxpool -> 11x(N-8)
    convpooldict_v['conv4']['kernels'] = [6, 3, 28, 36]
    convpooldict_v['conv4']['biases'] = [36]
    # after 6x3 filters -> 6x(N-10) image, then maxpool -> 3x(N-10)
    convpooldict['v'] = convpooldict_v

    convpooldict['nfeat_dense_tower'] = 196
    convpooldict['nfeat_concat_dense'] = 98

    convpooldict['regularizer'] = tf.contrib.layers.l2_regularizer(
        scale=0.0001
    )

    return convpooldict


class TriColSTEpsilon:
    """
    Tri-Columnar SpaceTime Epsilon
    """
    _allowed_strategies = ['Adam', 'AdaGrad']
    
    def __init__(self, n_classes, data_format='NHWC'):
        self.n_classes = n_classes
        self.dropout_keep_prob = None
        self.global_step = None
        self.is_training = None
        self.padding = 'VALID'
        # note, 'NCHW' is only supported on GPUs
        self.data_format = data_format

    def _build_network(self, features_list, kbd):
        """
        _build_network does a bit more than just build the network; it also
        handles some initialization that is tricky in the init function in
        terms of belonging to the same graph as what is actually used.

        features_list[0] == X, [1] == U, [2] == V;
        kbd = kernels-biases-dict (convpooldict)
        """
        LOGGER.info('Building network from structure: %s' % str(kbd))
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob'
        )
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step'
        )
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope('input_images'):
            self.X_img = tf.cast(features_list[0], tf.float32)
            self.U_img = tf.cast(features_list[1], tf.float32)
            self.V_img = tf.cast(features_list[2], tf.float32)

        def make_kernels(name, shp_list):
            return tf.get_variable(
                name, shp_list, initializer=xavier_init(uniform=False),
                regularizer=kbd['regularizer']
            )

        def make_biases(name, shp_list):
            return tf.get_variable(
                name, shp_list, initializer=xavier_init(uniform=False),
                regularizer=kbd['regularizer']
            )

        def make_active_conv(
                input_lyr, kernels, biases, name, act=tf.nn.relu
        ):
            conv = tf.nn.conv2d(
                input_lyr, kernels, strides=[1, 1, 1, 1],
                padding=self.padding, data_format=self.data_format
            )
            return act(tf.nn.bias_add(
                conv, biases, data_format=self.data_format, name=name
            ))

        def make_pool(
                input_lyr, name,
                ksize=kbd['pool_ksize'], strides=kbd['pool_strides']
        ):
            return tf.nn.max_pool(
                input_lyr, ksize=ksize, strides=strides,
                padding=self.padding, data_format=self.data_format,
                name=name
            )

        def make_weights_and_biases(
                input_lyr, name, shp_list
        ):
            pass

        def make_convolutional_tower(view, input_layer, kbd, n_layers=4):
            """
            input_layer == e.g., self.X_img, etc. corresponding to 'view'
            """
            inp_lyr = None
            out_lyr = None

            # scope the tower
            twr = view + '_tower'
            with tf.variable_scope(twr):

                for i in range(n_layers):
                    layer = str(i + 1)
                    if layer == '1':
                        inp_lyr = input_layer
                    else:
                        inp_lyr = out_lyr

                    # scope the convolutional layer
                    nm = 'conv' + layer
                    with tf.variable_scope(nm):
                        k = make_kernels('kernels', kbd[view][nm]['kernels'])
                        b = make_biases('biases', kbd[view][nm]['biases'])
                        conv = make_active_conv(inp_lyr, k, b, nm + '_conv')

                    # scope the pooling layer
                    scope_name = 'pool' + layer
                    with tf.variable_scope(scope_name):
                        out_lyr = make_pool(conv, scope_name + '_pool')

                # reshape pool/out_lyr to 2 dimensional
                out_lyr_shp = out_lyr.shape.as_list()
                nfeat_tower = out_lyr_shp[1] * out_lyr_shp[2] * out_lyr_shp[3]
                out_lyr = tf.reshape(out_lyr, [-1, nfeat_tower])

                # final dense layer parameters
                w = tf.get_variable(
                    'dense_weights',
                    [nfeat_tower, kbd['nfeat_dense_tower']],
                    initializer=xavier_init(uniform=False),
                    regularizer=kbd['regularizer']
                )
                b = tf.get_variable(
                    'dense_biases',
                    [kbd['nfeat_dense_tower']],
                    initializer=xavier_init(uniform=False),
                    regularizer=kbd['regularizer']
                )
                # apply relu on matmul of pool2/out_lyr and w + b
                fc = tf.nn.relu(
                    tf.nn.bias_add(
                        tf.matmul(out_lyr, w, name='matmul'), b,
                        data_format=self.data_format,
                        name='bias_add'
                    ),
                    name='reul_activation'
                )
                # apply dropout
                fc = tf.nn.dropout(
                    fc, self.dropout_keep_prob, name='relu_dropout'
                )

            return fc

        out_x = make_convolutional_tower('x', self.X_img, kbd)
        out_u = make_convolutional_tower('u', self.U_img, kbd)
        out_v = make_convolutional_tower('v', self.V_img, kbd)

        # next, concat, then 'final' fc...
        with tf.variable_scope('fully_connected') as scope:
            tower_joined = tf.concat(
                [out_x, out_u, out_v], axis=1, name=scope.name + '_concat'
            )

            joined_shp = tower_joined.shape.as_list()
            nfeatures_joined = joined_shp[1]
            w = tf.get_variable(
                'weights',
                [nfeatures_joined, kbd['nfeat_concat_dense']],
                initializer=xavier_init(uniform=False),
                regularizer=kbd['regularizer']
            )
            b = tf.get_variable(
                'biases',
                [kbd['nfeat_concat_dense']],
                initializer=xavier_init(uniform=False),
                regularizer=kbd['regularizer']
            )
            fc_lyr = tf.nn.bias_add(
                tf.matmul(tower_joined, w), b, data_format=self.data_format
            )
            # apply relu on matmul of joined and w + b
            fc_lyr = tf.nn.relu(fc_lyr)
            # apply dropout
            self.fc = tf.nn.dropout(
                fc_lyr, self.dropout_keep_prob, name='relu_dropout'
            )

        with tf.variable_scope('softmax_linear'):
            self.weights_softmax = tf.get_variable(
                'weights',
                [kbd['nfeat_concat_dense'], self.n_classes],
                initializer=xavier_init(uniform=False),
                regularizer=kbd['regularizer']
            )
            self.biases_softmax = tf.get_variable(
                'biases',
                [self.n_classes],
                initializer=xavier_init(uniform=False),
                regularizer=kbd['regularizer']
            )
            self.logits = tf.nn.bias_add(
                tf.matmul(self.fc, self.weights_softmax),
                self.biases_softmax,
                data_format=self.data_format,
                name='logits'
            )

    def _set_targets(self, targets):
        with tf.variable_scope('targets'):
            self.targets = tf.cast(targets, tf.float32)

    def _define_loss(self):
        with tf.variable_scope('loss'):
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES
            )
            self.reg_loss = sum(regularization_losses)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.targets
                ),
                axis=0,
                name='loss'
            ) + self.reg_loss
            preds = tf.nn.softmax(self.logits, name='preds')
            correct_preds = tf.equal(
                tf.argmax(preds, 1), tf.argmax(self.targets, 1),
                name='correct_preds'
            )
            self.accuracy = tf.divide(
                tf.reduce_sum(tf.cast(correct_preds, tf.float32)),
                tf.cast(tf.shape(self.targets)[0], tf.float32),
                name='accuracy'
            )

    def _define_train_op(self, learning_rate, strategy):
        LOGGER.info('Building train op with learning_rate = %f' %
                    learning_rate)
        if strategy in self._allowed_strategies:
            with tf.variable_scope('training'):
                if strategy == 'Adam':
                    self.optimizer = tf.train.AdamOptimizer(
                        learning_rate=learning_rate
                    ).minimize(self.loss, global_step=self.global_step)
                elif strategy == 'AdaGrad':
                    self.optimizer = tf.train.MomentumOptimizer(
                        learning_rate=learning_rate,
                        momentum=0.9, use_nesterov=True
                    ).minimize(self.loss, global_step=self.global_step)
        else:
            raise ValueError('Invalid training strategy choice!')

    def _create_summaries(self):
        # assume we built the readers before the model...
        base_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        with tf.variable_scope('summaries/train'):
            train_reg_loss = tf.summary.scalar('reg_loss', self.reg_loss)
            train_loss = tf.summary.scalar('loss', self.loss)
            train_histo_loss = tf.summary.histogram(
                'histogram_loss', self.loss
            )
            train_summaries = [
                train_reg_loss, train_loss, train_histo_loss
            ]
            train_summaries.extend(base_summaries)
            self.train_summary_op = tf.summary.merge(
                train_summaries
            )
        with tf.variable_scope('summaries/valid'):
            valid_loss = tf.summary.scalar('loss', self.loss)
            valid_histo_loss = tf.summary.histogram(
                'histogram_loss', self.loss
            )
            valid_accuracy = tf.summary.scalar('accuracy', self.accuracy)
            valid_histo_accuracy = tf.summary.histogram(
                'histogram_accuracy', self.accuracy
            )
            valid_summaries = [
                valid_loss, valid_histo_loss,
                valid_accuracy, valid_histo_accuracy
            ]
            valid_summaries.extend(base_summaries)
            self.valid_summary_op = tf.summary.merge(
                valid_summaries
            )

    def prepare_for_inference(self, features, kbd):
        """ kbd == kernels_biases_dict (convpooldict) """
        self._build_network(features, kbd)

    def prepare_for_training(
            self, targets, learning_rate=0.001, strategy='Adam'
    ):
        """ prep the train op and compute loss, plus summaries """
        self._set_targets(targets)
        self._define_loss()
        self._define_train_op(learning_rate, strategy)
        self._create_summaries()

    def prepare_for_loss_computation(self, targets):
        """ compute the loss and prepare summaries (good for testing) """
        self._set_targets(targets)
        self._define_loss()
        self._create_summaries()

    def get_output_nodes(self):
        """ list of output nodes for graph freezing """
        return ["softmax_linear/logits"]


def test():
    tf.reset_default_graph()
    img_depth = 2
    Xshp = [None, 127, 50, img_depth]
    UVshp = [None, 127, 25, img_depth]
    X = tf.placeholder(tf.float32, shape=Xshp, name='X')
    U = tf.placeholder(tf.float32, shape=UVshp, name='U')
    V = tf.placeholder(tf.float32, shape=UVshp, name='V')
    targ = tf.placeholder(tf.float32, shape=[None, 11], name='targ')
    f = [X, U, V]
    d = make_default_convpooldict(img_depth=img_depth)
    t = TriColSTEpsilon(11)
    t.prepare_for_inference(f, d)
    t.prepare_for_training(targ)

    # test reassignment
    Xp = tf.placeholder(tf.float32, shape=Xshp, name='Xp')
    Up = tf.placeholder(tf.float32, shape=UVshp, name='Up')
    Vp = tf.placeholder(tf.float32, shape=UVshp, name='Vp')
    targp = tf.placeholder(tf.float32, shape=[None, 11], name='targp')
    fp = [Xp, Up, Vp]
    t.reassign_features(fp)
    t.reassign_targets(targp)


if __name__ == '__main__':
    test()
