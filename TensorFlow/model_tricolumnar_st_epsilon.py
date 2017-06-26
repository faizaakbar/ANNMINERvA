"""
MINERvA Tri-columnar "spacetime" (energy+time tensors) epsilon network
"""
import tensorflow as tf


def default_convpooldict(data_format='NHWC'):
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

    # assume 127x(N) images
    convpooldict_x = {}
    convpooldict_x['conv1'] = {}
    convpooldict_x['conv2'] = {}
    convpooldict_x['conv3'] = {}
    convpooldict_x['conv4'] = {}
    convpooldict_x['conv1']['kernels'] = [8, 3, 1, 12]
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
    convpooldict_u['conv1']['kernels'] = [8, 5, 1, 12]
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
    convpooldict_v['conv1']['kernels'] = [8, 5, 1, 12]
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

    return convpooldict


class TriColSTEpsilon:
    def __init__(self, n_classes, params=dict()):
        self.learning_rate = params.get('LEARNING_RATE', 0.001)
        self.batch_size = params.get('BATCH_SIZE', 128)
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.n_classes = n_classes
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step'
        )
        # note, 'NCHW' is only supported on GPUs
        self.data_format = "NHWC"

    def _build_network(self, features_list, kbd):
        """
        features_list[0] == X, [1] == U, [2] == V;
        kbd = kernels-biases-dict (convpooldict)
        """
        self.weights_biases = {}

        with tf.name_scope('input_images'):
            self.X_img = tf.cast(features_list[0], tf.float32)
            self.U_img = tf.cast(features_list[1], tf.float32)
            self.V_img = tf.cast(features_list[2], tf.float32)

        def make_kernels(
                name, shp_list, init=tf.truncated_normal_initializer()
        ):
            return tf.get_variable(
                name, shp_list, initializer=init
            )
        
        def make_biases(name, shp_list, init=tf.random_normal_initializer()):
            return tf.get_variable(name, shp_list, initializer=init)

        def make_active_conv(
                input_lyr, kernels, biases, name, act=tf.nn.relu
        ):
            conv = tf.nn.conv2d(
                input_lyr, kernels, strides=[1, 1, 1, 1],
                padding='SAME', data_format=self.data_format
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
                padding='SAME', data_format=self.data_format,
                name=name
            )

        def make_convolutional_tower(view, input_layer, kbd):
            """
            input_layer == e.g., self.X_img, etc. corresponding to 'view'
            """
            inp_lyr = None
            out_lyr = None
            for layer in ['1', '2', '3', '4']:
                if layer == '1':
                    inp_lyr = input_layer
                else:
                    inp_lyr = out_lyr

                # need to fix scope name!
                scope_name = 'conv' + layer
                with tf.variable_scope(scope_name) as scope:
                    self.weights_biases[scope.name] = {}
                    self.weights_biases[scope.name]['kernels'] = make_kernels(
                        'kernels', kbd[view][scope.name]['kernels'],
                    )
                    self.weights_biases[scope.name]['biases'] = make_biases(
                        'biases', kbd[view][scope.name]['biases'],
                    )
                    conv = make_active_conv(
                        inp_lyr,
                        self.weights_biases[scope.name]['kernels'],
                        self.weights_biases[scope.name]['biases'],
                        scope.name
                    )

                # need to fix scope name!
                scope_name = 'pool' + layer
                with tf.variable_scope(scope_name) as scope:
                    out_lyr = make_pool(conv, scope.name)

            return out_lyr
        
        out_x = make_convolutional_tower('x', self.X_img, kbd)
        out_u = make_convolutional_tower('u', self.X_img, kbd)
        out_v = make_convolutional_tower('v', self.X_img, kbd)

        # next, concat, then fc...
        #
        with tf.variable_scope('fc') as scope:
            # use weight of dimension 7 * 7 * 64 x 1024
            input_features = 7 * 7 * 64
            self.weights_fc = tf.get_variable(
                'weights',
                [input_features, 1024],
                initializer=tf.random_normal_initializer()
            )
            self.biases_fc = tf.get_variable(
                'biases',
                [1024],
                initializer=tf.random_normal_initializer()
            )
            # reshape pool2/out_lyr to 2 dimensional
            out_lyr = tf.reshape(out_lyr, [-1, input_features])
            # apply relu on matmul of pool2/out_lyr and w + b
            self.fc = tf.nn.relu(
                tf.matmul(out_lyr, self.weights_fc) + self.biases_fc
            )
            # apply dropout
            self.fc = tf.nn.dropout(self.fc, self.dropout, name='relu_dropout')

        with tf.variable_scope('softmax_linear') as scope:
            self.weights_softmax = tf.get_variable(
                'weights',
                [1024, self.n_classes],
                initializer=tf.random_normal_initializer()
            )
            self.biases_softmax = tf.get_variable(
                'biases',
                [self.n_classes],
                initializer=tf.random_normal_initializer()
            )
            self.logits = tf.add(
                tf.matmul(self.fc, self.weights_softmax),
                self.biases_softmax,
                name='logits'
            )

    def _set_targets(self, targets):
        with tf.name_scope('targets'):
            self.Y = tf.cast(targets, tf.float32)

    def _define_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.Y
                ),
                axis=0,
                name='loss'
            )

    def _define_train_op(self):
        with tf.name_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            ).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def prepare_for_inference(self, features):
        self._build_network(features)

    def prepare_for_training(self, targets):
        self._set_targets(targets)
        self._define_loss()
        self._define_train_op()
        self._create_summaries()
