#!/usr/bin/env python

import lasagne


def build_triamese_alpha(input_var_x=None, input_var_u=None, input_var_v=None):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using two layers of convolutions and pooling.
    """
    # Input layer
    l_in1_x = lasagne.layers.InputLayer(shape=(None, 1, 50, 50),
                                        input_var=input_var_x)
    l_in1_u = lasagne.layers.InputLayer(shape=(None, 1, 50, 50),
                                        input_var=input_var_u)
    l_in1_v = lasagne.layers.InputLayer(shape=(None, 1, 50, 50),
                                        input_var=input_var_v)

    # Convolutional layer with 32 kernels of size 3x3.
    # Filtering reduces the image to (50-3+1, 50-3+1) = (48, 48), (ndim-filt+1)
    # maxpooling reduces this further to (48/2, 48/2) = (24, 24), (dim/poolhw)
    #
    # NOTE: it isn't obvious I have h & w in the right "order" here...
    filt_h1 = 3
    filt_w1 = 3
    filter_size1 = (filt_h1, filt_w1)
    pool_size1 = (2, 2)
    l_conv2d1_x = lasagne.layers.Conv2DLayer(
        l_in1_x, num_filters=32, filter_size=filter_size1,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_conv2d1_u = lasagne.layers.Conv2DLayer(
        l_in1_u, num_filters=32, filter_size=filter_size1,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_conv2d1_v = lasagne.layers.Conv2DLayer(
        l_in1_v, num_filters=32, filter_size=filter_size1,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    # Max-pooling layer
    l_maxpool1_x = lasagne.layers.MaxPool2DLayer(l_conv2d1_x,
                                                 pool_size=pool_size1)
    l_maxpool1_u = lasagne.layers.MaxPool2DLayer(l_conv2d1_u,
                                                 pool_size=pool_size1)
    l_maxpool1_v = lasagne.layers.MaxPool2DLayer(l_conv2d1_v,
                                                 pool_size=pool_size1)

    # More convolution and pooling layers...
    # Filtering reduces the image to (24-3+1, 24-3+1) = (22, 22), (ndim-filt+1)
    # maxpooling reduces this further to (22/2, 22/2) = (11, 11), (dim/poolhw)
    filt_h2 = 3
    filt_w2 = 3
    filter_size2 = (filt_h2, filt_w2)
    pool_size2 = (2, 2)
    l_conv2d2_x = lasagne.layers.Conv2DLayer(
        l_maxpool1_x, num_filters=32, filter_size=filter_size2,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_conv2d2_u = lasagne.layers.Conv2DLayer(
        l_maxpool1_u, num_filters=32, filter_size=filter_size2,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_conv2d2_v = lasagne.layers.Conv2DLayer(
        l_maxpool1_v, num_filters=32, filter_size=filter_size2,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    # Max-pooling layer
    l_maxpool2_x = lasagne.layers.MaxPool2DLayer(l_conv2d2_x,
                                                 pool_size=pool_size2)
    l_maxpool2_u = lasagne.layers.MaxPool2DLayer(l_conv2d2_u,
                                                 pool_size=pool_size2)
    l_maxpool2_v = lasagne.layers.MaxPool2DLayer(l_conv2d2_v,
                                                 pool_size=pool_size2)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    l_dense1_x = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_maxpool2_x, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_dense1_u = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_maxpool2_u, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_dense1_v = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_maxpool2_v, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # Concatenate the two parallel inputs
    l_concat = lasagne.layers.ConcatLayer((l_dense1_x, l_dense1_u, l_dense1_v))

    # And, finally, the 11-unit output layer with 50% dropout on its inputs:
    outp = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_concat, p=.5),
        num_units=11,
        nonlinearity=lasagne.nonlinearities.softmax)

    return outp
