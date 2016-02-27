#!/usr/bin/env python

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer


def build_triamese_alpha(input_var_x=None, input_var_u=None, input_var_v=None,
                         imgh=50, imgw=50):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using two layers of convolutions and pooling.
    """
    # Input layer
    tshape = (None, 1, imgh, imgw)
    l_in1_x = InputLayer(shape=tshape, input_var=input_var_x)
    l_in1_u = InputLayer(shape=tshape, input_var=input_var_u)
    l_in1_v = InputLayer(shape=tshape, input_var=input_var_v)

    def make_branch(input_layer,
                    num_filters1, filter_size1, pool_size1,
                    num_filters2, filter_size2, pool_size2):
        """
        see: http://lasagne.readthedocs.org/en/latest/modules/layers.html
        """
        convlayer1 = Conv2DLayer(input_layer, num_filters=num_filters1,
                                filter_size=filter_size1,
                                nonlinearity=lasagne.nonlinearities.rectify,
                                W=lasagne.init.GlorotUniform())
        maxpoollayer1 = MaxPool2DLayer(convlayer1, pool_size=pool_size1)
        convlayer2 = Conv2DLayer(maxpoollayer1, num_filters=num_filters2,
                                filter_size=filter_size1,
                                nonlinearity=lasagne.nonlinearities.rectify,
                                W=lasagne.init.GlorotUniform())
        maxpoollayer2 = MaxPool2DLayer(convlayer2, pool_size=pool_size2)
        dense1 = DenseLayer(
            lasagne.layers.dropout(maxpoollayer2, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
        return dense1

    l_branch_x = make_branch(l_in1_x, 32, (3, 3), (2, 2),
                             32, (3, 3), (2, 2))
    l_branch_u = make_branch(l_in1_u, 32, (3, 3), (2, 2),
                             32, (3, 3), (2, 2))
    l_branch_v = make_branch(l_in1_v, 32, (3, 3), (2, 2),
                             32, (3, 3), (2, 2))

    # Concatenate the two parallel inputs
    l_concat = ConcatLayer((l_branch_x, l_branch_u, l_branch_v))

    # And, finally, the 11-unit output layer with 50% dropout on its inputs:
    outp = DenseLayer(
        lasagne.layers.dropout(l_concat, p=.5),
        num_units=11,
        nonlinearity=lasagne.nonlinearities.softmax)

    return outp
