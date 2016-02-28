#!/usr/bin/env python
from __future__ import print_function

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import dropout
from lasagne.layers import flatten


def build_triamese_alpha(input_var_x=None, input_var_u=None, input_var_v=None,
                         imgh=50, imgw=50,
                         convpool1dict=None, convpool2dict=None):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using two layers of convolutions and pooling.
    """
    # Input layer
    tshape = (None, 1, imgh, imgw)
    l_in1_x = InputLayer(shape=tshape, input_var=input_var_x)
    l_in1_u = InputLayer(shape=tshape, input_var=input_var_u)
    l_in1_v = InputLayer(shape=tshape, input_var=input_var_v)

    if convpool1dict is None:
        convpool1dict = {}
        convpool1dict['nfilters'] = 32
        convpool1dict['filter_size'] = (3, 3)
        convpool1dict['pool_size'] = (2, 2)
    print("Convpool1 params:", convpool1dict)

    if convpool2dict is None:
        convpool2dict = {}
        convpool2dict['nfilters'] = 32
        convpool2dict['filter_size'] = (3, 3)
        convpool2dict['pool_size'] = (2, 2)
    print("Convpool2 params:", convpool2dict)

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
            dropout(maxpoollayer2, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
        return dense1

    l_branch_x = make_branch(l_in1_x,
                             convpool1dict['nfilters'],
                             convpool1dict['filter_size'],
                             convpool1dict['pool_size'],
                             convpool2dict['nfilters'],
                             convpool2dict['filter_size'],
                             convpool2dict['pool_size'])
    l_branch_u = make_branch(l_in1_u, 
                             convpool1dict['nfilters'],
                             convpool1dict['filter_size'],
                             convpool1dict['pool_size'],
                             convpool2dict['nfilters'],
                             convpool2dict['filter_size'],
                             convpool2dict['pool_size'])
    l_branch_v = make_branch(l_in1_v,
                             convpool1dict['nfilters'],
                             convpool1dict['filter_size'],
                             convpool1dict['pool_size'],
                             convpool2dict['nfilters'],
                             convpool2dict['filter_size'],
                             convpool2dict['pool_size'])

    # Concatenate the two parallel inputs
    l_concat = ConcatLayer((l_branch_x, l_branch_u, l_branch_v))

    # And, finally, the 11-unit output layer with 50% dropout on its inputs:
    outp = DenseLayer(
        dropout(l_concat, p=.5),
        num_units=11,
        nonlinearity=lasagne.nonlinearities.softmax)

    return outp


def build_inception_module(name, input_layer, nfilters):
    """
    See: https://github.com/Lasagne/Recipes/blob/master/modelzoo/googlenet.py
    TODO: the pooling size is wrong here - 3x3 doesn't really make sense
    on our images - consider recomputing the pooling -> also requires
    recomputing all the filter sizes, etc.
    """
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = MaxPool2DLayer(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = Conv2DLayer(net['pool'], nfilters[0], 1)

    net['1x1'] = Conv2DLayer(input_layer, nfilters[1], 1)

    net['3x3_reduce'] = Conv2DLayer(input_layer, nfilters[2], 1)
    net['3x3'] = Conv2DLayer(net['3x3_reduce'], nfilters[3], 3, pad=1)

    net['5x5_reduce'] = Conv2DLayer(input_layer, nfilters[4], 1)
    net['5x5'] = Conv2DLayer(net['5x5_reduce'], nfilters[5], 5, pad=2)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
    ])
    
    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def build_triamese_inception(input_var_x=None,
                             input_var_u=None,
                             input_var_v=None, imgh=50, imgw=50):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using a slightly modified set of Google inception modules
    """
    net = {}
    # Input layer
    tshape = (None, 1, imgh, imgw)
    net['input_x'] = InputLayer(shape=tshape, input_var=input_var_x)
    net['input_u'] = InputLayer(shape=tshape, input_var=input_var_u)
    net['input_v'] = InputLayer(shape=tshape, input_var=input_var_v)

    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    nfilters = [32, 64, 96, 128, 16, 32]
    net.update(build_inception_module('inc_x1', net['input_x'], nfilters))
    net.update(build_inception_module('inc_u1', net['input_u'], nfilters))
    net.update(build_inception_module('inc_v1', net['input_v'], nfilters))

    net['dense_x'] = DenseLayer(
        dropout(flatten(net['inc_x1/output']), p=.5),
        num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
    net['dense_u'] = DenseLayer(
        dropout(flatten(net['inc_u1/output']), p=.5),
        num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
    net['dense_v'] = DenseLayer(
        dropout(flatten(net['inc_v1/output']), p=.5),
        num_units=100, nonlinearity=lasagne.nonlinearities.rectify)

    # Concatenate the parallel inputs
    net['concat'] = ConcatLayer((net['dense_x'],
                                 net['dense_u'],
                                 net['dense_v']))

    # And, finally, the 11-unit output layer with 50% dropout on its inputs:
    net['output_prob'] = DenseLayer(
        dropout(net['concat'], p=.5),
        num_units=11,
        nonlinearity=lasagne.nonlinearities.softmax)

    return net['output_prob']
