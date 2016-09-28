#!/usr/bin/env python
import logging

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import dropout
from lasagne.layers import flatten

logger = logging.getLogger(__name__)


def build_triamese_alpha(inputlist, imgh=50, imgw=50,
                         convpool1dict=None, convpool2dict=None,
                         convpooldictlist=None, nhidden=None,
                         dropoutp=None, noutputs=11):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using two layers of convolutions and pooling.
    """
    # Input layer
    input_var_x, input_var_u, input_var_v = \
        inputlist[0], inputlist[1], inputlist[2]
    tshape = (None, 1, imgw, imgh)
    l_in1_x = InputLayer(shape=tshape, input_var=input_var_x)
    l_in1_u = InputLayer(shape=tshape, input_var=input_var_u)
    l_in1_v = InputLayer(shape=tshape, input_var=input_var_v)

    if convpool1dict is None:
        convpool1dict = {}
        convpool1dict['nfilters'] = 32
        convpool1dict['filter_size'] = (3, 3)
        convpool1dict['pool_size'] = (2, 2)
    logger.info("Convpool1 params: {}".format(convpool1dict))

    if convpool2dict is None:
        convpool2dict = {}
        convpool2dict['nfilters'] = 32
        convpool2dict['filter_size'] = (3, 3)
        convpool2dict['pool_size'] = (2, 2)
    logger.info("Convpool2 params: {}".format(convpool2dict))
    logger.info("Network: one dense layer per column...")

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

    # Concatenate the parallel inputs
    l_concat = ConcatLayer((l_branch_x, l_branch_u, l_branch_v))
    logger.info("Network: Concat all three columns...")

    # And, finally, the noutputs-unit output layer
    outp = DenseLayer(
        l_concat,
        num_units=noutputs,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    logger.info("Network: Softmax classification layer.")

    logger.info("n-parameters: {}".format(lasagne.layers.count_params(outp)))
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


def make_Nconvpool_1dense_branch(view, input_layer, cpdictlist,
                                 nhidden=256, dropoutp=0.5):
    """
    see: http://lasagne.readthedocs.org/en/latest/modules/layers.html
    loop through the `cpdictlist` for set of convolutional filter and pooling
    parameter specifications; when through, add a dense layer with dropout
    """
    net = {}
    convname = ''
    mpname = ''
    for i, cpdict in enumerate(cpdictlist):
        convname = 'conv-{}-{}'.format(view, i)
        logger.info("Convpool {} params: {}".format(convname, cpdict))
        # the first time through, use `input`, after use the last layer
        # from the previous iteration - ah loose scoping rules...
        if i == 0:
            layer = input_layer
        else:
            layer = net[mpname]
        net[convname] = Conv2DLayer(
            layer, num_filters=cpdict['nfilters'],
            filter_size=cpdict['filter_size'],
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        mpname = 'maxpool-{}-{}'.format(view, i)
        net[mpname] = MaxPool2DLayer(
            net[convname], pool_size=cpdict['pool_size'])
        logger.info("Convpool {}".format(mpname))
    densename = 'dense-{}'.format(view)
    net[densename] = DenseLayer(
        dropout(net[mpname], p=dropoutp),
        num_units=nhidden,
        nonlinearity=lasagne.nonlinearities.rectify)
    logger.info("Dense {} with nhidden = {}, dropout = {}".format(
        densename, nhidden, dropoutp))
    return net


def build_triamese_inception(inputlist, imgh=50, imgw=50):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using a slightly modified set of Google inception modules
    """
    input_var_x, input_var_u, input_var_v = \
        inputlist[0], inputlist[1], inputlist[2]
    net = {}
    # Input layer
    tshape = (None, 1, imgw, imgh)
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

    logger.info("n-parameters: {}".format(
        lasagne.layers.count_params(net['output_prob']))
    )
    return net['output_prob']


def build_triamese_beta(inputlist, imgh=50, imgw=50, convpooldictlist=None,
                        nhidden=None, dropoutp=None, noutputs=11):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using two layers of convolutions and pooling.
    """
    net = {}
    # Input layer
    input_var_x, input_var_u, input_var_v = \
        inputlist[0], inputlist[1], inputlist[2]
    tshape = (None, 1, imgw, imgh)
    net['input-x'] = InputLayer(shape=tshape, input_var=input_var_x)
    net['input-u'] = InputLayer(shape=tshape, input_var=input_var_u)
    net['input-v'] = InputLayer(shape=tshape, input_var=input_var_v)

    if convpooldictlist is None:
        convpooldictlist = []
        convpool1dict = {}
        convpool1dict['nfilters'] = 32
        convpool1dict['filter_size'] = (3, 3)
        convpool1dict['pool_size'] = (2, 2)
        convpooldictlist.append(convpool1dict)
        convpool2dict = {}
        convpool2dict['nfilters'] = 32
        convpool2dict['filter_size'] = (3, 3)
        convpool2dict['pool_size'] = (2, 2)
        convpooldictlist.append(convpool2dict)

    if nhidden is None:
        nhidden = 256

    if dropoutp is None:
        dropoutp = 0.5

    net.update(
        make_Nconvpool_1dense_branch('x', net['input-x'], convpooldictlist,
                                     nhidden, dropoutp))
    net.update(
        make_Nconvpool_1dense_branch('u', net['input-u'], convpooldictlist,
                                     nhidden, dropoutp))
    net.update(
        make_Nconvpool_1dense_branch('v', net['input-v'], convpooldictlist,
                                     nhidden, dropoutp))

    # Concatenate the two parallel inputs
    net['concat'] = ConcatLayer((net['dense-x'],
                                 net['dense-u'],
                                 net['dense-v']))
    logger.info("Network: concat columns...")

    # One more dense layer
    net['dense-across'] = DenseLayer(
        dropout(net['concat'], p=dropoutp),
        num_units=(nhidden // 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    logger.info("Dense {} with nhidden = {}, dropout = {}".format(
        'dense-across', nhidden // 2, dropoutp))

    # And, finally, the `noutputs`-unit output layer
    net['output_prob'] = DenseLayer(
        net['dense-across'],
        num_units=noutputs,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    logger.info("Softmax output prob with n_units = {}".format(noutputs))

    logger.info("n-parameters: {}".format(
        lasagne.layers.count_params(net['output_prob']))
    )
    return net['output_prob']


def build_triamese_gamma(inputlist, imgh=50, imgw=50, convpooldictlist=None,
                         nhidden=None, dropoutp=None, noutputs=11):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using two layers of convolutions - no pooling.
    """
    net = {}
    # Input layer
    input_var_x, input_var_u, input_var_v = \
        inputlist[0], inputlist[1], inputlist[2]
    tshape = (None, 1, imgw, imgh)
    net['input-x'] = InputLayer(shape=tshape, input_var=input_var_x)
    net['input-u'] = InputLayer(shape=tshape, input_var=input_var_u)
    net['input-v'] = InputLayer(shape=tshape, input_var=input_var_v)

    if convpooldictlist is None:
        convpooldictlist = []
        convpool1dict = {}
        convpool1dict['nfilters'] = 32
        convpool1dict['filter_size'] = (3, 3)
        convpooldictlist.append(convpool1dict)
        convpool2dict = {}
        convpool2dict['nfilters'] = 16
        convpool2dict['filter_size'] = (3, 3)
        convpooldictlist.append(convpool2dict)

    if nhidden is None:
        nhidden = 256

    if dropoutp is None:
        dropoutp = 0.5

    def make_branch(view, input_layer, cpdictlist, nhidden=256, dropoutp=0.5):
        """
        see: http://lasagne.readthedocs.org/en/latest/modules/layers.html
        convolution only - no pooling
        """
        net = {}
        convname = ''
        prev_layername = ''
        for i, cpdict in enumerate(cpdictlist):
            convname = 'conv-{}-{}'.format(view, i)
            logger.info("Convpool {} params: {}".format(convname, cpdict))
            # the first time through, use `input`, after use the last layer
            # from the previous iteration - ah loose scoping rules...
            if i == 0:
                layer = input_layer
            else:
                layer = net[prev_layername]
            net[convname] = Conv2DLayer(
                layer, num_filters=cpdict['nfilters'],
                filter_size=cpdict['filter_size'],
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
            prev_layername = convname
        densename = 'dense-{}'.format(view)
        net[densename] = DenseLayer(
            dropout(net[convname], p=dropoutp),
            num_units=nhidden,
            nonlinearity=lasagne.nonlinearities.rectify)
        logger.info("Dense {} with nhidden = {}, dropout = {}".format(
            densename, nhidden, dropoutp))
        return net

    net.update(make_branch('x', net['input-x'], convpooldictlist,
                           nhidden, dropoutp))
    net.update(make_branch('u', net['input-u'], convpooldictlist,
                           nhidden, dropoutp))
    net.update(make_branch('v', net['input-v'], convpooldictlist,
                           nhidden, dropoutp))

    # Concatenate the two parallel inputs
    net['concat'] = ConcatLayer((net['dense-x'],
                                 net['dense-u'],
                                 net['dense-v']))
    logger.info("Network: concat columns...")

    # One more dense layer
    net['dense-across'] = DenseLayer(
        dropout(net['concat'], p=dropoutp),
        num_units=(nhidden // 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    logger.info("Dense {} with nhidden = {}, dropout = {}".format(
        'dense-across', nhidden // 2, dropoutp))

    # And, finally, the `noutputs`-unit output layer
    net['output_prob'] = DenseLayer(
        net['dense-across'],
        num_units=noutputs,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    logger.info("Softmax output prob with n_units = {}".format(noutputs))

    logger.info("n-parameters: {}".format(
        lasagne.layers.count_params(net['output_prob']))
    )
    return net['output_prob']


def build_triamese_delta(inputlist, imgh=68, imgw=127, convpooldictlist=None,
                         nhidden=None, dropoutp=None, noutputs=67):
    """
    'triamese' (one branch for each view, feeding a fully-connected network),
    model using two layers of convolutions and pooling.

    This model is basically identical to the `beta` model, except we have
    a softmax output of `noutputs` (def 67) for the full set of planecodes.
    """
    net = {}
    # Input layer
    input_var_x, input_var_u, input_var_v = \
        inputlist[0], inputlist[1], inputlist[2]
    tshape = (None, 1, imgw, imgh)
    net['input-x'] = InputLayer(shape=tshape, input_var=input_var_x)
    net['input-u'] = InputLayer(shape=tshape, input_var=input_var_u)
    net['input-v'] = InputLayer(shape=tshape, input_var=input_var_v)

    if convpooldictlist is None:
        convpooldictlist = []
        convpool1dict = {}
        convpool1dict['nfilters'] = 32
        convpool1dict['filter_size'] = (3, 3)
        convpool1dict['pool_size'] = (2, 2)
        convpooldictlist.append(convpool1dict)
        convpool2dict = {}
        convpool2dict['nfilters'] = 32
        convpool2dict['filter_size'] = (3, 3)
        convpool2dict['pool_size'] = (2, 2)
        convpooldictlist.append(convpool2dict)

    if nhidden is None:
        nhidden = 256

    if dropoutp is None:
        dropoutp = 0.5

    net.update(
        make_Nconvpool_1dense_branch('x', net['input-x'], convpooldictlist,
                                     nhidden, dropoutp))
    net.update(
        make_Nconvpool_1dense_branch('u', net['input-u'], convpooldictlist,
                                     nhidden, dropoutp))
    net.update(
        make_Nconvpool_1dense_branch('v', net['input-v'], convpooldictlist,
                                     nhidden, dropoutp))

    # Concatenate the two parallel inputs
    net['concat'] = ConcatLayer((net['dense-x'],
                                 net['dense-u'],
                                 net['dense-v']))
    logger.info("Network: concat columns...")

    # One more dense layer
    net['dense-across'] = DenseLayer(
        dropout(net['concat'], p=dropoutp),
        num_units=(nhidden // 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    logger.info("Dense {} with nhidden = {}, dropout = {}".format(
        'dense-across', nhidden // 2, dropoutp))

    # And, finally, the `noutputs`-unit output layer
    net['output_prob'] = DenseLayer(
        net['dense-across'],
        num_units=noutputs,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    logger.info("Softmax output prob with n_units = {}".format(noutputs))

    logger.info("n-parameters: {}".format(
        lasagne.layers.count_params(net['output_prob']))
    )
    return net['output_prob']


def build_beta_single_view(inputlist, view='x', imgh=68, imgw=127,
                           convpooldictlist=None,
                           nhidden=None, dropoutp=None, noutputs=11):
    """
    This network is modeled after the 'triamese' (tri-columnar) beta model,
    but is meant to operate on one view only.

    This function has a differen signature than the rest of the functions
    in this module, so it is really not meant to be used as a `build_cnn`
    function in the runner scripts (although, in Python, that would work).
    """
    net = {}
    # Input layer
    input_var = inputlist[0]
    tshape = (None, 1, imgw, imgh)
    input_name = 'input-' + view
    net[input_name] = InputLayer(shape=tshape, input_var=input_var)

    if convpooldictlist is None:
        convpooldictlist = []
        convpool1dict = {}
        convpool1dict['nfilters'] = 32
        convpool1dict['filter_size'] = (3, 3)
        convpool1dict['pool_size'] = (2, 2)
        convpooldictlist.append(convpool1dict)
        convpool2dict = {}
        convpool2dict['nfilters'] = 32
        convpool2dict['filter_size'] = (3, 3)
        convpool2dict['pool_size'] = (2, 2)
        convpooldictlist.append(convpool2dict)

    if nhidden is None:
        nhidden = 256

    if dropoutp is None:
        dropoutp = 0.5

    net.update(
        make_Nconvpool_1dense_branch(view, net[input_name], convpooldictlist,
                                     nhidden, dropoutp))

    # One more dense layer
    dense_name = 'dense-' + view
    net['dense-across'] = DenseLayer(
        dropout(net[dense_name], p=dropoutp),
        num_units=(nhidden // 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    logger.info("Dense {} with nhidden = {}, dropout = {}".format(
        'dense-across', nhidden // 2, dropoutp))

    # And, finally, the `noutputs`-unit output layer
    net['output_prob'] = DenseLayer(
        net['dense-across'],
        num_units=noutputs,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    logger.info("Softmax output prob with n_units = {}".format(noutputs))

    logger.info("n-parameters: {}".format(
        lasagne.layers.count_params(net['output_prob']))
    )
    return net['output_prob']


def build_beta_x(inputlist, imgh=68, imgw=127, convpooldictlist=None,
                 nhidden=None, dropoutp=None, noutputs=11):
    """
    This network is modeled after the 'triamese' (tri-columnar) beta model,
    but is meant to operate on the u-view only.
    """
    return build_beta_single_view(inputlist=inputlist, view='x',
                                  imgh=imgh, imgw=imgw,
                                  convpooldictlist=convpooldictlist,
                                  nhidden=nhidden, dropoutp=dropoutp,
                                  noutputs=noutputs)


def build_beta_u(inputlist, imgh=68, imgw=127, convpooldictlist=None,
                 nhidden=None, dropoutp=None, noutputs=11):
    """
    This network is modeled after the 'triamese' (tri-columnar) beta model,
    but is meant to operate on the u-view only.
    """
    return build_beta_single_view(inputlist=inputlist, view='u',
                                  imgh=imgh, imgw=imgw,
                                  convpooldictlist=convpooldictlist,
                                  nhidden=nhidden, dropoutp=dropoutp,
                                  noutputs=noutputs)


def build_beta_v(inputlist, imgh=68, imgw=127, convpooldictlist=None,
                 nhidden=None, dropoutp=None, noutputs=11):
    """
    This network is modeled after the 'triamese' (tri-columnar) beta model,
    but is meant to operate on the v-view only.
    """
    return build_beta_single_view(inputlist=inputlist, view='v',
                                  imgh=imgh, imgw=imgw,
                                  convpooldictlist=convpooldictlist,
                                  nhidden=nhidden, dropoutp=dropoutp,
                                  noutputs=noutputs)


def build_triamese_epsilon(inputlist, imgh=(50, 25, 25), imgw=127,
                           convpooldictlist=None,
                           nhidden=None, dropoutp=None, noutputs=11):
    """
    'triamese' (one branch for each view, feeding a fully-connected network)

    here, `imgh` is a tuple of sizes for `(x, u, v)`. `imgw` is the same
    for all three views.

    also, the `convpooldictlist` here must be a dictionary of dictionaries,
    with the set of convolution and pooling defined independently for 'x', 'u',
    and 'v' - e.g., `convpooldictlist['x']` will be a dictionary similar to
    the dictionaries used by network models like `beta`, etc.
    """
    net = {}
    # Input layer
    input_var_x, input_var_u, input_var_v = \
        inputlist[0], inputlist[1], inputlist[2]
    net['input-x'] = InputLayer(shape=(None, 1, imgw, imgh[0]),
                                input_var=input_var_x)
    net['input-u'] = InputLayer(shape=(None, 1, imgw, imgh[1]),
                                input_var=input_var_u)
    net['input-v'] = InputLayer(shape=(None, 1, imgw, imgh[2]),
                                input_var=input_var_v)

    if convpooldictlist is None:
        raise Exception('Conv-pool dictionaries must be defined!')

    if nhidden is None:
        nhidden = 256

    if dropoutp is None:
        dropoutp = 0.5

    net.update(
        make_Nconvpool_1dense_branch('x', net['input-x'],
                                     convpooldictlist['x'],
                                     nhidden, dropoutp))
    net.update(
        make_Nconvpool_1dense_branch('u', net['input-u'],
                                     convpooldictlist['u'],
                                     nhidden, dropoutp))
    net.update(
        make_Nconvpool_1dense_branch('v', net['input-v'],
                                     convpooldictlist['v'],
                                     nhidden, dropoutp))

    # Concatenate the two parallel inputs
    net['concat'] = ConcatLayer((net['dense-x'],
                                 net['dense-u'],
                                 net['dense-v']))
    logger.info("Network: concat columns...")

    # One more dense layer
    net['dense-across'] = DenseLayer(
        dropout(net['concat'], p=dropoutp),
        num_units=(nhidden // 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    logger.info("Dense {} with nhidden = {}, dropout = {}".format(
        'dense-across', nhidden // 2, dropoutp))

    # And, finally, the `noutputs`-unit output layer
    net['output_prob'] = DenseLayer(
        net['dense-across'],
        num_units=noutputs,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    logger.info("Softmax output prob with n_units = {}".format(noutputs))

    logger.info(
        "n-parameters: %s" % lasagne.layers.count_params(net['output_prob'])
    )
    return net['output_prob']


def build_network_zeta(inputlist,
                       imgh=(50, 25, 25),
                       imgw=127,
                       convpooldictlist=None,
                       nhidden=None,
                       dropoutp=None,
                       noutputs=11):
    """
    here, `inputlist` should have img tensors for x, u, v, and for muon_data

    here, `imgh` is a tuple of sizes for `(x, u, v)`. `imgw` is the same
    for all three views.

    also, the `convpooldictlist` here must be a dictionary of dictionaries,
    with the set of convolution and pooling defined independently for 'x', 'u',
    and 'v' - e.g., `convpooldictlist['x']` will be a dictionary similar to
    the dictionaries used by network models like `beta`, etc.
    """
    net = {}
    # Input layer
    input_var_x, input_var_u, input_var_v, input_var_muon = \
        inputlist[0], inputlist[1], inputlist[2], inputlist[3]
    net['input-x'] = InputLayer(shape=(None, 1, imgw, imgh[0]),
                                input_var=input_var_x)
    net['input-u'] = InputLayer(shape=(None, 1, imgw, imgh[1]),
                                input_var=input_var_u)
    net['input-v'] = InputLayer(shape=(None, 1, imgw, imgh[2]),
                                input_var=input_var_v)
    net['input-muon-dat'] = InputLayer(shape=(None, 10),
                                       input_var=input_var_muon)

    if convpooldictlist is None:
        raise Exception('Conv-pool dictionaries must be defined!')

    if nhidden is None:
        nhidden = 256

    if dropoutp is None:
        dropoutp = 0.5

    net.update(
        make_Nconvpool_1dense_branch('x', net['input-x'],
                                     convpooldictlist['x'],
                                     nhidden, dropoutp))
    net.update(
        make_Nconvpool_1dense_branch('u', net['input-u'],
                                     convpooldictlist['u'],
                                     nhidden, dropoutp))
    net.update(
        make_Nconvpool_1dense_branch('v', net['input-v'],
                                     convpooldictlist['v'],
                                     nhidden, dropoutp))

    # Concatenate the parallel inputs, include the muon data
    net['concat'] = ConcatLayer((
        net['dense-x'],
        net['dense-u'],
        net['dense-v'],
        net['input-muon-dat']
    ))
    logger.info("Network: concat columns...")

    # One more dense layer
    net['dense-across'] = DenseLayer(
        dropout(net['concat'], p=dropoutp),
        num_units=(nhidden // 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    logger.info("Dense {} with nhidden = {}, dropout = {}".format(
        'dense-across', nhidden // 2, dropoutp))

    # And, finally, the `noutputs`-unit output layer
    net['output_prob'] = DenseLayer(
        net['dense-across'],
        num_units=noutputs,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    logger.info("Softmax output prob with n_units = {}".format(noutputs))

    logger.info(
        "n-parameters: %s" % lasagne.layers.count_params(net['output_prob'])
    )
    return net['output_prob']
