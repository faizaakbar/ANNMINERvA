#!/usr/bin/env python
"""
This is an attempt at a "triamese" network operating on Minerva X, U, V.

Execution:
    python minerva_triamese_lasagnefuel.py -h / --help

At a minimum, we must supply either the `--train` or `--predict` flag.

See ANNMINERvA/fuel_up_convdata.py for an HDF5 builder that sets up an
appropriate data file.

"""
from __future__ import print_function

import time
import os

import numpy as np
import theano
import theano.tensor as T

import lasagne
import network_repr
from lasagne.objectives import categorical_crossentropy

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from six.moves import range


def load_dataset(data_file, load_in_memory=False):
    """
    See ANNMINERvA/fuel_up_convdata.py for an HDF5 builder that sets up an
    appropriate data file.
    """
    if os.path.exists(data_file):
        train_set = H5PYDataset(data_file, which_sets=('train',),
                                load_in_memory=load_in_memory)
        valid_set = H5PYDataset(data_file, which_sets=('valid',),
                                load_in_memory=load_in_memory)
        test_set = H5PYDataset(data_file, which_sets=('test',),
                                load_in_memory=load_in_memory)
    else:
        raise Exception('Data file', data_file, 'not found!')

    return train_set, valid_set, test_set


def make_scheme_and_stream(dset, batchsize, msg_string, shuffle=True):
    """
    dset is a Fuel `DataSet` and batchsize is an int representing the number of
    examples requested per minibatch
    """
    if shuffle:
        print(msg_string +
              " Preparing shuffled datastream for {} examples.".format(
                  dset.num_examples))
        scheme = ShuffledScheme(examples=dset.num_examples,
                                batch_size=batchsize)
    else:
        print(msg_string +
              "Preparing sequential datastream for {} examples.".format(
                  dset.num_examples))
        scheme = SequentialScheme(examples=dset.num_examples,
                                  batch_size=batchsize)
    data_stream = DataStream(dataset=dset,
                             iteration_scheme=scheme)
    return scheme, data_stream


def build_cnn(input_var_x=None, input_var_u=None, input_var_v=None):
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


def split_inputs_xuv(inputs):
    """
    inputs has shape (# items, 3 views, w, h)
    we want to split into three 4-tensors, 1 for each view
    -> each should have shape: (# items, 1, w, h)

    TODO: pre-split while preparing HDF5 files so we don't waste time doing
    this every time we run...
    """
    shpvar = np.shape(inputs)
    shpvar = (shpvar[0], 1, shpvar[2], shpvar[3])
    inputx = inputs[:, 0, :, :]
    inputu = inputs[:, 1, :, :]
    inputv = inputs[:, 2, :, :]
    inputx = np.reshape(inputx, shpvar)
    inputu = np.reshape(inputu, shpvar)
    inputv = np.reshape(inputv, shpvar)
    return inputx, inputu, inputv


def train(num_epochs=500, learning_rate=0.01, momentum=0.9,
          l2_penalty_scale=1e-04, batchsize=500,
          data_file=None, save_model_file='./params_file.npz',
          start_with_saved_params=False, load_in_memory=False):
    print("Loading data...")
    train_set, valid_set, _ = load_dataset(data_file, load_in_memory)

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the model
    network = build_cnn(input_var_x, input_var_u, input_var_v)
    print(network_repr.get_network_str(
        lasagne.layers.get_all_layers(network),
        get_network=False, incomings=True, outgoings=True))
    if start_with_saved_params and os.path.isfile(save_model_file):
        with np.load(save_model_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for training.
    prediction = lasagne.layers.get_output(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(
        lasagne.layers.get_all_layers(network),
        lasagne.regularization.l2) * l2_penalty_scale
    loss = categorical_crossentropy(prediction, target_var) + l2_penalty
    loss = loss.mean()

    # Create update expressions for training.
    params = lasagne.layers.get_all_params(network, trainable=True)
    print(
        """
        ////
        Use AdaGrad update schedule for learning rate, see Duchi, Hazan, and
        Singer (2011) "Adaptive subgradient methods for online learning and
        stochasitic optimization." JMLR, 12:2121-2159
        ////
        """)
    updates_adagrad = lasagne.updates.adagrad(
        loss, params, learning_rate=learning_rate, epsilon=1e-06)
    print(
        """
        ////
        Apply Nesterov momentum using Lisa Lab's modifications.
        ////
        """)
    updates = lasagne.updates.apply_nesterov_momentum(
        updates_adagrad, params, momentum=momentum)

    # Create a loss expression for validation/testing. Note we do a
    # deterministic forward pass through the network, disabling dropout.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = categorical_crossentropy(test_prediction, target_var) + \
        l2_penalty
    test_loss = test_loss.mean()
    # Also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var_x, input_var_u, input_var_v,
                                target_var],
                               loss, updates=updates,
                               allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var_x, input_var_u, input_var_v,
                              target_var],
                             [test_loss, test_acc],
                             allow_input_downcast=True)

    print("Starting training...")
    _, train_dstream = make_scheme_and_stream(train_set, batchsize,
                                              "Preparing training data:")
    _, valid_dstream = make_scheme_and_stream(valid_set, batchsize,
                                              "Preparing validation data:")

    #
    # TODO: early stopping logic goes here...
    #

    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for data in train_dstream.get_epoch_iterator():
            _, inputs, targets = data[0], data[1], data[2]
            inputx, inputu, inputv = split_inputs_xuv(inputs)
            train_err += train_fn(inputx, inputu, inputv, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for data in valid_dstream.get_epoch_iterator():
            _, inputs, targets = data[0], data[1], data[2]
            inputx, inputu, inputv = split_inputs_xuv(inputs)
            err, acc = val_fn(inputx, inputu, inputv, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Dump the current network weights to file
        np.savez(save_model_file,
                 *lasagne.layers.get_all_param_values(network))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    print("Finished {} epochs.".format(epoch + 1))


def predict(data_file, l2_penalty_scale, save_model_file='./params_file.npz',
            batchsize=500, load_in_memory=False, be_verbose=False):
    print("Loading data for prediction...")
    _, _, test_set = load_dataset(data_file, load_in_memory)

    # extract timestamp from model file - assume it is the first set of numbers
    # otherwise just use "now"
    import re
    import time
    tstamp = str(time.time()).split('.')[0]
    m = re.search(r"[0-9]+", save_model_file)
    if m:
        tstamp = m.group(0)

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the model
    network = build_cnn(input_var_x, input_var_u, input_var_v)
    with np.load(save_model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for testing.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    l2_penalty = lasagne.regularization.regularize_layer_params(
        lasagne.layers.get_all_layers(network),
        lasagne.regularization.l2) * l2_penalty_scale
    test_loss = categorical_crossentropy(test_prediction, target_var) + \
                l2_penalty
    test_loss = test_loss.mean()
    # Also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    # Look at the classifications
    test_prediction_values = T.argmax(test_prediction, axis=1)

    # Compile a function computing the validation loss and accuracy:
    val_fn = theano.function([input_var_x, input_var_u, input_var_v,
                              target_var],
                             [test_loss, test_acc],
                             allow_input_downcast=True)
    # Compute the actual predictions - also instructive is to look at
    # `test_prediction` as an output (array of softmax probabilities)
    # (but that prints a _lot_ of stuff to screen...)
    pred_fn = theano.function([input_var_x, input_var_u, input_var_v],
                              [test_prediction_values],
                              allow_input_downcast=True)

    # look at some concrete predictions
    targ_numbers = [1, 2, 3, 4, 5]
    pred_target = np.array([0, 0, 0, 0, 0])
    true_target = np.array([0, 0, 0, 0, 0])
    targs_mat = np.zeros(11 * 11).reshape(11, 11)
    _, test_dstream = make_scheme_and_stream(test_set, 5,
                                             "Preparing test data:",
                                             shuffle=False)
    for data in test_dstream.get_epoch_iterator():
        _, inputs, targets = data[0], data[1], data[2]
        inputx, inputu, inputv = split_inputs_xuv(inputs)
        pred = pred_fn(inputx, inputu, inputv)
        pred_targ = zip(pred[0], targets)
        if be_verbose:
            print("(prediction, true target):", pred_targ)
            print("----------------")
        for p, t in pred_targ:
            targs_mat[t][p] += 1
            if t in targ_numbers:
                true_target[t-1] += 1
                if p == t:
                    pred_target[p-1] += 1

    acc_target = 100.0 * pred_target / true_target.astype('float32')
    perf_file = 'perfmat' + tstamp + '.npy'
    np.save(perf_file, targs_mat)

    # compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for data in test_dstream.get_epoch_iterator():
        _, inputs, targets = data[0], data[1], data[2]
        inputx, inputu, inputv = split_inputs_xuv(inputs)
        err, acc = val_fn(inputx, inputu, inputv, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    for i, v in enumerate(acc_target):
        print("   target {} accuracy:\t\t\t{:.3f} %".format(
            (i + 1), acc_target[i]))


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data', dest='dataset',
                      default='./skim_data_convnet.hdf5',
                      help='Data set', metavar='DATASET')
    parser.add_option('-n', '--nepochs', dest='n_epochs', default=200,
                      help='Number of epochs', metavar='N_EPOCHS',
                      type='int')
    parser.add_option('-b', '--batch_size', dest='batchsize', default=500,
                      help='Batch size for SGD', metavar='BATCH_SIZE',
                      type='int')
    parser.add_option('-r', '--rate', dest='lrate', default=0.01,
                      help='Learning rate', metavar='LRATE',
                      type='float')
    parser.add_option('-g', '--regularization', dest='l2_penalty_scale',
                      default=1e-04, help='L2 regularization scale',
                      metavar='L2_REG_SCALE', type='float')
    parser.add_option('-m', '--momentum', dest='momentum', default=0.9,
                      help='Momentum', metavar='MOMENTUM',
                      type='float')
    parser.add_option('-t', '--train', dest='do_train', default=False,
                      help='Run the training', metavar='DO_TRAIN',
                      action='store_true')
    parser.add_option('-p', '--predict', dest='do_predict', default=False,
                      help='Run a prediction', metavar='DO_PREDICT',
                      action='store_true')
    parser.add_option('-v', '--verbose', dest='be_verbose', default=False,
                      help='Verbose predictions', metavar='BE_VERBOSE',
                      action='store_true')
    parser.add_option('-s', '--save_file', dest='save_model_file',
                      default='./lminervatriamese_model.npz',
                      help='File name for parameters',
                      metavar='SAVE_FILE_NAME')
    parser.add_option('-l', '--load_params', dest='start_with_saved_params',
                      default=False, help='Begin training with saved pars',
                      metavar='LOAD_PARAMS', action='store_true')
    parser.add_option('-y', '--load_in_memory', dest='load_in_memory',
                      default=False, help='Attempt to load full dset in memory',
                      metavar='LOAD_IN_MEMORY', action='store_true')
    (options, args) = parser.parse_args()

    if not options.do_train and not options.do_predict:
        print("\nMust specify at least either train or predict:\n\n")
        print(__doc__)

    print("Starting...")
    print(__file__)
    print(" Begin with saved parameters?", options.start_with_saved_params)
    print(" Saved parameters file:", options.save_model_file)
    print(" Saved parameters file exists?",
          os.path.isfile(options.save_model_file))
    print(" Dataset:", options.dataset)
    dataset_statsinfo = os.stat(options.dataset)
    print(" Dataset size:", dataset_statsinfo.st_size)
    print(" Planned number of epochs:", options.n_epochs)
    print(" Learning rate:", options.lrate)
    print(" Momentum:", options.momentum)
    print(" L2 regularization penalty scale:", options.l2_penalty_scale)
    print(" Batch size:", options.batchsize)

    if options.do_train:
        train(num_epochs=options.n_epochs,
              learning_rate=options.lrate,
              momentum=options.momentum,
              l2_penalty_scale=options.l2_penalty_scale,
              batchsize=options.batchsize,
              data_file=options.dataset,
              save_model_file=options.save_model_file,
              start_with_saved_params=options.start_with_saved_params,
              load_in_memory=options.load_in_memory)

    if options.do_predict:
        predict(data_file=options.dataset,
                l2_penalty_scale=options.l2_penalty_scale,
                save_model_file=options.save_model_file,
                batchsize=options.batchsize,
                load_in_memory=options.load_in_memory,
                be_verbose=options.be_verbose)
