#!/usr/bin/env python
"""
This is an attempt at a "triamese" network operating on Minerva X, U, V.

Execution:
    python lasagne_triamese_minerva_simple.py -h / --help

    At a minimum, we must supply either the `--train` or `--predict` flag.
"""
# Several functions are borrowed from the Lasagne tutorials and docs.
# See, e.g.: http://lasagne.readthedocs.org/en/latest/user/tutorial.html
#
# Several functions and snippets probably inherit from the Theano docs as
# well. See, e.g.: http://deeplearning.net/tutorial/
#
from __future__ import print_function

import time
import os

import numpy as np
import theano
import theano.tensor as T

import lasagne

# TODO: pass these in on the command line...
SAVE_MODEL_FILE = './lminervatriamese_simple_model.npz'
START_WITH_SAVED_PARAMS = True


def load_dataset(data_file='./skim_data_convnet_target0.pkl.gz'):
    import gzip
    import cPickle

    # try-catch on no data
    f = gzip.open(data_file, 'rb')
    learn_data, test_data, valid_data = cPickle.load(f)
    f.close()

    X_learn = learn_data[0]
    y_learn = learn_data[1]
    X_valid = valid_data[0]
    y_valid = valid_data[1]
    X_test = test_data[0]
    y_test = test_data[1]

    # return all the arrays in order
    return X_learn, y_learn, X_valid, y_valid, X_test, y_test


def build_cnn(input_var_x=None, input_var_u=None, input_var_v=None):
    # Input layer
    l_in1_x = lasagne.layers.InputLayer(shape=(None, 1, 22, 50),
                                        input_var=input_var_x)
    l_in1_u = lasagne.layers.InputLayer(shape=(None, 1, 22, 50),
                                        input_var=input_var_u)
    l_in1_v = lasagne.layers.InputLayer(shape=(None, 1, 22, 50),
                                        input_var=input_var_v)

    # Convolutional layer with 32 kernels of size 5x5.
    # Filtering reduces the image to (22-3+1, 50-3+1) = (20, 48) (ndim-filt+1),
    # maxpooling reduces this further to (20/2, 48/2) = (10, 24), (dim/poolhw),
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
    # Max-pooling layer of factor 2 in both dimensions:
    l_maxpool1_x = lasagne.layers.MaxPool2DLayer(l_conv2d1_x,
                                                 pool_size=pool_size1)
    l_maxpool1_u = lasagne.layers.MaxPool2DLayer(l_conv2d1_u,
                                                 pool_size=pool_size1)
    l_maxpool1_v = lasagne.layers.MaxPool2DLayer(l_conv2d1_v,
                                                 pool_size=pool_size1)

    # More convolution and pooling layers...

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    l_dense1_x = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_maxpool1_x, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_dense1_u = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_maxpool1_u, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_dense1_v = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_maxpool1_v, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # Concatenate the two parallel inputs
    l_concat = lasagne.layers.ConcatLayer((l_dense1_x, l_dense1_u, l_dense1_v))

    # And, finally, the 6-unit output layer with 50% dropout on its inputs:
    outp = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_concat, p=.5),
        num_units=6,
        nonlinearity=lasagne.nonlinearities.softmax)

    return outp


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Simple helper function for iterating over training data in mini-batches
    of a given size, optionally in random order. It assumes the data are
    available as numpy arrays.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def split_inputs_xuv(inputs):
    """
    inputs has shape (# items, 3 views, w, h)
    we want to split into three 4-tensors, 1 for each view
    -> each should have shape: (# items, 1, w, h)
    """
    shpvar = np.shape(inputs)
    # print(shpvar)
    shpvar = (shpvar[0], 1, shpvar[2], shpvar[3])
    # print(shpvar)
    inputx = inputs[:, 0, :, :]
    inputu = inputs[:, 1, :, :]
    inputv = inputs[:, 2, :, :]
    inputx = np.reshape(inputx, shpvar)
    inputu = np.reshape(inputu, shpvar)
    inputv = np.reshape(inputv, shpvar)
    # print(np.shape(inputx))
    return inputx, inputu, inputv


def train(num_epochs=500, learning_rate=0.01, momentum=0.9,
          data_file=None):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = \
        load_dataset(data_file)

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the model
    network = build_cnn(input_var_x, input_var_u, input_var_v)
    if START_WITH_SAVED_PARAMS and os.path.isfile(SAVE_MODEL_FILE):
        with np.load(SAVE_MODEL_FILE) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for training.
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate, momentum=momentum)

    # Create a loss expression for validation/testing. Note we do a
    # deterministic forward pass through the network, disabling dropout.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
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
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            inputx, inputu, inputv = split_inputs_xuv(inputs)
            train_err += train_fn(inputx, inputu, inputv, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            inputx, inputu, inputv = split_inputs_xuv(inputs)
            err, acc = val_fn(inputx, inputu, inputv, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Dump the current network weights to file
        np.savez(SAVE_MODEL_FILE,
                 *lasagne.layers.get_all_param_values(network))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        inputx, inputu, inputv = split_inputs_xuv(inputs)
        err, acc = val_fn(inputx, inputu, inputv, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


def predict(data_file=None):
    print("Loading data for prediction...")
    _, _, _, _, X_test, y_test = load_dataset(data_file)

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the model
    network = build_cnn(input_var_x, input_var_u, input_var_v)
    with np.load(SAVE_MODEL_FILE) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for testing.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
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
    for batch in iterate_minibatches(X_test, y_test, 16, shuffle=False):
        inputs, targets = batch
        inputx, inputu, inputv = split_inputs_xuv(inputs)
        pred = pred_fn(inputx, inputu, inputv)
        print("predictions:", pred)
        print("targets:", targets)
        print("----------------")

    # compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        inputx, inputu, inputv = split_inputs_xuv(inputs)
        err, acc = val_fn(inputx, inputu, inputv, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data', dest='dataset',
                      default='./skim_data_convnet_target0.pkl.gz',
                      help='Data set', metavar='DATASET')
    parser.add_option('-n', '--nepochs', dest='n_epochs', default=200,
                      help='Number of epochs', metavar='N_EPOCHS',
                      type='int')
    parser.add_option('-r', '--rate', dest='lrate', default=0.01,
                      help='Learning rate', metavar='LRATE',
                      type='float')
    parser.add_option('-m', '--momentum', dest='momentum', default=0.9,
                      help='Momentum', metavar='MOMENTUM',
                      type='float')
    parser.add_option('-t', '--train', dest='do_train', default=False,
                      help='Run the training', metavar='DO_TRAIN',
                      action='store_true')
    parser.add_option('-p', '--predict', dest='do_predict', default=False,
                      help='Run a prediction', metavar='DO_PREDICT',
                      action='store_true')
    (options, args) = parser.parse_args()

    if not options.do_train and not options.do_predict:
        print("\nMust specify at least either train or predict:\n\n")
        print(__doc__)

    print("Starting...")
    print(" Begin with saved parameters?", START_WITH_SAVED_PARAMS)
    print(" Saved parameters file:", SAVE_MODEL_FILE)
    if options.do_train:
        train(num_epochs=options.n_epochs,
              learning_rate=options.lrate,
              momentum=options.momentum,
              data_file=options.dataset)

    if options.do_predict:
        predict(data_file=options.dataset)
