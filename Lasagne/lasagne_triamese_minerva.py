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
import network_repr


def load_dataset(data_file='./skim_data_convnet_target0.pkl.gz'):
    if os.path.exists(data_file):
        # check pickle vs hdf5
        name_parts = data_file.split('.')
        if 'pkl' in name_parts or 'pickle' in name_parts:
            import gzip
            import cPickle
            f = gzip.open(data_file, 'rb')
            learn_data, test_data, valid_data = cPickle.load(f)
            f.close()
            X_learn = learn_data[0]  # x, u, v
            y_learn = learn_data[1]  # targets
            X_valid = valid_data[0]
            y_valid = valid_data[1]  # etc.
            X_test = test_data[0]
            y_test = test_data[1]
        elif 'hdf5' in name_parts:
            import h5py
            f = h5py.File(data_file, 'r')
            # learn
            X_learn = np.zeros(np.shape(f['learn/hits']), dtype='f')
            y_learn = np.zeros(np.shape(f['learn/segments']), dtype='f')
            f['learn/hits'].read_direct(X_learn)
            f['learn/segments'].read_direct(y_learn)
            print("Learning data size:", np.shape(f['learn/hits']))
            # valid
            X_valid = np.zeros(np.shape(f['valid/hits']), dtype='f')
            y_valid = np.zeros(np.shape(f['valid/segments']), dtype='f')
            f['valid/hits'].read_direct(X_valid)
            f['valid/segments'].read_direct(y_valid)
            # test
            X_test = np.zeros(np.shape(f['test/hits']), dtype='f')
            y_test = np.zeros(np.shape(f['test/segments']), dtype='f')
            f['test/hits'].read_direct(X_test)
            f['test/segments'].read_direct(y_test)
            f.close()
    else:
        raise Exception('Data file', data_file, 'not found!')

    # return all the arrays in order
    return X_learn, y_learn, X_valid, y_valid, X_test, y_test


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
          data_file=None, save_model_file='./params_file.npz',
          start_with_saved_params=False):
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
    print(network_repr.get_network_str(
        lasagne.layers.get_all_layers(network),
        get_network=False, incomings=True, outgoings=True))
    if start_with_saved_params and os.path.isfile(save_model_file):
        with np.load(save_model_file) as f:
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
        np.savez(save_model_file,
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


def predict(data_file=None, save_model_file='./params_file.npz'):
    print("Loading data for prediction...")
    _, _, _, _, X_test, y_test = load_dataset(data_file)

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
    targ_numbers = [1, 2, 3, 4, 5]
    pred_target = np.array([0, 0, 0, 0, 0])
    true_target = np.array([0, 0, 0, 0, 0])
    for batch in iterate_minibatches(X_test, y_test, 5, shuffle=False):
        inputs, targets = batch
        inputx, inputu, inputv = split_inputs_xuv(inputs)
        pred = pred_fn(inputx, inputu, inputv)
        pred_targ = zip(pred[0], targets)
        print("predictions :", pred)
        print("true targets:", targets)
        print("(prediction, true target):", pred_targ)
        print("----------------")
        for p, t in pred_targ:
            if t in targ_numbers:
                true_target[t-1] += 1
                if p == t:
                    pred_target[p-1] += 1

    acc_target = 100.0 * pred_target / true_target.astype('float32')

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
    parser.add_option('-s', '--save_file', dest='save_model_file',
                      default='./lminervatriamese_model.npz',
                      help='File name for parameters',
                      metavar='SAVE_FILE_NAME')
    parser.add_option('-l', '--load_params', dest='start_with_saved_params',
                      default=False, help='Begin training with saved pars',
                      metavar='LOAD_PARAMS', action='store_true')
    (options, args) = parser.parse_args()

    if not options.do_train and not options.do_predict:
        print("\nMust specify at least either train or predict:\n\n")
        print(__doc__)

    print("Starting...")
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

    if options.do_train:
        train(num_epochs=options.n_epochs,
              learning_rate=options.lrate,
              momentum=options.momentum,
              data_file=options.dataset,
              save_model_file=options.save_model_file,
              start_with_saved_params=options.start_with_saved_params)

    if options.do_predict:
        predict(data_file=options.dataset,
                save_model_file=options.save_model_file)
