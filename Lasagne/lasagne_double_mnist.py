#!/usr/bin/env python
"""
This is a modification of the example MNIST network designed to do a "double"
input network.
"""
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

SAVE_MODEL_FILE = 'ldoublemnist_model.npz'
LOAD_PARAMS = True


def load_dataset():
    from urllib import urlretrieve
    import gzip

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in [0,~1].
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(input_var1=None, input_var2=None):
    # Input layer
    # We do not apply input dropout, as it tends to work less well
    # for convolutional layers.
    l_in1_1 = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var1)
    l_in1_2 = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var2)

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    l_conv2d1_1 = lasagne.layers.Conv2DLayer(
        l_in1_1, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_conv2d1_2 = lasagne.layers.Conv2DLayer(
        l_in1_2, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    l_maxpool1_1 = lasagne.layers.MaxPool2DLayer(l_conv2d1_1, pool_size=(2, 2))
    l_maxpool1_2 = lasagne.layers.MaxPool2DLayer(l_conv2d1_2, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    l_conv2d2_1 = lasagne.layers.Conv2DLayer(
        l_maxpool1_1, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
    l_maxpool2_1 = lasagne.layers.MaxPool2DLayer(l_conv2d2_1, pool_size=(2, 2))
    l_conv2d2_2 = lasagne.layers.Conv2DLayer(
        l_maxpool1_2, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
    l_maxpool2_2 = lasagne.layers.MaxPool2DLayer(l_conv2d2_2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    l_dense1_1 = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_maxpool2_1, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_dense1_2 = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_maxpool2_2, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # Concatenate the two parallel inputs
    l_concat = lasagne.layers.ConcatLayer((l_dense1_1, l_dense1_2))

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    outp = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_concat, p=.5),
        num_units=10,
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


def main(num_epochs=500, ):
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var1 = T.tensor4('inputs')
    input_var2 = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the model
    network = build_cnn(input_var1, input_var2)
    if LOAD_PARAMS:
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
        loss, params, learning_rate=0.01, momentum=0.9)

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
    train_fn = theano.function([input_var1, input_var2, target_var],
                               loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var1, input_var2, target_var],
                             [test_loss, test_acc])

    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

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
        err, acc = val_fn(inputs, inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Now dump the network weights to a file:
    np.savez(SAVE_MODEL_FILE,
             *lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)
