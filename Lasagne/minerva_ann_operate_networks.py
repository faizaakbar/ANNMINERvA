#!/usr/bin/env python
from __future__ import print_function

import os
import time
from six.moves import range

import numpy as np
import theano
import theano.tensor as T

import lasagne
import network_repr
from lasagne.objectives import categorical_crossentropy

from minerva_ann_datasets import load_datasubset
from minerva_ann_datasets import load_all_datasubsets
from minerva_ann_datasets import get_dataset_sizes
from minerva_ann_datasets import slices_maker
from minerva_ann_datasets import make_scheme_and_stream


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


def categorical_learn_and_val_memdt(build_cnn=None, num_epochs=500,
                                    learning_rate=0.01, momentum=0.9,
                                    l2_penalty_scale=1e-04, batchsize=500,
                                    data_file=None,
                                    save_model_file='./params_file.npz',
                                    start_with_saved_params=False,
                                    do_validation_pass=True,
                                    convpooldictlist=None,
                                    nhidden=None, dropoutp=None,
                                    debug_print=False):
    """
    Run learning and validation for triamese networks using AdaGrad for
    learning rate evolution, nesterov momentum; read the data files in
    chunks into memory.
    """
    print("Loading data...")
    train_size, valid_size, _ = get_dataset_sizes(data_file)
    print(" Learning sample size = {} examples".format(train_size))
    print(" Validation sample size = {} examples".format(valid_size))

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the model
    network = build_cnn(input_var_x, input_var_u, input_var_v,
                        convpooldictlist=convpooldictlist, nhidden=nhidden,
                        dropoutp=dropoutp)
    print(network_repr.get_network_str(
        lasagne.layers.get_all_layers(network),
        get_network=False, incomings=True, outgoings=True))
    if start_with_saved_params and os.path.isfile(save_model_file):
        print(" Loading parameters file:", save_model_file)
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
    #
    # TODO: early stopping logic goes here...
    #
    train_slices = slices_maker(train_size, slice_size=50000)
    valid_slices = slices_maker(valid_size, slice_size=50000)

    epoch = 0
    for epoch in range(num_epochs):

        start_time = time.time()
        # In each epoch, we do a full pass over the training data:
        for tslice in train_slices:

            t0 = time.time()
            train_set = load_datasubset(data_file, 'train', tslice)
            _, train_dstream = make_scheme_and_stream(train_set, batchsize)
            t1 = time.time()
            print("  Loading slice {} took {:.3f}s.".format(
                tslice, t1 - t0))
            if debug_print:
                print("   dset sources:", train_set.provides_sources)

            t0 = time.time()
            train_err = 0
            train_batches = 0
            for data in train_dstream.get_epoch_iterator():
                _, inputs, _, targets, _ = \
                    data[0], data[1], data[2], data[3], data[4]
                inputx, inputu, inputv = split_inputs_xuv(inputs)
                train_err += train_fn(inputx, inputu, inputv, targets)
                train_batches += 1
            t1 = time.time()
            print("  -Iterating over the slice took {:.3f}s.".format(t1 - t0))

            del train_set       # hint to garbage collector
            del train_dstream   # hint to garbage collector

        # And a full pass over the validation data
        if do_validation_pass:
            t0 = time.time()
            for vslice in valid_slices:
                valid_set = load_datasubset(data_file, 'valid', vslice)
                _, valid_dstream = make_scheme_and_stream(valid_set, batchsize)

                val_err = 0
                val_acc = 0
                val_batches = 0
                for data in valid_dstream.get_epoch_iterator():
                    _, inputs, _, targets, _ = \
                        data[0], data[1], data[2], data[3], data[4]
                    inputx, inputu, inputv = split_inputs_xuv(inputs)
                    err, acc = val_fn(inputx, inputu, inputv, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1

                del valid_set
                del valid_dstream

            t1 = time.time()
            print("  The validation pass took {:.3f}s.".format(t1 - t0))

        # Dump the current network weights to file
        np.savez(save_model_file,
                 *lasagne.layers.get_all_param_values(network))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if do_validation_pass:
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            print("---")

    print("Finished {} epochs.".format(epoch + 1))


def categorical_test_memdt(build_cnn=None, data_file=None,
                           l2_penalty_scale=1e-04,
                           save_model_file='./params_file.npz', batchsize=500,
                           be_verbose=False, convpooldictlist=None,
                           nhidden=None, dropoutp=None, write_db=True,
                           test_all_data=False, debug_print=False):
    """
    Run tests on the reserved test sample ("trainiing" examples with true
    values to check that were not used for learning or validation); read the
    data files in chunks into memory.
    """
    metadata = None
    try:
        import predictiondb
        from sqlalchemy import MetaData
    except ImportError:
        print("Cannot import sqlalchemy...")
        write_db = False
    print("Loading data for testing...")
    train_size, valid_size, test_size = get_dataset_sizes(data_file)
    used_data_size = test_size
    if test_all_data:
        used_data_size = train_size + valid_size + test_size
    print(" Testing sample size = {} examples".format(used_data_size))

    # extract timestamp from model file - assume it is the first set of numbers
    # otherwise just use "now"
    import re
    tstamp = str(time.time()).split('.')[0]
    m = re.search(r"[0-9]+", save_model_file)
    if m:
        tstamp = m.group(0)
    if write_db:
        metadata = MetaData()
        dbname = 'prediction' + tstamp
        eng = predictiondb.get_engine(dbname)
        con = predictiondb.get_connection(eng)
        tbl = predictiondb.get_active_table(metadata, eng)

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the model
    network = build_cnn(input_var_x, input_var_u, input_var_v,
                        convpooldictlist=convpooldictlist, nhidden=nhidden,
                        dropoutp=dropoutp)
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
    probs_fn = theano.function([input_var_x, input_var_u, input_var_v],
                               [test_prediction],
                               allow_input_downcast=True)

    print("Starting testing...")
    test_slices = slices_maker(used_data_size, slice_size=50000)
    # compute and print the test error and...
    test_err = 0
    test_acc = 0
    test_batches = 0
    # look at some concrete predictions
    targ_numbers = [1, 2, 3, 4, 5]
    pred_target = np.array([0, 0, 0, 0, 0])
    true_target = np.array([0, 0, 0, 0, 0])
    targs_mat = np.zeros(11 * 11).reshape(11, 11)

    for tslice in test_slices:
        t0 = time.time()
        test_set = None
        if test_all_data:
            test_set = load_all_datasubsets(data_file, tslice)
        else:
            test_set = load_datasubset(data_file, 'test', tslice)
        _, test_dstream = make_scheme_and_stream(test_set, 1,
                                                 shuffle=False)
        t1 = time.time()
        print("  Loading slice {} took {:.3f}s.".format(
            tslice, t1 - t0))
        if debug_print:
            print("   dset sources:", test_set.provides_sources)

        t0 = time.time()
        for data in test_dstream.get_epoch_iterator():
            eventids, inputs, _, targets, _ = \
                data[0], data[1], data[2], data[3], data[4]
            inputx, inputu, inputv = split_inputs_xuv(inputs)
            err, acc = val_fn(inputx, inputu, inputv, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
            pred = pred_fn(inputx, inputu, inputv)
            pred_targ = zip(pred[0], targets)
            probs = probs_fn(inputx, inputu, inputv)
            if write_db:
                filldb(tbl, con, eventids, pred, probs)
            if be_verbose:
                print("(prediction, true target):", pred_targ, probs)
                print("----------------")
            for p, t in pred_targ:
                targs_mat[t][p] += 1
                if t in targ_numbers:
                    true_target[t-1] += 1
                    if p == t:
                        pred_target[p-1] += 1
        t1 = time.time()
        print("  -Iterating over the slice took {:.3f}s.".format(t1 - t0))

        del test_set
        del test_dstream

    acc_target = 100.0 * pred_target / true_target.astype('float32')
    perf_file = 'perfmat' + tstamp + '.npy'
    np.save(perf_file, targs_mat)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    for i, v in enumerate(acc_target):
        print("   target {} accuracy:\t\t\t{:.3f} %".format(
            (i + 1), acc_target[i]))


def decode_eventid(eventid):
    """
    assume encoding from fuel_up_nukecc.py, etc.
    """
    eventid = str(eventid)
    phys_evt = eventid[-2:]
    eventid = eventid[:-2]
    gate = eventid[-4:]
    eventid = eventid[:-4]
    subrun = eventid[-4:]
    eventid = eventid[:-4]
    run = eventid
    return (run, subrun, gate, phys_evt)


def filldb(dbtable, dbconnection,
           eventid, pred, probs, db='sqlite-zsegment_prediction'):
    """
    expect pred to have shape (batch, prediction) and probs to have
    shape (batch?, ?, probability)
    """
    result = None
    if db == 'sqlite-zsegment_prediction':
        run, sub, gate, pevt = decode_eventid(eventid[0])
        ins = dbtable.insert().values(
            run=run,
            subrun=sub,
            gate=gate,
            phys_evt=pevt,
            segment=pred[0][0],
            prob00=probs[0][0][0],
            prob01=probs[0][0][1],
            prob02=probs[0][0][2],
            prob03=probs[0][0][3],
            prob04=probs[0][0][4],
            prob05=probs[0][0][5],
            prob06=probs[0][0][6],
            prob07=probs[0][0][7],
            prob08=probs[0][0][8],
            prob09=probs[0][0][9],
            prob10=probs[0][0][10])
        try:
            result = dbconnection.execute(ins)
        except:
            import sys
            e = sys.exc_info()[0]
            print('db error: {}'.format(e))
        return result
    print('unknown database interface!')
    return None
