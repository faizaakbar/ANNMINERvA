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


def get_and_print_dataset_subsizes(data_file_list):
    """
    get a list of the sizes of each 'subset' (train/valid/test) in
    each data file and return a 3-tuple of lists of lists of sizes
    """
    train_sizes = []
    valid_sizes = []
    test_sizes = []
    for data_file in data_file_list:
        lsize, vsize, tsize = get_dataset_sizes(data_file)
        train_sizes.append(lsize)
        valid_sizes.append(vsize)
        test_sizes.append(tsize)
    print(" Learning sample size = {} examples".format(
        np.sum(train_sizes)))
    print(" Validation sample size = {} examples".format(
        np.sum(valid_sizes)))
    print(" Testing sample size = {} examples".format(
        np.sum(test_sizes)))
    return train_sizes, valid_sizes, test_sizes


def get_used_data_sizes_for_testing(train_sizes, valid_sizes, test_sizes,
                                    test_all_data):
    used_data_size = np.sum(test_sizes)
    used_sizes = test_sizes
    if test_all_data:
        used_data_size += np.sum(train_sizes) + np.sum(valid_sizes)
        used_sizes = \
            list(np.sum([train_sizes, valid_sizes, test_sizes], axis=0))
    print(" Used testing sample size = {} examples".format(used_data_size))
    return used_sizes, used_data_size


def build_inputlist(input_var_x, input_var_u, input_var_v, views):
    inputlist = []
    if 'x' in views:
        inputlist.append(input_var_x)
    if 'u' in views:
        inputlist.append(input_var_u)
    if 'v' in views:
        inputlist.append(input_var_v)
    return inputlist


def get_list_of_hits_and_targets_from_data(data, views, target_idx):
    """
    data[0] should be eventids
    """
    inputs = []
    if views == 'xuv':
        inputu, inputv, inputx, targets = \
            data[1], data[2], data[3], data[target_idx]
        inputs = [inputx, inputu, inputv, targets]
    else:
        input_view, targets = \
            data[1], data[target_idx]
        inputs = [input_view, targets]
    return inputs


def get_eventids_hits_and_targets_from_data(data, views, target_idx):
    """
    data[0] should be eventids
    """
    inputs = []
    if views == 'xuv':
        eventids, inputu, inputv, inputx, targets = \
            data[0], data[1], data[2], data[3], data[target_idx]
        inputs = [inputx, inputu, inputv]
    else:
        eventids, input_view, targets = \
            data[0], data[1], data[target_idx]
        inputs = [input_view]
    return eventids, inputs, targets


def get_id_tagged_inputlist_from_data(data, views):
    """
    data[0] should be eventids
    """
    inputs = []
    if views == 'xuv':
        eventids, inputu, inputv, inputx = \
            data[0], data[1], data[2], data[3]
        inputs = [inputx, inputu, inputv]
    else:
        eventids, input_view = \
            data[0], data[1]
        inputs = [input_view]
    return eventids, inputs


def get_list_of_hits_from_data(data, views):
    """
    data[0] should be eventids
    """
    inputs = []
    if views == 'xuv':
        inputu, inputv, inputx = data[1], data[2], data[3]
        inputs = [inputx, inputu, inputv]
    else:
        input_view = data[1]
        inputs = [input_view]
    return inputs


def get_tstamp_from_model_name(save_model_file):
    """
    extract timestamp from model file - assume it is the first set of numbers;
    otherwise just use "now"
    """
    import re
    tstamp = str(time.time()).split('.')[0]
    m = re.search(r"[0-9]+", save_model_file)
    if m:
        tstamp = m.group(0)
    return tstamp


def categorical_learn_and_validate(build_cnn=None, num_epochs=500,
                                   learning_rate=0.01, momentum=0.9,
                                   l2_penalty_scale=1e-04, batchsize=500,
                                   imgw=50, imgh=50, views='xuv',
                                   target_idx=5, noutputs=11,
                                   data_file_list=None,
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
    train_sizes, valid_sizes, _ = \
        get_and_print_dataset_subsizes(data_file_list)

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    inputlist = build_inputlist(input_var_x, input_var_u, input_var_v, views)

    # Build the model
    network = build_cnn(inputlist=inputlist, imgw=imgw, imgh=imgh,
                        convpooldictlist=convpooldictlist, nhidden=nhidden,
                        dropoutp=dropoutp, noutputs=noutputs)
    print(network_repr.get_network_str(
        lasagne.layers.get_all_layers(network),
        get_network=False, incomings=True, outgoings=True))
    if start_with_saved_params and os.path.isfile(save_model_file):
        print(" Loading parameters file:", save_model_file)
        with np.load(save_model_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    else:
        # Dump the current network weights to file in case we want to study
        # intialization trends, etc.
        np.savez('./initial_parameters.npz',
                 *lasagne.layers.get_all_param_values(network))

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
    inputlist.append(target_var)
    train_fn = theano.function(inputlist, loss, updates=updates,
                               allow_input_downcast=True)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(inputlist, [test_loss, test_acc],
                             allow_input_downcast=True)

    print("Starting training...")
    #
    # TODO: early stopping logic goes here...
    #
    train_slices = []
    for tsize in train_sizes:
        train_slices.append(slices_maker(tsize, slice_size=50000))
    valid_slices = []
    for vsize in valid_sizes:
        valid_slices.append(slices_maker(vsize, slice_size=50000))
    train_set = None
    valid_set = None

    epoch = 0
    for epoch in range(num_epochs):

        start_time = time.time()
        train_err = 0
        train_batches = 0
        for i, data_file in enumerate(data_file_list):
            # In each epoch, we do a full pass over the training data:
            for tslice in train_slices[i]:

                t0 = time.time()
                train_set = load_datasubset(data_file, 'train', tslice)
                _, train_dstream = make_scheme_and_stream(train_set, batchsize)
                t1 = time.time()
                print("  Loading slice {} from {} took {:.3f}s.".format(
                    tslice, data_file, t1 - t0))
                if debug_print:
                    print("   dset sources:", train_set.provides_sources)

                t0 = time.time()
                for data in train_dstream.get_epoch_iterator():
                    # data order in the hdf5 looks like:
                    #  ids, hits-u, hits-v, hits-x, planes, segments, zs
                    # (Check the file carefully for data names, etc.)
                    inputs = get_list_of_hits_and_targets_from_data(
                        data, views, target_idx)
                    train_err += train_fn(*inputs)
                    train_batches += 1
                t1 = time.time()
                print("  -Iterating over the slice took {:.3f}s.".format(
                    t1 - t0))

                del train_set       # hint to garbage collector
                del train_dstream   # hint to garbage collector

        if do_validation_pass:
            # And a full pass over the validation data
            t0 = time.time()
            val_err = 0
            val_acc = 0
            val_batches = 0
            for i, data_file in enumerate(data_file_list):
                for vslice in valid_slices[i]:
                    valid_set = load_datasubset(data_file, 'valid', vslice)
                    _, valid_dstream = make_scheme_and_stream(valid_set,
                                                              batchsize)

                    for data in valid_dstream.get_epoch_iterator():
                        # data order in the hdf5 looks like:
                        #  ids, hits-u, hits-v, hits-x, planes, segments, zs
                        # (Check the file carefully for data names, etc.)
                        inputs = get_list_of_hits_and_targets_from_data(
                            data, views, target_idx)
                        err, acc = val_fn(*inputs)
                        val_err += err
                        val_acc += acc
                        val_batches += 1

                    del valid_set
                    del valid_dstream

            t1 = time.time()
            print("  The validation pass took {:.3f}s.".format(t1 - t0))

        # Dump the current network weights to file at the end of epoch
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


def categorical_test(build_cnn=None, data_file_list=None,
                     l2_penalty_scale=1e-04, views='xuv',
                     imgw=50, imgh=50, target_idx=5,
                     save_model_file='./params_file.npz',
                     be_verbose=False, convpooldictlist=None,
                     nhidden=None, dropoutp=None,
                     test_all_data=False, debug_print=False,
                     noutputs=11):
    """
    Run tests on the reserved test sample ("trainiing" examples with true
    values to check that were not used for learning or validation); read the
    data files in chunks into memory.

    noutputs=11 for zsegments, other vals for planecodes, etc.
    """
    print("Loading data for testing...")
    tstamp = get_tstamp_from_model_name(save_model_file)
    train_sizes, valid_sizes, test_sizes = \
        get_and_print_dataset_subsizes(data_file_list)
    used_sizes, used_data_size = get_used_data_sizes_for_testing(train_sizes,
                                                                 valid_sizes,
                                                                 test_sizes,
                                                                 test_all_data)

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    inputlist = build_inputlist(input_var_x, input_var_u, input_var_v, views)

    # Build the model
    network = build_cnn(inputlist=inputlist, imgw=imgw, imgh=imgh,
                        convpooldictlist=convpooldictlist, nhidden=nhidden,
                        dropoutp=dropoutp, noutputs=noutputs)
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

    # Compute the actual predictions - also instructive is to look at
    # `test_prediction` as an output (array of softmax probabilities)
    pred_fn = theano.function(inputlist,
                              [test_prediction, test_prediction_values],
                              allow_input_downcast=True)
    # Compile a function computing the validation loss and accuracy:
    inputlist.append(target_var)
    val_fn = theano.function(inputlist, [test_loss, test_acc],
                             allow_input_downcast=True)

    print("Starting testing...")
    # compute and print the test error and...
    test_err = 0
    test_acc = 0
    test_batches = 0
    # look at some concrete predictions
    num_poss_segs = noutputs
    pred_target = np.zeros(num_poss_segs, dtype='float32')
    true_target = np.zeros(num_poss_segs, dtype='float32')
    targs_mat = np.zeros(num_poss_segs * num_poss_segs,
                         dtype='float32').reshape(num_poss_segs, num_poss_segs)

    test_slices = []
    for tsize in used_sizes:
        test_slices.append(slices_maker(tsize, slice_size=50000))
    test_set = None

    evt_print_freq = 1   # TODO: pass this in?
    evtcounter = 0
    for i, data_file in enumerate(data_file_list):

        for tslice in test_slices[i]:
            t0 = time.time()
            test_set = None
            if test_all_data:
                test_set = load_all_datasubsets(data_file, tslice)
            else:
                test_set = load_datasubset(data_file, 'test', tslice)
            _, test_dstream = make_scheme_and_stream(test_set, 1,
                                                     shuffle=False)
            t1 = time.time()
            print("  Loading slice {} from {} took {:.3f}s.".format(
                tslice, data_file, t1 - t0))
            if debug_print:
                print("   dset sources:", test_set.provides_sources)

            t0 = time.time()
            for data in test_dstream.get_epoch_iterator():
                eventids, inputlist, targets = \
                    get_eventids_hits_and_targets_from_data(
                        data, views, target_idx)
                inputlist.append(targets)
                err, acc = val_fn(*inputlist)
                test_err += err
                test_acc += acc
                test_batches += 1
                hits_list = get_list_of_hits_from_data(data, views)
                probs, pred = pred_fn(*hits_list)
                pred_targ = zip(pred, targets)
                evtcounter += 1
                if be_verbose:
                    if evtcounter % evt_print_freq == 0:
                        print("{}/{} - {}: (prediction, true target): {}, {}".
                              format(evtcounter,
                                     used_data_size,
                                     eventids[0],
                                     pred_targ, probs))
                for p, t in pred_targ:
                    targs_mat[t][p] += 1
                    true_target[t] += 1
                    if p == t:
                        pred_target[p] += 1
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
            i, acc_target[i]))


def categorical_predict(build_cnn=None, data_file_list=None,
                        views='xuv', imgw=50, imgh=50, target_idx=5,
                        save_model_file='./params_file.npz',
                        be_verbose=False, convpooldictlist=None,
                        nhidden=None, dropoutp=None, write_db=True,
                        test_all_data=False, debug_print=False,
                        noutputs=11):
    """
    Make predictions based on the model _only_ (e.g., this routine should
    be used to produce prediction db's quickly or for data)

    noutputs=11 for zsegments, other vals for planecodes, etc.
    """
    print("Loading data for testing...")
    train_sizes, valid_sizes, test_sizes = \
        get_and_print_dataset_subsizes(data_file_list)
    used_sizes, used_data_size = get_used_data_sizes_for_testing(train_sizes,
                                                                 valid_sizes,
                                                                 test_sizes,
                                                                 test_all_data)

    metadata = None
    try:
        import predictiondb
        from sqlalchemy import MetaData
    except ImportError:
        print("Cannot import sqlalchemy...")
        write_db = False
    if write_db:
        tstamp = get_tstamp_from_model_name(save_model_file)
        metadata = MetaData()
        dbname = 'prediction' + tstamp
        eng = predictiondb.get_engine(dbname)
        con = predictiondb.get_connection(eng)
        tbl = predictiondb.get_active_table(metadata, eng)

    # Prepare Theano variables for inputs
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    inputlist = build_inputlist(input_var_x, input_var_u, input_var_v, views)

    # Build the model
    network = build_cnn(inputlist=inputlist, imgw=imgw, imgh=imgh,
                        convpooldictlist=convpooldictlist, nhidden=nhidden,
                        dropoutp=dropoutp, noutputs=noutputs)
    with np.load(save_model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Compute the prediction
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction_values = T.argmax(test_prediction, axis=1)
    pred_fn = theano.function(inputlist,
                              [test_prediction, test_prediction_values],
                              allow_input_downcast=True)

    print("Starting prediction...")

    test_slices = []
    for tsize in used_sizes:
        test_slices.append(slices_maker(tsize, slice_size=50000))
    test_set = None

    evtcounter = 0
    batch_size = 500
    evt_print_freq = batch_size * 4
    for i, data_file in enumerate(data_file_list):

        for tslice in test_slices[i]:
            t0 = time.time()
            test_set = None
            if test_all_data:
                test_set = load_all_datasubsets(data_file, tslice)
            else:
                test_set = load_datasubset(data_file, 'test', tslice)
            _, test_dstream = make_scheme_and_stream(test_set,
                                                     batch_size,
                                                     shuffle=False)
            t1 = time.time()
            print("  Loading slice {} from {} took {:.3f}s.".format(
                tslice, data_file, t1 - t0))
            if debug_print:
                print("   dset sources:", test_set.provides_sources)

            t0 = time.time()
            for data in test_dstream.get_epoch_iterator():
                eventids, hits_list = \
                    get_id_tagged_inputlist_from_data(data, views)
                probs, pred = pred_fn(*hits_list)
                evtcounter += batch_size
                if write_db:
                    for i, evtid in enumerate(eventids):
                        filldb(tbl, con, evtid, pred[i], probs[i])
                if be_verbose:
                    if evtcounter % evt_print_freq == 0:
                        print("processed {}/{}". format(evtcounter,
                                                        used_data_size))
            t1 = time.time()
            print("  -Iterating over the slice took {:.3f}s.".format(t1 - t0))

            del test_set
            del test_dstream

    print("Finished producing predictions!")


def view_layer_activations(build_cnn=None, data_file_list=None,
                           imgw=50, imgh=50, views='xuv', target_idx=5,
                           save_model_file='./params_file.npz',
                           be_verbose=False, convpooldictlist=None,
                           nhidden=None, dropoutp=None, write_db=True,
                           test_all_data=False, debug_print=False):
    """
    Run tests on the reserved test sample ("trainiing" examples with true
    values to check that were not used for learning or validation); read the
    data files in chunks into memory.
    """
    print("Loading data for testing...")
    train_sizes, valid_sizes, test_sizes = \
        get_and_print_dataset_subsizes(data_file_list)
    used_sizes, _ = get_used_data_sizes_for_testing(train_sizes,
                                                    valid_sizes,
                                                    test_sizes,
                                                    test_all_data)

    # extract timestamp from model file - assume it is the first set of numbers
    # otherwise just use "now"
    import re
    tstamp = str(time.time()).split('.')[0]
    m = re.search(r"[0-9]+", save_model_file)
    if m:
        tstamp = m.group(0)

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')

    inputlist = build_inputlist(input_var_x, input_var_u, input_var_v, views)

    # Build the model
    network = build_cnn(inputlist=inputlist, imgw=imgw, imgh=imgh,
                        convpooldictlist=convpooldictlist, nhidden=nhidden,
                        dropoutp=dropoutp, noutputs=noutputs)
    with np.load(save_model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    print(network_repr.get_network_str(
        lasagne.layers.get_all_layers(network),
        get_network=False, incomings=True, outgoings=True))

    layers = lasagne.layers.get_all_layers(network)
    # layer assignment is _highly_ network specific...
    layer_conv_x1 = lasagne.layers.get_output(layers[1])
    layer_conv_u1 = lasagne.layers.get_output(layers[8])
    layer_conv_v1 = lasagne.layers.get_output(layers[15])
    layer_pool_x1 = lasagne.layers.get_output(layers[2])
    layer_pool_u1 = lasagne.layers.get_output(layers[9])
    layer_pool_v1 = lasagne.layers.get_output(layers[16])
    layer_conv_x2 = lasagne.layers.get_output(layers[3])
    layer_conv_u2 = lasagne.layers.get_output(layers[10])
    layer_conv_v2 = lasagne.layers.get_output(layers[17])
    layer_pool_x2 = lasagne.layers.get_output(layers[4])
    layer_pool_u2 = lasagne.layers.get_output(layers[11])
    layer_pool_v2 = lasagne.layers.get_output(layers[18])
    vis_conv_x1 = theano.function(inputlist,
                                  [layer_conv_x1],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_conv_u1 = theano.function(inputlist,
                                  [layer_conv_u1],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_conv_v1 = theano.function(inputlist,
                                  [layer_conv_v1],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_pool_x1 = theano.function(inputlist,
                                  [layer_pool_x1],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_pool_u1 = theano.function(inputlist,
                                  [layer_pool_u1],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_pool_v1 = theano.function(inputlist,
                                  [layer_pool_v1],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_conv_x2 = theano.function(inputlist,
                                  [layer_conv_x2],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_conv_u2 = theano.function(inputlist,
                                  [layer_conv_u2],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_conv_v2 = theano.function(inputlist,
                                  [layer_conv_v2],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_pool_x2 = theano.function(inputlist,
                                  [layer_pool_x2],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_pool_u2 = theano.function(inputlist,
                                  [layer_pool_u2],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')
    vis_pool_v2 = theano.function(inputlist,
                                  [layer_pool_v2],
                                  allow_input_downcast=True,
                                  on_unused_input='warn')

    print("Starting visualization...")
    test_slices = []
    for tsize in used_sizes:
        test_slices.append(slices_maker(tsize, slice_size=50000))
    test_set = None

    for i, data_file in enumerate(data_file_list):

        for tslice in test_slices[i]:
            t0 = time.time()
            test_set = None
            if test_all_data:
                test_set = load_all_datasubsets(data_file, tslice)
            else:
                test_set = load_datasubset(data_file, 'test', tslice)
            _, test_dstream = make_scheme_and_stream(test_set, 1,
                                                     shuffle=False)
            t1 = time.time()
            print("  Loading slice {} from {} took {:.3f}s.".format(
                tslice, data_file, t1 - t0))
            if debug_print:
                print("   dset sources:", test_set.provides_sources)

            t0 = time.time()
            for data in test_dstream.get_epoch_iterator():
                # data order in the hdf5 looks like:
                #  ids, hits-u, hits-v, hits-x, planes, segments, zs
                # (Check the file carefully for data names, etc.)
                eventids, inputlist, targets = \
                    get_eventids_hits_and_targets_from_data(
                        data, views, target_idx)
                conv_x1 = vis_conv_x1(*inputlist)
                conv_u1 = vis_conv_u1(*inputlist)
                conv_v1 = vis_conv_v1(*inputlist)
                pool_x1 = vis_pool_x1(*inputlist)
                pool_u1 = vis_pool_u1(*inputlist)
                pool_v1 = vis_pool_v1(*inputlist)
                conv_x2 = vis_conv_x2(*inputlist)
                conv_u2 = vis_conv_u2(*inputlist)
                conv_v2 = vis_conv_v2(*inputlist)
                pool_x2 = vis_pool_x2(*inputlist)
                pool_u2 = vis_pool_u2(*inputlist)
                pool_v2 = vis_pool_v2(*inputlist)
                vis_file = 'vis_' + str(targets[0]) + '_conv_1_' + tstamp + \
                    '_' + str(eventids[0]) + '.npy'
                np.save(vis_file, [conv_x1, conv_u1, conv_v1])
                vis_file = 'vis_' + str(targets[0]) + '_pool_1_' + tstamp + \
                    '_' + str(eventids[0]) + '.npy'
                np.save(vis_file, [pool_x1, pool_u1, pool_v1])
                vis_file = 'vis_' + str(targets[0]) + '_conv_2_' + tstamp + \
                    '_' + str(eventids[0]) + '.npy'
                np.save(vis_file, [conv_x2, conv_u2, conv_v2])
                vis_file = 'vis_' + str(targets[0]) + '_pool_2_' + tstamp + \
                    '_' + str(eventids[0]) + '.npy'
                np.save(vis_file, [pool_x2, pool_u2, pool_v2])
            t1 = time.time()
            print("  -Iterating over the slice took {:.3f}s.".format(t1 - t0))

            del test_set
            del test_dstream


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
        run, sub, gate, pevt = decode_eventid(eventid)
        ins = dbtable.insert().values(
            run=run,
            subrun=sub,
            gate=gate,
            phys_evt=pevt,
            segment=pred,
            prob00=probs[0],
            prob01=probs[1],
            prob02=probs[2],
            prob03=probs[3],
            prob04=probs[4],
            prob05=probs[5],
            prob06=probs[6],
            prob07=probs[7],
            prob08=probs[8],
            prob09=probs[9],
            prob10=probs[10])
        try:
            result = dbconnection.execute(ins)
        except:
            import sys
            e = sys.exc_info()[0]
            print('db error: {}'.format(e))
        return result
    print('unknown database interface!')
    return None
