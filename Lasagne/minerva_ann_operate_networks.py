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
from minerva_ann_datasets import get_and_print_dataset_subsizes
from minerva_ann_datasets import slices_maker
from minerva_ann_datasets import make_scheme_and_stream


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


def categorical_learn_and_validate(
        build_cnn_fn, hyperpars, imgdat, runopts, networkstr,
        get_list_of_hits_and_targets_fn
):
    """
    Run learning and validation for triamese networks using AdaGrad for
    learning rate evolution, nesterov momentum; read the data files in
    chunks into memory.

    `get_hits_and_targets` should extract a list `[inputs, targets]` from
    a data slice where `inputs` could be one item or 3 depending on the views
    studied (so total length is 2 or 4, most likely)
    """
    print("Loading data...")
    train_sizes, valid_sizes, _ = \
        get_and_print_dataset_subsizes(runopts['data_file_list'])

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    inputlist = build_inputlist(
        input_var_x, input_var_u, input_var_v, imgdat['views']
    )

    # Build the model
    network = build_cnn_fn(inputlist=inputlist,
                           imgw=imgdat['imgw'], imgh=imgdat['imgh'],
                           convpooldictlist=networkstr['topology'],
                           nhidden=networkstr['nhidden'],
                           dropoutp=networkstr['dropoutp'],
                           noutputs=networkstr['noutputs'])
    print(network_repr.get_network_str(
        lasagne.layers.get_all_layers(network),
        get_network=False, incomings=True, outgoings=True))
    if runopts['start_with_saved_params'] and \
       os.path.isfile(runopts['save_model_file']):
        print(" Loading parameters file:", runopts['save_model_file'])
        with np.load(runopts['save_model_file']) as f:
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
        lasagne.regularization.l2) * networkstr['l2_penalty_scale']
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
        loss, params, learning_rate=hyperpars['learning_rate'], epsilon=1e-06)
    print(
        """
        ////
        Apply Nesterov momentum using Lisa Lab's modifications.
        ////
        """)
    updates = lasagne.updates.apply_nesterov_momentum(
        updates_adagrad, params, momentum=hyperpars['momentum'])

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
    for epoch in range(hyperpars['num_epochs']):

        start_time = time.time()
        train_err = 0
        train_batches = 0
        for i, data_file in enumerate(runopts['data_file_list']):
            # In each epoch, we do a full pass over the training data:
            for tslice in train_slices[i]:

                t0 = time.time()
                train_set = load_datasubset(data_file, 'train', tslice)
                _, train_dstream = make_scheme_and_stream(
                    train_set, hyperpars['batchsize']
                )
                t1 = time.time()
                print("  Loading slice {} from {} took {:.3f}s.".format(
                    tslice, data_file, t1 - t0))
                if runopts['debug_print']:
                    print("   dset sources:", train_set.provides_sources)

                t0 = time.time()
                for data in train_dstream.get_epoch_iterator():
                    # data order in the hdf5 looks like:
                    #  ids, hits-u, hits-v, hits-x, planes, segments, zs
                    # (Check the file carefully for data names, etc.)
                    inputs = get_list_of_hits_and_targets_fn(data)
                    train_err += train_fn(*inputs)
                    train_batches += 1
                t1 = time.time()
                print("  -Iterating over the slice took {:.3f}s.".format(
                    t1 - t0))

                del train_set       # hint to garbage collector
                del train_dstream   # hint to garbage collector

        if runopts['do_validation_pass']:
            # And a full pass over the validation data
            t0 = time.time()
            val_err = 0
            val_acc = 0
            val_batches = 0
            for i, data_file in enumerate(runopts['data_file_list']):
                for vslice in valid_slices[i]:
                    valid_set = load_datasubset(data_file, 'valid', vslice)
                    _, valid_dstream = make_scheme_and_stream(
                        valid_set, hyperpars['batchsize']
                    )

                    for data in valid_dstream.get_epoch_iterator():
                        # data order in the hdf5 looks like:
                        #  ids, hits-u, hits-v, hits-x, planes, segments, zs
                        # (Check the file carefully for data names, etc.)
                        inputs = get_list_of_hits_and_targets_fn(data)
                        err, acc = val_fn(*inputs)
                        val_err += err
                        val_acc += acc
                        val_batches += 1

                    del valid_set
                    del valid_dstream

            t1 = time.time()
            print("  The validation pass took {:.3f}s.".format(t1 - t0))

        # Dump the current network weights to file at the end of epoch
        np.savez(runopts['save_model_file'],
                 *lasagne.layers.get_all_param_values(network))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, hyperpars['num_epochs'], time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if runopts['do_validation_pass']:
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            print("---")

    print("Finished {} epochs.".format(epoch + 1))


def categorical_test(
        build_cnn_fn, hyperpars, imgdat, runopts, networkstr,
        get_eventids_hits_and_targets_fn, get_list_of_hits_fn
):
    """
    Run tests on the reserved test sample ("trainiing" examples with true
    values to check that were not used for learning or validation); read the
    data files in chunks into memory.

    `get_eventids_hits_and_targets_fn` needs to extract from a data slice
    a tuple of (eventids, [inputs], targets), where `[inputs]` might hold
    a single view or all three, etc.

    `get_list_of_hits_fn` needs to extract from a data slice a list of
    `[inputs]` that might hold a single view or all three, etc.
    """
    print("Loading data for testing...")
    tstamp = get_tstamp_from_model_name(runopts['save_model_file'])
    train_sizes, valid_sizes, test_sizes = \
        get_and_print_dataset_subsizes(runopts['data_file_list'])
    used_sizes, used_data_size = get_used_data_sizes_for_testing(
        train_sizes, valid_sizes, test_sizes, runopts['test_all_data']
    )

    # Prepare Theano variables for inputs and targets
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    target_var = T.ivector('targets')

    inputlist = build_inputlist(
        input_var_x, input_var_u, input_var_v, imgdat['views']
    )

    # Build the model
    network = build_cnn_fn(inputlist=inputlist,
                           imgw=imgdat['imgw'], imgh=imgdat['imgh'],
                           convpooldictlist=networkstr['topology'],
                           nhidden=networkstr['nhidden'],
                           dropoutp=networkstr['dropoutp'],
                           noutputs=networkstr['noutputs'])
    with np.load(runopts['save_model_file']) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for testing.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    l2_penalty = lasagne.regularization.regularize_layer_params(
        lasagne.layers.get_all_layers(network),
        lasagne.regularization.l2) * networkstr['l2_penalty_scale']
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
    num_poss_segs = networkstr['noutputs']
    pred_target = np.zeros(num_poss_segs, dtype='float32')
    true_target = np.zeros(num_poss_segs, dtype='float32')
    targs_mat = np.zeros(num_poss_segs * num_poss_segs,
                         dtype='float32').reshape(num_poss_segs, num_poss_segs)

    test_slices = []
    for tsize in used_sizes:
        test_slices.append(slices_maker(tsize, slice_size=50000))
    test_set = None

    verbose_evt_print_freq = 1
    evtcounter = 0
    for i, data_file in enumerate(runopts['data_file_list']):

        for tslice in test_slices[i]:
            t0 = time.time()
            test_set = None
            if runopts['test_all_data']:
                test_set = load_all_datasubsets(data_file, tslice)
            else:
                test_set = load_datasubset(data_file, 'test', tslice)
            _, test_dstream = make_scheme_and_stream(
                test_set, 1, shuffle=False
            )
            t1 = time.time()
            print("  Loading slice {} from {} took {:.3f}s.".format(
                tslice, data_file, t1 - t0)
            )
            if runopts['debug_print']:
                print("   dset sources:", test_set.provides_sources)

            t0 = time.time()
            for data in test_dstream.get_epoch_iterator():
                eventids, inputlist, targets = \
                    get_eventids_hits_and_targets_fn(data)
                inputlist.append(targets)
                err, acc = val_fn(*inputlist)
                test_err += err
                test_acc += acc
                test_batches += 1
                hits_list = get_list_of_hits_fn(data)
                probs, pred = pred_fn(*hits_list)
                pred_targ = zip(pred, targets)
                evtcounter += 1
                if runopts['be_verbose']:
                    if evtcounter % verbose_evt_print_freq == 0:
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


def categorical_predict(
        build_cnn_fn, hyperpars, imgdat, runopts, networkstr,
        get_eventids_hits_and_targets_fn, get_id_tagged_inputlist_fn
):
    """
    Make predictions based on the model _only_ (e.g., this routine should
    be used to produce prediction db's quickly or for data)

    `get_eventids_hits_and_targets_fn` needs to extract from a data slice
    a tuple of (eventids, [inputs], targets), where `[inputs]` might hold
    a single view or all three, etc.
    """
    print("Loading data for testing...")
    train_sizes, valid_sizes, test_sizes = \
        get_and_print_dataset_subsizes(runopts['data_file_list'])
    used_sizes, used_data_size = get_used_data_sizes_for_testing(
        train_sizes, valid_sizes, test_sizes, runopts['test_all_data']
    )

    metadata = None
    try:
        import predictiondb
        from sqlalchemy import MetaData
    except ImportError:
        print("Cannot import sqlalchemy...")
        write_db = False
    if write_db:
        tstamp = get_tstamp_from_model_name(runopts['save_model_file'])
        metadata = MetaData()
        dbname = 'prediction' + tstamp
        eng = predictiondb.get_engine(dbname)
        con = predictiondb.get_connection(eng)
        tbl = predictiondb.get_active_table(metadata, eng)

    # Prepare Theano variables for inputs
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    inputlist = build_inputlist(
        input_var_x, input_var_u, input_var_v, imgdat['views']
    )

    # Build the model
    network = build_cnn_fn(inputlist=inputlist,
                           imgw=imgdat['imgw'], imgh=imgdat['imgh'],
                           convpooldictlist=networkstr['topology'],
                           nhidden=networkstr['nhidden'],
                           dropoutp=networkstr['dropoutp'],
                           noutputs=networkstr['noutputs'])
    with np.load(runopts['save_model_file']) as f:
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
    verbose_evt_print_freq = batch_size * 4
    for i, data_file in enumerate(runopts['data_file_list']):

        for tslice in test_slices[i]:
            t0 = time.time()
            test_set = None
            if test_all_data:
                test_set = load_all_datasubsets(data_file, tslice)
            else:
                test_set = load_datasubset(data_file, 'test', tslice)
            _, test_dstream = make_scheme_and_stream(test_set,
                                                     hyperpars['batch_size'],
                                                     shuffle=False)
            t1 = time.time()
            print("  Loading slice {} from {} took {:.3f}s.".format(
                tslice, data_file, t1 - t0)
            )
            if debug_print:
                print("   dset sources:", test_set.provides_sources)

            t0 = time.time()
            for data in test_dstream.get_epoch_iterator():
                eventids, hits_list = get_id_tagged_inputlist_fn(data)
                probs, pred = pred_fn(*hits_list)
                evtcounter += batch_size
                if write_db:
                    for i, evtid in enumerate(eventids):
                        filldb(tbl, con, evtid, pred[i], probs[i])
                if be_verbose:
                    if evtcounter % verbose_evt_print_freq == 0:
                        print("processed {}/{}". format(evtcounter,
                                                        used_data_size))
            t1 = time.time()
            print("  -Iterating over the slice took {:.3f}s.".format(t1 - t0))

            del test_set
            del test_dstream

    print("Finished producing predictions!")


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
