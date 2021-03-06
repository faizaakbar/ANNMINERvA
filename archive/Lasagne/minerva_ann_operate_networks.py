#!/usr/bin/env python
import os
import time
import logging
from random import shuffle

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

logger = logging.getLogger(__name__)


def get_used_data_sizes_for_testing(train_sizes, valid_sizes, test_sizes,
                                    test_all_data):
    used_data_size = np.sum(test_sizes)
    used_sizes = test_sizes
    if test_all_data:
        used_data_size += np.sum(train_sizes) + np.sum(valid_sizes)
        used_sizes = \
            list(np.sum([train_sizes, valid_sizes, test_sizes], axis=0))
    logger.info(" Used testing sample size = {} examples".format(used_data_size))
    return used_sizes, used_data_size


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
    logger.info("Loading data...")
    train_sizes, valid_sizes, _ = \
        get_and_print_dataset_subsizes(runopts['data_file_list'])

    # Prepare Theano variables for inputs and targets
    target_var = T.ivector('targets')
    inputlist = networkstr['input_list']

    # Build the model
    network = build_cnn_fn(inputlist=inputlist,
                           imgw=imgdat['imgw'], imgh=imgdat['imgh'],
                           convpooldictlist=networkstr['topology'],
                           nhidden=networkstr['nhidden'],
                           dropoutp=networkstr['dropoutp'],
                           noutputs=networkstr['noutputs'],
                           depth=networkstr['img_depth']
    )
    logger.info(network_repr.get_network_str(
        lasagne.layers.get_all_layers(network),
        get_network=False, incomings=True, outgoings=True))
    if runopts['start_with_saved_params'] and \
       os.path.isfile(runopts['save_model_file']):
        logger.info(" Loading parameters file: %s" % \
                    runopts['save_model_file'])
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
    logger.info(
        """
        ////
        Use AdaGrad update schedule for learning rate, see Duchi, Hazan, and
        Singer (2011) "Adaptive subgradient methods for online learning and
        stochasitic optimization." JMLR, 12:2121-2159
        ////
        """)
    updates_adagrad = lasagne.updates.adagrad(
        loss, params, learning_rate=hyperpars['learning_rate'], epsilon=1e-06)
    logger.info(
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

    logger.info("Starting training...")
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
        for slicelist in train_slices:
            shuffle(slicelist)
        logger.info("Train slices for epoch %d: %s" % (epoch, train_slices))

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
                logger.info(
                    "  Loading slice {} from {} took {:.3f}s.".format(
                        tslice, data_file, t1 - t0)
                )
                logger.debug(
                    "   dset sources: {}".format(train_set.provides_sources)
                )

                t0 = time.time()
                for data in train_dstream.get_epoch_iterator():
                    inputs = get_list_of_hits_and_targets_fn(data)
                    train_err += train_fn(*inputs)
                    train_batches += 1
                t1 = time.time()
                logger.info(
                    "  -Iterating over the slice took {:.3f}s.".format(t1 - t0)
                )

                del train_set       # hint to garbage collector
                del train_dstream   # hint to garbage collector

                # Dump the current network weights to file at end of slice
                np.savez(runopts['save_model_file'],
                         *lasagne.layers.get_all_param_values(network))

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
                        inputs = get_list_of_hits_and_targets_fn(data)
                        err, acc = val_fn(*inputs)
                        val_err += err
                        val_acc += acc
                        val_batches += 1

                    del valid_set
                    del valid_dstream

            t1 = time.time()
            logger.info("  The validation pass took {:.3f}s.".format(t1 - t0))

        # Print the results for this epoch:
        logger.info(
            "\nEpoch {} of {} took {:.3f}s"
            "\n  training loss:\t\t{:.6f}".format(
                epoch + 1, hyperpars['num_epochs'], time.time() - start_time,
                train_err / train_batches
            )
        )
        if runopts['do_validation_pass']:
            logger.info(
                "\n  validation loss:\t\t{:.6f}"
                "\n  validation accuracy:\t\t{:.2f} %".format(
                    val_err / val_batches,
                    val_acc / val_batches * 100
                )
            )
            logger.info("---")

    logger.info("Finished {} epochs.".format(epoch + 1))


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
    logger.info("Loading data for testing...")
    tstamp = get_tstamp_from_model_name(runopts['save_model_file'])
    train_sizes, valid_sizes, test_sizes = \
        get_and_print_dataset_subsizes(runopts['data_file_list'])
    used_sizes, used_data_size = get_used_data_sizes_for_testing(
        train_sizes, valid_sizes, test_sizes, runopts['test_all_data']
    )

    # Prepare Theano variables for inputs and targets
    inputlist = networkstr['input_list']
    target_var = T.ivector('targets')

    # Build the model
    network = build_cnn_fn(inputlist=inputlist,
                           imgw=imgdat['imgw'], imgh=imgdat['imgh'],
                           convpooldictlist=networkstr['topology'],
                           nhidden=networkstr['nhidden'],
                           dropoutp=networkstr['dropoutp'],
                           noutputs=networkstr['noutputs'],
                           depth=networkstr['img_depth']
    )
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

    logger.info("Starting testing...")
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
            logger.info("  Loading slice {} from {} took {:.3f}s.".format(
                tslice, data_file, t1 - t0)
            )
            logger.debug(
                "   dset sources: {}".format(test_set.provides_sources)
            )

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
                        logger.info("{}/{} - {}: (prediction, true target): {}, {}".
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
            logger.info("  -Iterating over the slice took {:.3f}s.".format(t1 - t0))

            del test_set
            del test_dstream

    acc_target = 100.0 * pred_target / true_target.astype('float32')
    perf_file = 'perfmat' + tstamp + '.npy'
    np.save(perf_file, targs_mat)
    logger.info(
        "\nFinal results:"
        "\n  test loss:\t\t\t{:.6f}"
        "\n  test accuracy:\t\t{:.2f} %".format(
            test_err / test_batches, test_acc / test_batches * 100)
    )
    for i, v in enumerate(acc_target):
        logger.info("   target {} accuracy:\t\t\t{:.3f} %".format(
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
    logger.info("Loading data for prediction...")
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
        logger.info("Cannot import sqlalchemy...")
        write_db = False
    if runopts['write_db']:
        db_tbl_fun = None
        if networkstr['noutputs'] == 67:
            db_tbl_fun = predictiondb.get_67segment_prediction_table
        elif networkstr['noutputs'] == 11:
            db_tbl_fun = predictiondb.get_11segment_prediction_table
        else:
            raise Exception('Invalid number of outputs for DB tables.')
        tstamp = get_tstamp_from_model_name(runopts['save_model_file'])
        metadata = MetaData()
        dbname = 'prediction' + tstamp
        eng = predictiondb.get_engine(dbname)
        con = predictiondb.get_connection(eng)
        tbl = predictiondb.get_active_table(metadata,
                                            eng,
                                            get_table_fn=db_tbl_fun
        )

    # Prepare Theano variables for inputs
    inputlist = networkstr['input_list']

    # Build the model
    network = build_cnn_fn(inputlist=inputlist,
                           imgw=imgdat['imgw'], imgh=imgdat['imgh'],
                           convpooldictlist=networkstr['topology'],
                           nhidden=networkstr['nhidden'],
                           dropoutp=networkstr['dropoutp'],
                           noutputs=networkstr['noutputs'],
                           depth=networkstr['img_depth']
    )
    with np.load(runopts['save_model_file']) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Compute the prediction
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction_values = T.argmax(test_prediction, axis=1)
    pred_fn = theano.function(inputlist,
                              [test_prediction, test_prediction_values],
                              allow_input_downcast=True)

    logger.info("Starting prediction...")

    test_slices = []
    for tsize in used_sizes:
        test_slices.append(slices_maker(tsize, slice_size=50000))
    test_set = None

    evtcounter = 0
    verbose_evt_print_freq = hyperpars['batchsize'] * 4
    for i, data_file in enumerate(runopts['data_file_list']):

        for tslice in test_slices[i]:
            t0 = time.time()
            test_set = None
            if runopts['test_all_data']:
                test_set = load_all_datasubsets(data_file, tslice)
            else:
                test_set = load_datasubset(data_file, 'test', tslice)
            _, test_dstream = make_scheme_and_stream(test_set,
                                                     hyperpars['batchsize'],
                                                     shuffle=False)
            t1 = time.time()
            logger.info("  Loading slice {} from {} took {:.3f}s.".format(
                tslice, data_file, t1 - t0)
            )
            logger.debug(
                "   dset sources: {}".format(test_set.provides_sources)
            )

            t0 = time.time()
            for data in test_dstream.get_epoch_iterator():
                eventids, hits_list = get_id_tagged_inputlist_fn(data)
                probs, pred = pred_fn(*hits_list)
                evtcounter += hyperpars['batchsize']
                if runopts['write_db']:
                    for i, evtid in enumerate(eventids):
                        filldb(tbl, con, evtid, pred[i], probs[i])
                if runopts['be_verbose']:
                    if evtcounter % verbose_evt_print_freq == 0:
                        logger.info("processed {}/{}". format(evtcounter,
                                                        used_data_size))
            t1 = time.time()
            logger.info("  -Iterating over the slice took {:.3f}s.".format(t1 - t0))

            del test_set
            del test_dstream

    logger.info("Finished producing predictions!")


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
        if len(probs) == 11:
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
                prob10=probs[10]
            )
        elif len(probs) == 67:
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
                prob10=probs[10],
                prob11=probs[11],
                prob12=probs[12],
                prob13=probs[13],
                prob14=probs[14],
                prob15=probs[15],
                prob16=probs[16],
                prob17=probs[17],
                prob18=probs[18],
                prob19=probs[19],
                prob20=probs[20],
                prob21=probs[21],
                prob22=probs[22],
                prob23=probs[23],
                prob24=probs[24],
                prob25=probs[25],
                prob26=probs[26],
                prob27=probs[27],
                prob28=probs[28],
                prob29=probs[29],
                prob30=probs[30],
                prob31=probs[31],
                prob32=probs[32],
                prob33=probs[33],
                prob34=probs[34],
                prob35=probs[35],
                prob36=probs[36],
                prob37=probs[37],
                prob38=probs[38],
                prob39=probs[39],
                prob40=probs[40],
                prob41=probs[41],
                prob42=probs[42],
                prob43=probs[43],
                prob44=probs[44],
                prob45=probs[45],
                prob46=probs[46],
                prob47=probs[47],
                prob48=probs[48],
                prob49=probs[49],
                prob50=probs[50],
                prob51=probs[51],
                prob52=probs[52],
                prob53=probs[53],
                prob54=probs[54],
                prob55=probs[55],
                prob56=probs[56],
                prob57=probs[57],
                prob58=probs[58],
                prob59=probs[59],
                prob60=probs[60],
                prob61=probs[61],
                prob62=probs[62],
                prob63=probs[63],
                prob64=probs[64],
                prob65=probs[65],
                prob66=probs[66]
            )
        else:
            raise Exception('Impossible number of outputs for db in filldb!')
        try:
            result = dbconnection.execute(ins)
        except:
            import sys
            e = sys.exc_info()[0]
            logger.info('db error: {}'.format(e))
        return result
    logger.info('unknown database interface!')
    return None
