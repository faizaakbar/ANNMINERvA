def view_layer_activations(
        build_cnn_fn=None, data_file_list=None,
        imgw=50, imgh=50, views='xuv', target_idx=5,
        save_model_file='./params_file.npz',
        be_verbose=False, convpooldictlist=None,
        nhidden=None, dropoutp=None, write_db=True,
        test_all_data=False, debug_print=False,
        get_eventids_hits_and_targets_fn
):
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
    network = build_cnn_fn(inputlist=inputlist, imgw=imgw, imgh=imgh,
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
                    get_eventids_hits_and_targets_fn(data)
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
