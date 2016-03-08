#!/usr/bin/env python
"""
This is an attempt at a "triamese" network operating on Minerva X, U, V.

Execution:
    python minerva_triamese_beta.py -h / --help

At a minimum, we must supply either the `--learn` or `--test` flag.

See ANNMINERvA/fuel_up_convdata.py for an HDF5 builder that sets up an
appropriate data file.

"""
from __future__ import print_function

import os

from minerva_ann_networks import build_triamese_beta
from minerva_ann_operate_networks import categorical_learn_and_val_memdt
from minerva_ann_operate_networks import categorical_test_memdt


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data', dest='dataset',
                      default='./nukecc_fuel.hdf5',
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
    parser.add_option('-l', '--learn', dest='do_learn', default=False,
                      help='Run the training', metavar='DO_LEARN',
                      action='store_true')
    parser.add_option('-t', '--test', dest='do_test', default=False,
                      help='Run a prediction', metavar='DO_TEST',
                      action='store_true')
    parser.add_option('-v', '--verbose', dest='be_verbose', default=False,
                      help='Verbose predictions', metavar='BE_VERBOSE',
                      action='store_true')
    parser.add_option('-s', '--save_file', dest='save_model_file',
                      default='./lminervatriamese_beta.npz',
                      help='File name for parameters',
                      metavar='SAVE_FILE_NAME')
    parser.add_option('-p', '--load_params', dest='start_with_saved_params',
                      default=False, help='Begin training with saved pars',
                      metavar='LOAD_PARAMS', action='store_true')
    parser.add_option('-a', '--test_all', dest='test_all_data',
                      default=False, help='Treat all data as test data',
                      metavar='ALL_TEST', action='store_true')
    (options, args) = parser.parse_args()

    if not options.do_learn and not options.do_test:
        print("\nMust specify at least either learn (-l) or test (-t):\n\n")
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

    build_network_function = build_triamese_beta
    learn = categorical_learn_and_val_memdt
    test = categorical_test_memdt

    # assume 50x50 images
    convpooldictlist = []
    convpool1dict = {}
    convpool1dict['nfilters'] = 12
    convpool1dict['filter_size'] = (5, 5)
    convpool1dict['pool_size'] = (2, 2)
    convpooldictlist.append(convpool1dict)
    # after 5x5 filters -> 46x46 image, then maxpool -> 23x23
    convpool2dict = {}
    convpool2dict['nfilters'] = 8
    convpool2dict['filter_size'] = (4, 4)
    convpool2dict['pool_size'] = (2, 2)
    convpooldictlist.append(convpool2dict)
    # after 4x4 filters -> 20x20 image, then maxpool -> 10x10
    convpool3dict = {}
    convpool3dict['nfilters'] = 8
    convpool3dict['filter_size'] = (3, 3)
    convpool3dict['pool_size'] = (2, 2)
    convpooldictlist.append(convpool3dict)
    # after 3x3 filters -> 8x8 image, then maxpool -> 4x4

    if options.do_learn:
        learn(build_cnn=build_network_function,
              num_epochs=options.n_epochs,
              learning_rate=options.lrate,
              momentum=options.momentum,
              l2_penalty_scale=options.l2_penalty_scale,
              batchsize=options.batchsize,
              data_file=options.dataset,
              save_model_file=options.save_model_file,
              start_with_saved_params=options.start_with_saved_params,
              convpooldictlist=convpooldictlist)

    if options.do_test:
        test(build_cnn=build_network_function,
             data_file=options.dataset,
             l2_penalty_scale=options.l2_penalty_scale,
             save_model_file=options.save_model_file,
             batchsize=options.batchsize,
             be_verbose=options.be_verbose,
             convpooldictlist=convpooldictlist,
             test_all_data=options.test_all_data)
