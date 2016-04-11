#!/usr/bin/env python
"""
This is an attempt at a single-view network operating on Minerva X.

Execution:
    python minerva_beta_v.py -h / --help

At a minimum, we must supply either the `--learn` or `--test` flag.
"""
from __future__ import print_function

import os

from minerva_ann_networks import build_beta_v
from minerva_ann_operate_networks import categorical_learn_and_validate
from minerva_ann_operate_networks import categorical_test


def arg_list_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data_list', dest='dataset',
                      help='Data set list (csv)', metavar='DATASETLIST',
                      type='string', action='callback',
                      callback=arg_list_split)
    parser.add_option('-n', '--nepochs', dest='n_epochs', default=200,
                      help='Number of epochs', metavar='N_EPOCHS',
                      type='int')
    parser.add_option('-b', '--batch_size', dest='batchsize', default=500,
                      help='Batch size for SGD', metavar='BATCH_SIZE',
                      type='int')
    parser.add_option('-r', '--rate', dest='lrate', default=0.0025,
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
                      default='./lminerva_betav.npz',
                      help='File name for parameters',
                      metavar='SAVE_FILE_NAME')
    parser.add_option('-p', '--load_params', dest='start_with_saved_params',
                      default=False, help='Begin training with saved pars',
                      metavar='LOAD_PARAMS', action='store_true')
    parser.add_option('-a', '--test_all', dest='test_all_data',
                      default=False, help='Treat all data as test data',
                      metavar='ALL_TEST', action='store_true')
    parser.add_option('--imgw', dest='imgw', default=127,
                      help='Image width (x/u/v)', metavar='IMGW', type='int')
    parser.add_option('--imgh', dest='imgh', default=50,
                      help='Image height (z)', metavar='IMGH', type='int')
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
    print(" Datasets:", options.dataset)
    dataset_statsinfo = 0
    for d in options.dataset:
        dataset_statsinfo += os.stat(d).st_size
    print(" Dataset size:", dataset_statsinfo)
    print(" Planned number of epochs:", options.n_epochs)
    print(" Learning rate:", options.lrate)
    print(" Momentum:", options.momentum)
    print(" L2 regularization penalty scale:", options.l2_penalty_scale)
    print(" Batch size:", options.batchsize)

    build_network_function = build_beta_v
    learn = categorical_learn_and_validate
    test = categorical_test

    convpooldictlist = []
    # assume 127x(N) images
    convpool1dict = {}
    convpool1dict['nfilters'] = 12
    convpool1dict['filter_size'] = (8, 5)
    convpool1dict['pool_size'] = (2, 2)
    convpooldictlist.append(convpool1dict)
    # after 8x5 filters -> 120x(N-4) image, then maxpool -> 60x(N-4)/2
    convpool2dict = {}
    convpool2dict['nfilters'] = 20
    convpool2dict['filter_size'] = (7, 3)
    convpool2dict['pool_size'] = (2, 1)
    convpooldictlist.append(convpool2dict)
    # after 7x3 filters -> 54x(((N-4)/2)-2) image, then maxpool -> 27x(((N-4)/2)-2)
    convpool3dict = {}
    convpool3dict['nfilters'] = 28
    convpool3dict['filter_size'] = (6, 3)
    convpool3dict['pool_size'] = (2, 1)
    convpooldictlist.append(convpool3dict)
    # after 6x3 filters -> 22x(((N-4)/2)-4) image, then maxpool -> 11x(((N-4)/2)-4)
    convpool4dict = {}
    convpool4dict['nfilters'] = 36
    convpool4dict['filter_size'] = (6, 3)
    convpool4dict['pool_size'] = (2, 1)
    convpooldictlist.append(convpool4dict)
    # after 6x3 filters -> 6x(((N-4)/2)-6) image, then maxpool -> 3x(((N-4)/2)-6)

    nhidden = 196
    imgw = options.imgw
    imgh = options.imgh

    if options.do_learn:
        learn(build_cnn=build_network_function,
              imgw=imgw,
              imgh=imgh,
              views='v',
              target_idx=3,
              num_epochs=options.n_epochs,
              learning_rate=options.lrate,
              momentum=options.momentum,
              l2_penalty_scale=options.l2_penalty_scale,
              batchsize=options.batchsize,
              data_file_list=options.dataset,
              save_model_file=options.save_model_file,
              start_with_saved_params=options.start_with_saved_params,
              convpooldictlist=convpooldictlist,
              nhidden=nhidden)

    if options.do_test:
        test(build_cnn=build_network_function,
             imgw=imgw,
             imgh=imgh,
             views='v',
             target_idx=3,
             data_file_list=options.dataset,
             l2_penalty_scale=options.l2_penalty_scale,
             save_model_file=options.save_model_file,
             be_verbose=options.be_verbose,
             convpooldictlist=convpooldictlist,
             test_all_data=options.test_all_data,
             nhidden=nhidden)
