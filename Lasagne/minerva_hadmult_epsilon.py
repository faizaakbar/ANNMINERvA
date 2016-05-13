#!/usr/bin/env python
"""
This is an attempt at a "triamese" network operating on Minerva X, U, V.

Execution:
    python minerva_hadmult_epsilon.py -h / --help

At a minimum, we must supply either the `--learn` or `--test` flag.
"""
from __future__ import print_function

import os

from minerva_ann_networks import build_triamese_epsilon
from minerva_ann_operate_networks import categorical_learn_and_validate
from minerva_ann_operate_networks import categorical_test
from minerva_ann_operate_networks import categorical_predict


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
    parser.add_option('-r', '--rate', dest='lrate', default=0.001,
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
                      help='Run a test', metavar='DO_TEST',
                      action='store_true')
    parser.add_option('-p', '--predict', dest='do_predict', default=False,
                      help='Run a prediction', metavar='DO_PREDICT',
                      action='store_true')
    parser.add_option('-v', '--verbose', dest='be_verbose', default=False,
                      help='Verbose predictions', metavar='BE_VERBOSE',
                      action='store_true')
    parser.add_option('-s', '--save_file', dest='save_model_file',
                      default='./lminervatriamese_epsilon.npz',
                      help='File name for parameters',
                      metavar='SAVE_FILE_NAME')
    parser.add_option('-o', '--load_params', dest='start_with_saved_params',
                      default=False, help='Begin training with saved pars',
                      metavar='LOAD_PARAMS', action='store_true')
    parser.add_option('-a', '--test_all', dest='test_all_data',
                      default=False, help='Treat all data as test data',
                      metavar='ALL_TEST', action='store_true')
    parser.add_option('--imgw', dest='imgw', default=127,
                      help='Image width (x/u/v)', metavar='IMGW', type='int')
    parser.add_option('--imgh_x', dest='imgh_x', default=50,
                      help='Image height (z) x', metavar='IMGH_X', type='int')
    parser.add_option('--imgh_u', dest='imgh_u', default=25,
                      help='Image height (z) u', metavar='IMGH_U', type='int')
    parser.add_option('--imgh_v', dest='imgh_v', default=25,
                      help='Image height (z) v', metavar='IMGH_V', type='int')
    (options, args) = parser.parse_args()

    if not options.do_learn and \
            not options.do_test and \
            not options.do_predict:
        print("\nSpecify learn (-l), test (-t), and/or predict (-p):\n\n")
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

    build_network_function = build_triamese_epsilon
    learn = categorical_learn_and_validate
    test = categorical_test
    predict = categorical_predict

    convpooldictlist = {}

    # x
    convpooldictlist_x = []
    # assume 127x72 images
    x_convpool1dict = {}
    x_convpool1dict['nfilters'] = 12
    x_convpool1dict['filter_size'] = (8, 5)
    x_convpool1dict['pool_size'] = (2, 2)
    convpooldictlist_x.append(x_convpool1dict)
    # after 8x5 filters -> 120x68 image, then maxpool -> 60x34
    x_convpool2dict = {}
    x_convpool2dict['nfilters'] = 20
    x_convpool2dict['filter_size'] = (7, 3)
    x_convpool2dict['pool_size'] = (2, 2)
    convpooldictlist_x.append(x_convpool2dict)
    # after 7x3 filters -> 54x32 image, then maxpool -> 27x16
    x_convpool3dict = {}
    x_convpool3dict['nfilters'] = 28
    x_convpool3dict['filter_size'] = (6, 3)
    x_convpool3dict['pool_size'] = (2, 2)
    convpooldictlist_x.append(x_convpool3dict)
    # after 6x3 filters -> 22x14 image, then maxpool -> 11x7
    x_convpool4dict = {}
    x_convpool4dict['nfilters'] = 36
    x_convpool4dict['filter_size'] = (3, 3)
    x_convpool4dict['pool_size'] = (1, 1)
    convpooldictlist_x.append(x_convpool4dict)
    # after 6x3 filters -> 9x5 image, then maxpool -> 9x5

    # u
    convpooldictlist_u = []
    # assume 127x36 images
    u_convpool1dict = {}
    u_convpool1dict['nfilters'] = 12
    u_convpool1dict['filter_size'] = (8, 5)
    u_convpool1dict['pool_size'] = (2, 2)
    convpooldictlist_u.append(u_convpool1dict)
    # after 8x3 filters -> 120x32 image, then maxpool -> 60x16
    u_convpool2dict = {}
    u_convpool2dict['nfilters'] = 20
    u_convpool2dict['filter_size'] = (7, 3)
    u_convpool2dict['pool_size'] = (2, 2)
    convpooldictlist_u.append(u_convpool2dict)
    # after 7x3 filters -> 54x14 image, then maxpool -> 27x7
    u_convpool3dict = {}
    u_convpool3dict['nfilters'] = 28
    u_convpool3dict['filter_size'] = (6, 3)
    u_convpool3dict['pool_size'] = (2, 1)
    convpooldictlist_u.append(u_convpool3dict)
    # after 6x3 filters -> 22x5 image, then maxpool -> 11x5
    u_convpool4dict = {}
    u_convpool4dict['nfilters'] = 36
    u_convpool4dict['filter_size'] = (3, 3)
    u_convpool4dict['pool_size'] = (1, 1)
    convpooldictlist_u.append(u_convpool4dict)
    # after 6x3 filters -> 9x3 image, then maxpool -> 9x3

    # v
    convpooldictlist_v = []
    # assume 127x36 images
    v_convpool1dict = {}
    v_convpool1dict['nfilters'] = 12
    v_convpool1dict['filter_size'] = (8, 5)
    v_convpool1dict['pool_size'] = (2, 2)
    convpooldictlist_v.append(v_convpool1dict)
    # after 8x3 filters -> 120x32 image, then maxpool -> 60x16
    v_convpool2dict = {}
    v_convpool2dict['nfilters'] = 20
    v_convpool2dict['filter_size'] = (7, 3)
    v_convpool2dict['pool_size'] = (2, 2)
    convpooldictlist_v.append(v_convpool2dict)
    # after 7x3 filters -> 54x14 image, then maxpool -> 27x7
    v_convpool3dict = {}
    v_convpool3dict['nfilters'] = 28
    v_convpool3dict['filter_size'] = (6, 3)
    v_convpool3dict['pool_size'] = (2, 1)
    convpooldictlist_v.append(v_convpool3dict)
    # after 6x3 filters -> 22x5 image, then maxpool -> 11x5
    v_convpool4dict = {}
    v_convpool4dict['nfilters'] = 36
    v_convpool4dict['filter_size'] = (3, 3)
    v_convpool4dict['pool_size'] = (1, 1)
    convpooldictlist_v.append(v_convpool4dict)
    # after 6x3 filters -> 9x3 image, then maxpool -> 9x3

    convpooldictlist['x'] = convpooldictlist_x
    convpooldictlist['u'] = convpooldictlist_u
    convpooldictlist['v'] = convpooldictlist_v

    nhidden = 196
    imgw = options.imgw
    imgh = (options.imgh_x,
            options.imgh_u,
            options.imgh_v)

    if options.do_learn:
        learn(build_cnn=build_network_function,
              imgw=imgw,
              imgh=imgh,
              target_idx=4,
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
             target_idx=4,
             data_file_list=options.dataset,
             l2_penalty_scale=options.l2_penalty_scale,
             save_model_file=options.save_model_file,
             be_verbose=options.be_verbose,
             convpooldictlist=convpooldictlist,
             test_all_data=options.test_all_data,
             nhidden=nhidden,
             noutputs=6)

    if options.do_predict:
        predict(build_cnn=build_network_function,
                imgw=imgw,
                imgh=imgh,
                target_idx=4,
                data_file_list=options.dataset,
                save_model_file=options.save_model_file,
                be_verbose=options.be_verbose,
                convpooldictlist=convpooldictlist,
                test_all_data=options.test_all_data,
                nhidden=nhidden,
                noutputs=6)
