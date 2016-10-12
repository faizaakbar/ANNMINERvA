#!/usr/bin/env python
"""
This is a "tricolumnar" network operating on Minerva X, U, V (time). It assumes
an HDF5 input file with groups organized like so:

groups in the hdf5 file
-----------------------
eventids
hits-u
hits-v
hits-x
muon_data
planecodes
segments
times-u
times-v
times-x
zs
None

Execution:
    python minerva_tricolmnar_epsilon.py -h / --help

At a minimum, we must supply either the `--learn`, `--test`, or `--predict`
flag.
"""
from __future__ import print_function

import os
import sys

import theano.tensor as T

from minerva_ann_networks import build_triamese_epsilon
from minerva_ann_operate_networks import categorical_learn_and_validate
from minerva_ann_operate_networks import categorical_test
from minerva_ann_operate_networks import categorical_predict


def arg_list_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

def get_theano_input_tensors():
    """
    Prepare Theano variables for inputs - must remake these for each graph
    """
    input_var_x = T.tensor4('inputs')
    input_var_u = T.tensor4('inputs')
    input_var_v = T.tensor4('inputs')
    return [input_var_x, input_var_u, input_var_v]


def get_hits(data):
    """
    data[7], [8], [9] should be times-u, -v, -x
    
    return a list of [inp-x, inp-u, inp-v]
    """
    inputu, inputv, inputx = data[7], data[8], data[9]
    inputs = [inputx, inputu, inputv]
    return inputs


def get_hits_and_targets(data):
    """
    data[0] should be eventids
    data[7], [8], [9] should be hits-u, -v, -x
    data[6] should be segments (the target)

    return everything in one list [inputs, targets]
    """
    inputu, inputv, inputx = data[7], data[8], data[9]
    inputs = [inputx, inputu, inputv, data[6]]
    return inputs


def get_hits_and_targets_tup(data):
    """
    data[0] should be eventids
    """
    eventids, inputu, inputv, inputx = data[0], data[7], data[8], data[9]
    inputs = [inputx, inputu, inputv]
    return eventids, inputs


def get_eventids_hits_and_targets(data):
    """
    data[0] should be eventids
    data[7], [8], [9] should be times-u, -v, -x
    data[6] should be segments
    
    return a tuple of (eventids, [inputs], targets)
    """
    inputs = [data[9], data[7], data[8]]
    return data[0], inputs, data[6]


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data_list', dest='dataset',
                      help='Data set list (csv)', metavar='DATASETLIST',
                      type='string', action='callback',
                      callback=arg_list_split)
    parser.add_option('-f', '--logfile', dest='logfilename',
                      help='Log file name', metavar='LOGFILENAME',
                      default=None, type='string')
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
    parser.add_option('--noutputs', dest='noutputs', default=11,
                      help='number of outputs', metavar='NOUTPUTS', type='int')
    parser.add_option('--img_depth', dest='img_depth', default=1,
                      help='image depth', metavar='IMG_DEPTH', type='int')
    (options, args) = parser.parse_args()

    if not options.do_learn and \
       not options.do_test and \
       not options.do_predict:
        print("\nSpecify learn (-l), test (-t), and/or predict (-p):\n\n")
        print(__doc__)
        sys.exit(1)

    import logging
    logfilename = options.logfilename or 'minerva_tricolumnar_time_epsilon.log'
    logging.basicConfig(
        filename=logfilename, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting...")
    logger.info(__file__)

    build_network_function = build_triamese_epsilon
    learn = categorical_learn_and_validate
    test = categorical_test
    predict = categorical_predict

    convpooldictlist = {}

    # x
    convpooldictlist_x = []
    # assume 127x(N) images
    x_convpool1dict = {}
    x_convpool1dict['nfilters'] = 12
    x_convpool1dict['filter_size'] = (8, 3)
    x_convpool1dict['pool_size'] = (2, 1)
    convpooldictlist_x.append(x_convpool1dict)
    # after 8x3 filters -> 120x(N-2) image, then maxpool -> 60x(N-2)
    x_convpool2dict = {}
    x_convpool2dict['nfilters'] = 20
    x_convpool2dict['filter_size'] = (7, 3)
    x_convpool2dict['pool_size'] = (2, 1)
    convpooldictlist_x.append(x_convpool2dict)
    # after 7x3 filters -> 54x(N-4) image, then maxpool -> 27x(N-4)
    x_convpool3dict = {}
    x_convpool3dict['nfilters'] = 28
    x_convpool3dict['filter_size'] = (6, 3)
    x_convpool3dict['pool_size'] = (2, 1)
    convpooldictlist_x.append(x_convpool3dict)
    # after 6x3 filters -> 22x(N-6) image, then maxpool -> 11x(N-6)
    x_convpool4dict = {}
    x_convpool4dict['nfilters'] = 36
    x_convpool4dict['filter_size'] = (6, 3)
    x_convpool4dict['pool_size'] = (2, 1)
    convpooldictlist_x.append(x_convpool4dict)
    # after 6x3 filters -> 6x(N-6) image, then maxpool -> 3x(N-6)

    # u
    convpooldictlist_u = []
    # assume 127x(N) images
    u_convpool1dict = {}
    u_convpool1dict['nfilters'] = 12
    u_convpool1dict['filter_size'] = (8, 5)
    u_convpool1dict['pool_size'] = (2, 1)
    convpooldictlist_u.append(u_convpool1dict)
    # after 8x3 filters -> 120x(N-2) image, then maxpool -> 60x(N-2)
    u_convpool2dict = {}
    u_convpool2dict['nfilters'] = 20
    u_convpool2dict['filter_size'] = (7, 3)
    u_convpool2dict['pool_size'] = (2, 1)
    convpooldictlist_u.append(u_convpool2dict)
    # after 7x3 filters -> 54x(N-4) image, then maxpool -> 27x(N-4)
    u_convpool3dict = {}
    u_convpool3dict['nfilters'] = 28
    u_convpool3dict['filter_size'] = (6, 3)
    u_convpool3dict['pool_size'] = (2, 1)
    convpooldictlist_u.append(u_convpool3dict)
    # after 6x3 filters -> 22x(N-6) image, then maxpool -> 11x(N-6)
    u_convpool4dict = {}
    u_convpool4dict['nfilters'] = 36
    u_convpool4dict['filter_size'] = (6, 3)
    u_convpool4dict['pool_size'] = (2, 1)
    convpooldictlist_u.append(u_convpool4dict)
    # after 6x3 filters -> 6x(N-6) image, then maxpool -> 3x(N-6)

    # v
    convpooldictlist_v = []
    # assume 127x(N) images
    v_convpool1dict = {}
    v_convpool1dict['nfilters'] = 12
    v_convpool1dict['filter_size'] = (8, 5)
    v_convpool1dict['pool_size'] = (2, 1)
    convpooldictlist_v.append(v_convpool1dict)
    # after 8x3 filters -> 120x(N-2) image, then maxpool -> 60x(N-2)
    v_convpool2dict = {}
    v_convpool2dict['nfilters'] = 20
    v_convpool2dict['filter_size'] = (7, 3)
    v_convpool2dict['pool_size'] = (2, 1)
    convpooldictlist_v.append(v_convpool2dict)
    # after 7x3 filters -> 54x(N-4) image, then maxpool -> 27x(N-4)
    v_convpool3dict = {}
    v_convpool3dict['nfilters'] = 28
    v_convpool3dict['filter_size'] = (6, 3)
    v_convpool3dict['pool_size'] = (2, 1)
    convpooldictlist_v.append(v_convpool3dict)
    # after 6x3 filters -> 22x(N-6) image, then maxpool -> 11x(N-6)
    v_convpool4dict = {}
    v_convpool4dict['nfilters'] = 36
    v_convpool4dict['filter_size'] = (6, 3)
    v_convpool4dict['pool_size'] = (2, 1)
    convpooldictlist_v.append(v_convpool4dict)
    # after 6x3 filters -> 6x(N-6) image, then maxpool -> 3x(N-6)

    convpooldictlist['x'] = convpooldictlist_x
    convpooldictlist['u'] = convpooldictlist_u
    convpooldictlist['v'] = convpooldictlist_v

    hyperpars={}
    hyperpars['num_epochs'] = options.n_epochs
    hyperpars['learning_rate'] = options.lrate
    hyperpars['momentum'] = options.momentum
    hyperpars['batchsize'] = options.batchsize
    hyperpars['learning_strategy'] = 'adagrad'   # meaningless right now

    imgdat = {}
    imgdat['views'] = 'xuv'
    imgdat['imgw'] = options.imgw
    imgdat['imgh'] = (options.imgh_x, options.imgh_u, options.imgh_v)

    runopts = {}
    runopts['data_file_list'] = options.dataset
    runopts['save_model_file'] = options.save_model_file
    runopts['start_with_saved_params'] = options.start_with_saved_params
    runopts['do_validation_pass'] = True
    runopts['debug_print'] = False
    runopts['be_verbose'] = options.be_verbose
    runopts['test_all_data'] = options.test_all_data
    runopts['write_db'] = True

    networkstr = {}
    networkstr['topology'] = convpooldictlist
    networkstr['nhidden'] = 196
    networkstr['dropoutp'] = 0.5
    networkstr['noutputs'] = 11
    networkstr['img_depth'] = options.img_depth
    networkstr['l1_penalty_scale'] = None
    networkstr['l2_penalty_scale'] = options.l2_penalty_scale

    logger.info(
        " Begin with saved pars? %s" % runopts['start_with_saved_params']
    )
    logger.info(" Saved parameters file: %s" % runopts['save_model_file'])
    logger.info(" Saved parameters file exists? %s" % \
                os.path.isfile(runopts['save_model_file']))
    logger.info(" Datasets: %s" % runopts['data_file_list'])
    dataset_statsinfo = 0
    for d in runopts['data_file_list']:
        dataset_statsinfo += os.stat(d).st_size
    logger.info(" Dataset size: %s" % dataset_statsinfo)
    logger.info(" Planned number of epochs: %s" % hyperpars['num_epochs'])
    logger.info(" Learning rate: %s" % hyperpars['learning_rate'])
    logger.info(" Momentum: %s" % hyperpars['momentum'])
    logger.info(" Batch size: %s" % hyperpars['batchsize'])
    logger.info(
        " L2 regularization penalty scale: %s" % networkstr['l2_penalty_scale']
    )

    if options.do_learn:
        networkstr['input_list'] = get_theano_input_tensors()
        learn(build_cnn_fn=build_network_function,
              hyperpars=hyperpars,
              imgdat=imgdat,
              runopts=runopts,
              networkstr=networkstr,
              get_list_of_hits_and_targets_fn=get_hits_and_targets
        )

    if options.do_test:
        networkstr['input_list'] = get_theano_input_tensors()
        test(build_cnn_fn=build_network_function,
             hyperpars=hyperpars,
             imgdat=imgdat,
             runopts=runopts,
             networkstr=networkstr,
             get_eventids_hits_and_targets_fn=get_eventids_hits_and_targets,
             get_list_of_hits_fn=get_hits
        )

    if options.do_predict:
        networkstr['input_list'] = get_theano_input_tensors()
        predict(build_cnn_fn=build_network_function,
                hyperpars=hyperpars,
                imgdat=imgdat,
                runopts=runopts,
                networkstr=networkstr,
                get_eventids_hits_and_targets_fn=get_eventids_hits_and_targets,
                get_id_tagged_inputlist_fn=get_hits_and_targets_tup
        )
