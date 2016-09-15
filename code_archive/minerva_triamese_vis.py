#!/usr/bin/env python
"""
This is an attempt at a "triamese" network operating on Minerva X, U, V.

Execution:
    python minerva_triamese_test.py -h / --help

See ANNMINERvA/fuel_up_convdata.py for an HDF5 builder that sets up an
appropriate data file.
"""
from __future__ import print_function

import os

from minerva_ann_networks import build_triamese_beta
from minerva_ann_operate_networks import view_layer_activations


def arg_list_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data_list', dest='dataset',
                      help='Data set list (csv)', metavar='DATASETLIST',
                      type='string', action='callback',
                      callback=arg_list_split)
    parser.add_option('-v', '--verbose', dest='be_verbose', default=False,
                      help='Verbose predictions', metavar='BE_VERBOSE',
                      action='store_true')
    parser.add_option('-s', '--save_file', dest='save_model_file',
                      default='./lminervatriamese_beta.npz',
                      help='File name for parameters',
                      metavar='SAVE_FILE_NAME')
    parser.add_option('-a', '--test_all', dest='test_all_data',
                      default=False, help='Treat all data as test data',
                      metavar='ALL_TEST', action='store_true')
    (options, args) = parser.parse_args()

    print("Starting...")
    print(__file__)
    print(" Saved parameters file:", options.save_model_file)
    print(" Saved parameters file exists?",
          os.path.isfile(options.save_model_file))
    print(" Datasets:", options.dataset)
    dataset_statsinfo = 0
    for d in options.dataset:
        dataset_statsinfo += os.stat(d).st_size
    print(" Dataset size:", dataset_statsinfo)

    build_network_function = build_triamese_beta
    vis = view_layer_activations

    # assume 50x50 images
    convpooldictlist = []
    convpool1dict = {}
    convpool1dict['nfilters'] = 32
    convpool1dict['filter_size'] = (3, 3)
    convpool1dict['pool_size'] = (2, 1)
    convpooldictlist.append(convpool1dict)
    #
    convpool2dict = {}
    convpool2dict['nfilters'] = 32
    convpool2dict['filter_size'] = (3, 3)
    convpool2dict['pool_size'] = (2, 1)
    convpooldictlist.append(convpool2dict)
    #
    nhidden = 128

    vis(build_cnn=build_network_function,
        data_file_list=options.dataset,
        save_model_file=options.save_model_file,
        be_verbose=options.be_verbose,
        convpooldictlist=convpooldictlist,
        test_all_data=options.test_all_data,
        nhidden=nhidden)
