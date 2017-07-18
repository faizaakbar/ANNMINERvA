"""
minerva test
"""
from __future__ import print_function

import tensorflow as tf

import os
import time

from MnvModelsTricolumnar import TriColSTEpsilon
from MnvModelsTricolumnar import make_default_convpooldict
from MnvDataReaders import MnvDataReaderVertexST
from MnvTFRunners import MnvTFRunnerCategorical
import mnv_utils

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    run_params_dict = mnv_utils.make_default_run_params_dict()
    feature_targ_dict = mnv_utils.make_default_feature_targ_dict()
    train_params_dict = mnv_utils.make_default_train_params_dict()
    img_params_dict = mnv_utils.make_default_img_params_dict()
    runner = MnvTFRunnerCategorical(
        run_params_dict=run_params_dict,
        feature_targ_dict=feature_targ_dict,
        train_params_dict=train_params_dict,
        img_params_dict=img_params_dict
    )
    runner.run_training(do_validation=True, short=True)
    runner.run_testing(short=True)


if __name__ == '__main__':
    tf.app.run()
