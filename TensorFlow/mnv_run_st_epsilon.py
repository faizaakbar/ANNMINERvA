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


