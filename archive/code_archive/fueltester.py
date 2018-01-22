#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from fuel.datasets import H5PYDataset
from six.moves import range

def test_mem_loader(nevt=1000, load_in_memory=True):
    fname = 'nukecc_fuel_me1B.hdf5'

    train_set = H5PYDataset(fname,
                            which_sets=('train',),
                            sources=['hits'])
    nexamp = train_set.num_examples

    if nevt > nexamp:
        nevt = nexamp

        train_set = H5PYDataset(fname,
                                subset=slice(0, nevt),
                                which_sets=('train',),
                                sources=['hits'],
                                load_in_memory=load_in_memory)
        handle = train_set.open()
        data = train_set.get_data(handle, slice(0, nevt))
        length = np.shape(data[0])[0]
        if length != nevt:
            raise
        counter = 0
        for _ in range(length):
            counter += data[0][0]

        train_set.close(handle)
