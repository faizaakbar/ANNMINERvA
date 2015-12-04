#!/usr/bin/env python
import gzip
import cPickle

f = gzip.open('skim_data_target0.pkl.gz', 'rb')
learn_data, test_data, valid_data = cPickle.load(f)
f.close()
