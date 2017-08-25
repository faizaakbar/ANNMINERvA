#!/usr/bin/env python
from __future__ import print_function
from MnvRecorderSQLite import MnvCategoricalSQLiteRecorder
import numpy as np


def test1(dbhandler):
    pred = 0
    probs = np.array([0.95, 0.05] + 9 * [0.0])
    eventid = 1172000001001001
    dbhandler.write_data(eventid, pred, probs)


def test2(dbhandler):
    results = dbhandler.read_data()
    print(results)


if __name__ == '__main__':
    dbhandler = MnvCategoricalSQLiteRecorder(11, 'testdb')
    test1(dbhandler)
    test2(dbhandler)
