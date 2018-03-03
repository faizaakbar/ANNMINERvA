#!/usr/bin/env python
"""
"""
import gzip
from utils import encode_eventid


class MnvCategoricalTextReader:
    """
    record segments or planecodes in a sqlite db
    """
    def __init__(self, db_name):
        self.db_name = db_name
        self.is_zipped = db_name[-3:] == '.gz'
        if self.is_zipped:
            self.open_fn = gzip.open
        else:
            self.open_fn = open

    def _parse_line_perdue(self, line):
        elems = line.split(',')
        evtid = encode_eventid(elems[0], elems[1], elems[2], elems[3])
        elems = elems[4:]
        pred = int(elems[0])
        elems = elems[1:]
        elems = [float(x.strip()) for x in elems]
        return [evtid, pred, elems]

    def read_data_generator(self):
        """ create a generator for going through the data line by line """
        with self.open_fn(self.db_name, 'r') as fp:
            for line in fp:
                yield self._parse_line_perdue(line)

