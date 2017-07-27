#!/usr/bin/env python
"""
Do persistence
"""
import os
import logging

import mnv_utils

LOGGER = logging.getLogger(__name__)


class MnvCategoricalTextRecorder:
    """
    record segments or planecodes in a sqlite db
    """
    def __init__(self, db_base_name):
        self.db_name = db_base_name + '.txt'
        if os.path.isfile(self.db_name):
            LOGGER.info('found existing record file {}, removing'.format(
                self.db_name
            ))
            os.remove(self.db_name)
        LOGGER.info('using record file {}'.format(self.db_name))

    def write_data(self, eventid, pred, probs):
        with open(self.db_name, 'a+') as f:
            evtid_string = ','.join(mnv_utils.decode_eventid(eventid))
            probs_string = ','.join([str(i) for i in probs])
            msg = evtid_string + ',' + str(pred) + ',' + probs_string + '\n'
            f.write(msg)
        return None

    def read_data(self):
        """ do not call this on large files """
        with open(self.db_name, 'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        return content
