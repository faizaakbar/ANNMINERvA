#!/usr/bin/env/python
"""
read and check a predictions file (.txt.gz) or (.db.gz)
"""
from __future__ import print_function
try:
    from mnvtf.reader_sqlite import MnvCategoricalSQLiteReader
except ImportError as e:
    print('Cannot import recorder_sqlite')
    raise e
from mnvtf.reader_text import MnvCategoricalTextReader
from mnvtf.utils import encode_eventid


def arg_list_split(option, opt, value, parser):
    """ split comma separated args """
    setattr(parser.values, option.dest, value.split(','))


def get_playlist_name(file_name):
    import re
    data_pattern = re.compile(r'(.*)predictionsME[0-9][A-Z]DATANX(.*)')
    # optional _MISSINGFILES?


if __name__ == '__main__':
    from optparse import OptionParser
    import sys

    parser = OptionParser(usage=__doc__)
    parser.add_option('-i', type='string', action='callback', dest='inputs',
                      callback=arg_list_split)

    (options, args) = parser.parse_args()

    if options.inputs is None:
        print(__doc__)
        sys.exit(1)

    
