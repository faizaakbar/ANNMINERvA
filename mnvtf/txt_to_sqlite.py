#!/usr/bin/env/python
"""
Read a text file and produce a SQLite db.

python process_txt_to_sqlite.py -i f1.txt,f2.txt \
    [-o out.db] \
    [-n n_classes (67)]
    [-f format ('ghosh')]

Valid formats: 'ghosh', 'perdue'

* 'ghosh' format (tsv):
    run	subrun gate phys_evt prob00 prob01 ... probNN prediction
* 'perdue' format (csv):
    run,subrun,gate,phys_evt,prediction,prob00,prob01,...,probNN
"""
from __future__ import print_function
try:
    from MnvRecorderSQLite import MnvCategoricalSQLiteRecorder
except ImportError as e:
    print('Cannot import MnvRecorderSQLite')
    raise e
from mnv_utils import encode_eventid

VALID_FORMATS = ['ghosh', 'perdue']


def arg_list_split(option, opt, value, parser):
    """ split comma separated args """
    setattr(parser.values, option.dest, value.split(','))


def get_entry(inputs):
    line_count = 0
    for f in inputs:
        with open(f, 'r') as fp:
            for line in fp:
                yield line
                line_count += 1
                if line_count % 1000 == 0:
                    print('processed', line_count, 'lines')


def parse_line_ghosh(line):
    elems = line.split()
    evtid = encode_eventid(elems[0], elems[1], elems[2], elems[3])
    elems = elems[4:]
    pred = elems[-1]
    elems = elems[:-1]
    return evtid, pred, elems


def parse_line_perdue(line):
    elems = line.split(',')
    evtid = encode_eventid(elems[0], elems[1], elems[2], elems[3])
    elems = elems[4:]
    pred = elems[0]
    elems = elems[1:]
    return evtid, pred, elems
    

if __name__ == '__main__':
    from optparse import OptionParser
    import sys

    parser = OptionParser(usage=__doc__)
    parser.add_option('-i', type='string', action='callback', dest='inputs',
                      callback=arg_list_split)
    parser.add_option('-o', type='string', help=r'Output file', dest='output')
    parser.add_option('-n', type='int', help=r'Number of classes',
                      dest='n_classes', default=67)
    parser.add_option('-f', type='string', help=r'Format', dest='fmat',
                      default='ghosh')

    (options, args) = parser.parse_args()

    if options.inputs is None or options.fmat not in VALID_FORMATS:
        print(__doc__)
        sys.exit(1)

    if options.output is None:
        basename = options.inputs[0].split('/')[-1]
        output = basename.split('.')[0] + '.db'
    else:
        output = options.output

    parse_record_fn = None
    if options.fmat == 'ghosh':
        parse_record_fn = parse_line_ghosh
    elif options.fmat == 'perdue':
        parse_record_fn = parse_line_perdue

    data_recorder = MnvCategoricalSQLiteRecorder(
        options.n_classes, output
    )

    input_gen = get_entry(options.inputs)
    for record in input_gen:
        if record[0] == '#' or record[0:3] == 'run':
            continue
        evtid, pred, probs = parse_record_fn(record)
        data_recorder.write_data(evtid, pred, probs)
