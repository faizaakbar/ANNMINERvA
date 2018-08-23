"""
build a set from the event numbers in an HDF5 file and write
those numbers to a file.
"""
import h5py
import sys
from mnvtf.evtid_utils import decode_eventid
from mnvtf.utils import gz_compress

fname = sys.argv[1]
f = h5py.File(fname, 'r')
n_evt_raw = f['event_data']['eventids'].shape[0]
evtids = f['event_data']['eventids'][:]

out_name = '.'.join(fname.split('.')[:-1]) + '_evts_set.txt'
with open(out_name, 'w') as fout:
    for evtid in evtids:
        run, subrun, gate, phys_evt = decode_eventid(evtid)
        fout.write(','.join([run, subrun, gate, phys_evt]))
        fout.write('\n')
    evtids_set = set(list(evtids.reshape(n_evt_raw)))
    if len(evtids_set) != n_evt_raw:
        print('duplicate event ids in %s' % fname)
gz_compress(out_name)
