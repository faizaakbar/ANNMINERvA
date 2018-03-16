from __future__ import print_function
import numpy as np
import h5py
# import matplotlib.pyplot as plt


def extract_id(evtid):
    if np.shape(evtid) == ():
        eventid = str(evtid)
    elif np.shape(evtid) == (1,):
        eventid = str(evtid[0])
    elif np.shape(evtid) == (1, 1):
        eventid = str(evtid[0][0])
    else:
        raise ValueError('Improper shape for event id!')
    return eventid


def decode_eventid(eventid):
    """ assume 64bit encoding """
    eventid = extract_id(eventid)
    phys_evt = eventid[-2:]
    eventid = eventid[:-2]
    gate = eventid[-4:]
    eventid = eventid[:-4]
    subrun = eventid[-4:]
    eventid = eventid[:-4]
    # note that run is not zero-padded
    run = eventid
    return (run, subrun, gate, phys_evt)


def decode_eventid32a(eventid):
    """ assume 32-bit "a" encoding """
    eventid = extract_id(eventid)
    phys_evt = eventid[-2:]
    eventid = eventid[:-2]
    # note that run is not zero-padded
    run = eventid
    return (run, phys_evt)


def decode_eventid32b(eventid):
    """ assume 32-bit "b" encoding """
    eventid = extract_id(eventid)
    gate = eventid[-4:]
    eventid = eventid[:-4]
    subrun = eventid.zfill(4)
    return (subrun, gate)


def decode_eventid32_combo(evta, evtb):
    (run, phys_evt) = decode_eventid32a(evta)
    (subrun, gate) = decode_eventid32b(evtb)
    return (run, subrun, gate, phys_evt)


def compare_evtid_encodings(evtid64, evtid32a, evtid32b):
    (run64, subrun64, gate64, phys_evt64) = decode_eventid(evtid64)
    (run32, subrun32, gate32, phys_evt32) = \
        decode_eventid32_combo(evtid32a, evtid32b)
    if run64 == run32 and \
       subrun64 == subrun32 \
       and gate64 == gate32 \
       and phys_evt64 == phys_evt32:
        return True
    return False


if __name__ == '__main__':

    HDF5_DIR = '/Users/perdue/Documents/MINERvA/AI/hdf5/201801'
    f = h5py.File(HDF5_DIR + '/hadmultkineimgs_127x94_me1Amc.hdf5', 'r')
    n_events = f['event_data']['eventids'].shape[0]

    for i in range(n_events):
        evtid = f['event_data']['eventids'][i: i + 1]
        evtia = f['event_data']['eventids_a'][i: i + 1]
        evtib = f['event_data']['eventids_b'][i: i + 1]
        check = compare_evtid_encodings(evtid, evtia, evtib)
        if not check:
            print('found a mismatch!')
            print(i, evtid, evtia, evtib)
