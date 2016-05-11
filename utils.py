def decode_eventid(eventid):
    """
    assume encoding from fuel_up_nukecc.py, etc.
    """
    eventid = str(eventid)
    phys_evt = eventid[-2:]
    eventid = eventid[:-2]
    gate = eventid[-4:]
    eventid = eventid[:-4]
    subrun = eventid[-4:]
    eventid = eventid[:-4]
    run = eventid
    return (run, subrun, gate, phys_evt)

def desc(f, n):
    """
    Check a merged hdf5 file
    """
    print "evtid: ", f['eventids'][n]
    print "nkaon: ", f['n-kaons'][n]
    print "nneut: ", f['n-neutrons'][n]
    print "nothr: ", f['n-others'][n]
    print "npi0s: ", f['n-pi0s'][n]
    print "nchpi: ", f['n-pions'][n]
    print "nprot: ", f['n-protons'][n]
    print "   zs: ", f['zs'][n]
