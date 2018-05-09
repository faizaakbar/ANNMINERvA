import tensorflow as tf

EVENT_DATA = 'event_data'
EVENTIDS = 'eventids'
PLANECODES = 'planecodes'
VTX_DATA = 'vtx_data'
SEGMENTS = 'segments'
ZS = 'zs'
GEN_DATA = 'gen_data'
ENRGY = 'E'
LEP_ENRGY = 'leptE'
QSQRD = 'Q2'
WINV = 'W'
XBJ = 'x'
YBJ = 'y'
CURRENT = 'current'
SIG_TYPE = 'sig_type'
INT_TYPE = 'int_type'
TARGETZ = 'targetZ'
HADRO_DATA = 'hadro_data'
ESUM_CHGDKAONS = 'esum_chgdkaons'
ESUM_CHGDPIONS = 'esum_chgdpions'
ESUM_HADMULTMEAS = 'esum_hadmultmeas'
ESUM_NEUTPIONS = 'esum_neutpions'
ESUM_NEUTRONS = 'esum_neutrons'
ESUM_OTHERS = 'esum_others'
ESUM_PROTONS = 'esum_protons'
N_CHGDKAONS = 'n_chgdkaons'
N_CHGDPIONS = 'n_chgdpions'
N_HADMULTMEAS = 'n_hadmultmeas'
N_NEUTPIONS = 'n_neutpions'
N_NEUTRONS = 'n_neutrons'
N_OTHERS = 'n_others'
N_PROTONS = 'n_protons'
IMG_DATA = 'img_data'
HITIMESU = 'hitimes-u'
HITIMESV = 'hitimes-v'
HITIMESX = 'hitimes-x'
PID_DATA = 'pid_data'
PIDU = 'pid-u'
PIDV = 'pid-v'
PIDX = 'pid-x'
LEPTO_DATA = 'lepto_data'
ESUM_ELECTRONS = 'esum_electrons'
ESUM_MUONS = 'esum_muons'
ESUM_TAUS = 'esum_taus'
N_ELECTRONS = 'n_electrons'
N_MUONS = 'n_muons'
N_TAUS = 'n_taus'


def make_master_dset_dict():
    d = {}
    d[EVENTIDS] = (EVENTIDS, tf.int64, EVENT_DATA)
    d[PLANECODES] = (PLANECODES, tf.int32, VTX_DATA)
    d[SEGMENTS] = (SEGMENTS, tf.uint8, VTX_DATA)
    d[ZS] = (ZS, tf.float32, VTX_DATA)
    d[HITIMESU] = (HITIMESU, tf.float32, IMG_DATA)
    d[HITIMESV] = (HITIMESV, tf.float32, IMG_DATA)
    d[HITIMESX] = (HITIMESX, tf.float32, IMG_DATA)
    d[TARGETZ] = (TARGETZ, tf.int32, GEN_DATA)
    d[QSQRD] = (QSQRD, tf.float32, GEN_DATA)
    d[WINV] = (WINV, tf.float32, GEN_DATA)
    d[XBJ] = (XBJ, tf.float32, GEN_DATA)
    d[YBJ] = (YBJ, tf.float32, GEN_DATA)
    d[CURRENT] = (CURRENT, tf.int32, GEN_DATA)
    d[INT_TYPE] = (INT_TYPE, tf.int32, GEN_DATA)
    d[SIG_TYPE] = (SIG_TYPE, tf.int32, GEN_DATA)
    d[ENRGY] = (ENRGY, tf.float32, GEN_DATA)
    d[LEP_ENRGY] = (LEP_ENRGY, tf.float32, GEN_DATA)
    d[ESUM_CHGDKAONS] = (ESUM_CHGDKAONS, tf.float32, HADRO_DATA)
    d[ESUM_CHGDPIONS] = (ESUM_CHGDPIONS, tf.float32, HADRO_DATA)
    d[ESUM_HADMULTMEAS] = (ESUM_HADMULTMEAS, tf.float32, HADRO_DATA)
    d[ESUM_NEUTPIONS] = (ESUM_NEUTPIONS, tf.float32, HADRO_DATA)
    d[ESUM_NEUTRONS] = (ESUM_NEUTRONS, tf.float32, HADRO_DATA)
    d[ESUM_OTHERS] = (ESUM_OTHERS, tf.float32, HADRO_DATA)
    d[ESUM_PROTONS] = (ESUM_PROTONS, tf.float32, HADRO_DATA)
    d[N_CHGDKAONS] = (N_CHGDKAONS, tf.int32, HADRO_DATA)
    d[N_CHGDPIONS] = (N_CHGDPIONS, tf.int32, HADRO_DATA)
    d[N_HADMULTMEAS] = (N_HADMULTMEAS, tf.int32, HADRO_DATA)
    d[N_NEUTPIONS] = (N_NEUTPIONS, tf.int32, HADRO_DATA)
    d[N_NEUTRONS] = (N_NEUTRONS, tf.int32, HADRO_DATA)
    d[N_OTHERS] = (N_OTHERS, tf.int32, HADRO_DATA)
    d[N_PROTONS] = (N_PROTONS, tf.int32, HADRO_DATA)
    d[ESUM_ELECTRONS] = (ESUM_ELECTRONS, tf.float32, LEPTO_DATA)
    d[ESUM_MUONS] = (ESUM_MUONS, tf.float32, LEPTO_DATA)
    d[ESUM_TAUS] = (ESUM_TAUS, tf.float32, LEPTO_DATA)
    d[N_ELECTRONS] = (N_ELECTRONS, tf.int32, LEPTO_DATA)
    d[N_MUONS] = (N_MUONS, tf.int32, LEPTO_DATA)
    d[N_TAUS] = (N_TAUS, tf.int32, LEPTO_DATA)
    d[PIDU] = (PIDU, tf.int32, PID_DATA)
    d[PIDV] = (PIDV, tf.int32, PID_DATA)
    d[PIDX] = (PIDX, tf.int32, PID_DATA)

    return d


MASTER_DSET_DICT = make_master_dset_dict()

# groups in the HDF5 file marked for translation to TFRecord; ultimately, we
# can't build this list dynamically because there is no way to get it from
# the TFRecord file by inspection - we must unpack with foreknowledge of
# what is there to parse the raw bytes.
HADMULTKINE_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    VTX_DATA: [PLANECODES, SEGMENTS, ZS],
    GEN_DATA: [QSQRD, WINV, XBJ, YBJ, CURRENT, INT_TYPE, TARGETZ],
    HADRO_DATA: [
        ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS,
        ESUM_NEUTPIONS, ESUM_NEUTRONS, ESUM_OTHERS, ESUM_PROTONS,
        N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS, N_NEUTPIONS,
        N_NEUTRONS, N_OTHERS, N_PROTONS
    ],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX],
    LEPTO_DATA: [
        ESUM_ELECTRONS, ESUM_MUONS, ESUM_TAUS,
        N_ELECTRONS, N_MUONS, N_TAUS
    ]
}
HADMULTKINE_TYPE = 'hadmultkineimgs'

WHOLEVT_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    VTX_DATA: [ZS],
    GEN_DATA: [
        ENRGY, LEP_ENRGY, QSQRD, WINV, XBJ, YBJ,
        CURRENT, INT_TYPE, SIG_TYPE, TARGETZ
    ],
    HADRO_DATA: [
        ESUM_CHGDKAONS, ESUM_CHGDPIONS, ESUM_HADMULTMEAS,
        ESUM_NEUTPIONS, ESUM_NEUTRONS, ESUM_OTHERS, ESUM_PROTONS,
        N_CHGDKAONS, N_CHGDPIONS, N_HADMULTMEAS, N_NEUTPIONS,
        N_NEUTRONS, N_OTHERS, N_PROTONS
    ],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX],
}
WHOLEVT_TYPE = 'wholevtimgs'

VTXFINDING_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    VTX_DATA: [PLANECODES, SEGMENTS, ZS],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX]
}
VTXFINDING_TYPE = 'vtxfndingimgs'

IMGING_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    VTX_DATA: [PLANECODES, SEGMENTS, ZS],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX]
}
IMGING_TYPE = 'mnvimgs'

SEGMENTATION_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX],
    PID_DATA: [PIDU, PIDV, PIDX]
}
SEGMENTATION_TYPE = "segmentation"


def make_mnv_data_dict_from_fields(list_of_fields):
    data_list = []
    for f in list_of_fields:
        data_list.extend([MASTER_DSET_DICT[f]])
    mnv_data = {}
    for datum in data_list:
        mnv_data[datum[0]] = {}
        mnv_data[datum[0]]['dtype'] = datum[1]
        mnv_data[datum[0]]['byte_data'] = None
        mnv_data[datum[0]]['group'] = datum[2]

    return mnv_data
