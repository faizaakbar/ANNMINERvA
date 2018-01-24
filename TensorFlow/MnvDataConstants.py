EVENT_DATA = 'event_data'
EVENTIDS = 'eventids'
PLANECODES = 'planecodes'
SEGMENTS = 'segments'
ZS = 'zs'
GEN_DATA = 'gen_data'
QSQRD = 'Q2'
WINV = 'W'
XBJ = 'x'
YBJ = 'y'
CURRENT = 'current'
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
LEPTO_DATA = 'lepto_data'
ESUM_ELECTRONS = 'esum_electrons'
ESUM_MUONS = 'esum_muons'
ESUM_TAUS = 'esum_taus'
N_ELECTRONS = 'n_electrons'
N_MUONS = 'n_muons'
N_TAUS = 'n_taus'

# groups in the HDF5 file marked for translation to TFRecord; ultimately, we
# can't build this list dynamically because there is no way to get it from
# the TFRecord file by inspection - we must unpack with foreknowledge of
# what is there to parse the raw bytes.
HADMULTKINE_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS, PLANECODES, SEGMENTS, ZS],
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

VTXFINDING_GROUPS_DICT = {
    EVENT_DATA: [EVENTIDS, PLANECODES, SEGMENTS, ZS],
    IMG_DATA: [HITIMESU, HITIMESV, HITIMESX]
}

VALID_SET_OF_GROUPS = set(
    HADMULTKINE_GROUPS_DICT.keys() + VTXFINDING_GROUPS_DICT.keys()
)
