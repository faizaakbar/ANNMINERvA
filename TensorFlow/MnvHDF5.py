import h5py
import numpy as np
import tensorflow as tf
import logging
from MnvDataConstants import *


LOGGER = logging.getLogger(__name__)


class MnvHDF5Reader:
    """
    the `minerva_hdf5_reader` will return numpy ndarrays of data for given
    ranges. user should call `open()` and `close()` to start/finish.
    """
    def __init__(self, hdf5_file, data_dict):
        self.file = hdf5_file
        self._f = None
        self._dd = data_dict

    def open(self):
        LOGGER.info("Opening hdf5 file {}".format(self.file))
        self._f = h5py.File(self.file, 'r')
        for group in self._f:
            for dset in self._f[group]:
                LOGGER.info('{:>12}/{:>12}: {:>8}: shape = {}'.format(
                    group, dset,
                    np.dtype(self._f[group][dset]),
                    np.shape(self._f[group][dset])
                ))

    def close(self):
        try:
            self._f.close()
        except AttributeError:
            LOGGER.info('hdf5 file is not open yet.')

    def get_data(self, name, start_idx, stop_idx):
        try:
            group = self._dd[name]['group']
            return self._f[group][name][start_idx: stop_idx]
        except KeyError:
            LOGGER.info('{} data is not available in this HDF5 file.')
            return []

    def get_nevents(self, g=EVENT_DATA):
        sizes = [self._f[g][d].shape[0] for d in self._f[g]]
        if min(sizes) != max(sizes):
            msg = "All dsets must have the same size!"
            LOGGER.error(msg)
            raise ValueError(msg)
        return sizes[0]


def make_mnv_data_dict(list_of_groups):
    """
    create a dict of fields to extract from the hdf5 with target dtypes.
    """
    # eventids are really (in the hdf5) uint64, planecodes are really uint16;
    # use tf.{int64,int32,uint8} because these are the dtypes that one-hot
    # supports (_not_ int16 or uint16, at least in TF v1.2); use int64 instead
    # of unit64 because reshape supports int64 (and not uint64).
    data_list = []
    for g in list_of_groups:
        if g in VALID_SET_OF_GROUPS:
            if g == EVENT_DATA:
                data_list.extend([
                    (EVENTIDS, tf.int64, g),
                    (PLANECODES, tf.int32, g),
                    (SEGMENTS, tf.uint8, g),
                    (ZS, tf.float32, g),
                ])
            if g == IMG_DATA:
                data_list.extend([
                    (HITIMESU, tf.float32, g),
                    (HITIMESV, tf.float32, g),
                    (HITIMESX, tf.float32, g),
                ])
            if g == GEN_DATA:
                data_list.extend([
                    (QSQRD, tf.float32, g),
                    (WINV, tf.float32, g),
                    (XBJ, tf.float32, g),
                    (YBJ, tf.float32, g),
                    (CURRENT, tf.int32, g),
                    (INT_TYPE, tf.int32, g),
                    (TARGETZ, tf.int32, g),
                ])
            if g == HADRO_DATA:
                data_list.extend([
                    (ESUM_CHGDKAONS, tf.float32, g),
                    (ESUM_CHGDPIONS, tf.float32, g),
                    (ESUM_HADMULTMEAS, tf.float32, g),
                    (ESUM_NEUTPIONS, tf.float32, g),
                    (ESUM_NEUTRONS, tf.float32, g),
                    (ESUM_OTHERS, tf.float32, g),
                    (ESUM_PROTONS, tf.float32, g),
                    (N_CHGDKAONS, tf.int32, g),
                    (N_CHGDPIONS, tf.int32, g),
                    (N_HADMULTMEAS, tf.int32, g),
                    (N_NEUTPIONS, tf.int32, g),
                    (N_NEUTRONS, tf.int32, g),
                    (N_OTHERS, tf.int32, g),
                    (N_PROTONS, tf.int32, g),
                ])
            if g == LEPTO_DATA:
                data_list.extend([
                    (ESUM_ELECTRONS, tf.float32, g),
                    (ESUM_MUONS, tf.float32, g),
                    (ESUM_TAUS, tf.float32, g),
                    (N_ELECTRONS, tf.int32, g),
                    (N_MUONS, tf.int32, g),
                    (N_TAUS, tf.int32, g),
                ])
        else:
            raise ValueError('Unrecognized group')
    mnv_data = {}
    for datum in data_list:
        mnv_data[datum[0]] = {}
        mnv_data[datum[0]]['dtype'] = datum[1]
        mnv_data[datum[0]]['byte_data'] = None
        mnv_data[datum[0]]['group'] = datum[2]

    return mnv_data
