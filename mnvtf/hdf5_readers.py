import h5py
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


class MnvHDF5Reader(object):
    """
    the `minerva_hdf5_reader` will return numpy ndarrays of data for given
    ranges. user should call `open()` and `close()` to start/finish.

    Note that this class assumes a file structure like:
    f[group_1][dataset_1]
    f[group_1][dataset_2]
    ...
    f[group_n][dataset_m]
    """

    def __init__(self, hdf5_file):
        self.file = hdf5_file
        self._f = None
        self._dd = None
        self._groups = None

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
        self._groups = [group for group in self._f]
        self._dd = {}
        for g in self._groups:
            self._dd[g] = [dset for dset in self._f[g]]

    def close(self):
        try:
            self._f.close()
            self._dd = None
            self._groups = None
        except AttributeError:
            LOGGER.info('hdf5 file is not open yet.')

    def get_data(self, name, start_idx, stop_idx):
        try:
            for g in self._groups:
                if name in self._dd[g]:
                    group = g
                    break
            else:
                raise ValueError(
                    '{} is not present in any group.'.format(name)
                )
            return self._f[group][name][start_idx: stop_idx]
        except KeyError:
            msg = '{} data is not available in this HDF5 file.'.format(name)
            LOGGER.info(msg)
            raise ValueError(msg)

    def get_vtxfinder_datum(self, idx):
        x = self._f['img_data']['hitimes-x'][idx]
        x = np.moveaxis(x, 0, -1)
        u = self._f['img_data']['hitimes-u'][idx]
        u = np.moveaxis(u, 0, -1)
        v = self._f['img_data']['hitimes-v'][idx]
        v = np.moveaxis(v, 0, -1)
        evtids = self._f['event_data']['eventids'][idx]
        pcodes = self._f['vtx_data']['planecodes'][idx].reshape([-1])
        oh_pcodes = np.zeros((1, 174))
        oh_pcodes[0, pcodes] = 1
        return x, u, v, evtids, oh_pcodes.reshape(174,)

    def get_vtxfinder_data(self, start, stop):
        x = self._f['img_data/hitimes-x'][start: stop]
        u = self._f['img_data/hitimes-u'][start: stop]
        v = self._f['img_data/hitimes-v'][start: stop]
        x = np.moveaxis(x, 1, -1)
        u = np.moveaxis(u, 1, -1)
        v = np.moveaxis(v, 1, -1)
        pcodes = self._f['vtx_data/planecodes'][start: stop].reshape([-1])
        oh_pcodes = np.zeros((pcodes.size, 174), dtype=np.int32)
        oh_pcodes[np.arange(pcodes.size), pcodes] = 1
        eventids = self._f['event_data/eventids'][start: stop].reshape([-1])
        return x, u, v, eventids, oh_pcodes

    def get_nevents(self, group=None):
        sizes = [self._f[group][d].shape[0] for d in self._f[group]]
        if min(sizes) != max(sizes):
            msg = "All dsets must have the same size!"
            LOGGER.error(msg)
            raise ValueError(msg)
        return sizes[0]


class MnvHDF5LegacyReader(object):
    """
    the `minerva_hdf5_reader` will return numpy ndarrays of data for given
    ranges. user should call `open()` and `close()` to start/finish.

    Note that this class assumes a file structure like:
    f[dataset_1]
    f[dataset_2]
    ...
    f[dataset_m]

    (This is the "legacy" HDF5 structure - from the old ROOT->txt->HDF5
    workflow.)
    """

    def __init__(self, hdf5_file):
        self.file = hdf5_file
        self._f = None

    def open(self):
        LOGGER.info("Opening hdf5 file {}".format(self.file))
        self._f = h5py.File(self.file, 'r')
        for dset in self._f:
            LOGGER.info('{:>12}: {:>8}: shape = {}'.format(
                dset, np.dtype(self._f[dset]), np.shape(self._f[dset])
            ))

    def close(self):
        try:
            self._f.close()
        except AttributeError:
            LOGGER.info('hdf5 file is not open yet.')

    def get_data(self, name, start_idx, stop_idx):
        try:
            return self._f[name][start_idx: stop_idx]
        except KeyError:
            msg = '{} data is not available in this HDF5 file.'.format(name)
            LOGGER.info(msg)
            raise ValueError(msg)

    def get_nevents(self, group=None):
        sizes = [self._f[d].shape[0] for d in self._f]
        if min(sizes) != max(sizes):
            msg = "All dsets must have the same size!"
            LOGGER.error(msg)
            raise ValueError(msg)
        return sizes[0]


def test():
    path = '/Users/perdue/Documents/MINERvA/AI/hdf5/201801/'
    filen = 'hadmultkineimgs_127x94_me1Amc_tiny.hdf5'
    reader = MnvHDF5Reader(path + filen)
    try:
        reader.open()
        print reader.get_data('eventids', 0, 10)
        try:
            print reader.get_data('n_taus', 0, 10)
        except ValueError:
            print 'n_taus is not stored here.'
        try:
            print reader.get_data('n_rainbow_unicorns', 0, 10)
        except ValueError:
            print 'of course there are no unicorns here.'
        reader.close()
    except IOError:
        print 'hdf5 file not found'


def test_legacy():
    path = '/Users/perdue/Documents/MINERvA/AI/hdf5/201600/'
    filen = 'minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25' \
            '_xtxutuvtv_me1Bmc.hdf5'
    reader = MnvHDF5LegacyReader(path + filen)
    try:
        reader.open()
        print reader.get_data('eventids', 0, 10)
        try:
            print reader.get_data('planecodes', 0, 10)
        except ValueError:
            print 'planecodes are not stored here.'
        try:
            print reader.get_data('n_rainbow_unicorns', 0, 10)
        except ValueError:
            print 'of course there are no unicorns here.'
        reader.close()
    except IOError:
        print 'hdf5 file not found'


if __name__ == '__main__':
    test_legacy()
