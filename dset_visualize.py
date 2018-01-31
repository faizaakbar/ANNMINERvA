#!/usr/bin/env python
"""
Usage:
    python dset_visualize.py -f [file name] -n [optional: # of evts, def==10]
"""
import sys
from collections import OrderedDict
import pylab
import tensorflow as tf
import numpy as np

from mnvtf.mnv_utils import get_reader_class
from mnvtf.mnv_utils import make_data_reader_dict
from mnvtf.MnvDataConstants import HITIMESU, HITIMESV, HITIMESX
from mnvtf.MnvDataConstants import EVENT_DATA, EVENTIDS
from mnvtf.MnvDataConstants import PLANECODES, SEGMENTS, ZS
from mnvtf.MnvHDF5 import MnvHDF5Reader


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


class MnvDataReader:
    def __init__(
            self,
            filename,
            n_events=10,
            views=['x', 'u', 'v'],
            img_sizes=(94, 47),
            n_planecodes=173,
            tfrecord_reader_type=None,
            data_format='NHWC'
    ):
        """
        currently, only work with compressed tfrecord files; assume compression
        for hdf5 is inside, etc.
        """
        self._f = None
        self.filename = filename
        self.n_events = n_events
        self.views = views
        self.img_sizes = img_sizes
        self.n_planecodes = n_planecodes
        self.img_shp = (127, img_sizes[0], img_sizes[1], 2)
        self.data_format = data_format

        ext = self.filename.split('.')[-1]
        self.compression = ext if ext in ['zz', 'gz'] else ''
        if self.compression in ['zz', 'gz']:
            self.filetype = filename.split('.')[-2]
        else:
            self.filetype = ext

        self.hdf5_extensions = ['hdf5', 'h5']
        self.tfr_extensions = ['tfrecord']

        if tfrecord_reader_type is None:
            # attempt to infer the reader type from the filename.
            tfrecord_reader_type = filename.split('/')[-1]
            tfrecord_reader_type = tfrecord_reader_type.split('_')[0]
        self.tfrecord_reader = get_reader_class(tfrecord_reader_type)

    def _read_tfr(self):
        data_dict = {}
        data_dict['energies+times'] = {}

        dd = make_data_reader_dict(
            filenames_list=[self.filename],
            batch_size=self.n_events,
            name='test_read',
            compression=self.compression,
            img_shp=self.img_shp,
            data_format=self.data_format,
            n_planecodes=self.n_planecodes
        )
        reader = self.tfrecord_reader(dd)
        # get an ordered dict
        batch_dict = reader.batch_generator()

        def tp_tnsr(tnsr):
            return np.transpose(tnsr, [0, 3, 1, 2])

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                tensor_list = sess.run(batch_dict.values())
                results = OrderedDict(zip(batch_dict.keys(), tensor_list))
                data_dict['energies+times']['x'] = tp_tnsr(results[HITIMESX])
                data_dict['energies+times']['u'] = tp_tnsr(results[HITIMESU])
                data_dict['energies+times']['v'] = tp_tnsr(results[HITIMESV])
                data_dict['eventids'] = results[EVENTIDS]
                if PLANECODES in results.keys():
                    data_dict['planecodes'] = np.argmax(
                        results[PLANECODES], axis=1
                    ).reshape(results[PLANECODES].shape[0], 1)
                if SEGMENTS in results.keys():
                    data_dict['segments'] = np.argmax(
                        results[SEGMENTS], axis=1
                    ).reshape(results[SEGMENTS].shape[0], 1)
                if ZS in results.keys():
                    data_dict['zs'] = results[ZS]
            except tf.errors.OutOfRangeError:
                print('Reading stopped - queue is empty.')
            finally:
                coord.request_stop()
                coord.join(threads)

        return data_dict

    def _read_hdf5(self):
        """
        possibilities: energy tensors, time tensors, energy+time tensors
        (2-deep). get everything there into a dictionary keyed by type,
        and then by view.
        """
        data_dict = {}
        data_dict['energies+times'] = {}

        m = MnvHDF5Reader(self.filename)
        m.open()
        n_events = m.get_nevents(group=EVENT_DATA)
        n_read = min(n_events, self.n_events)
        data_dict['energies+times']['x'] = m.get_data(HITIMESX, 0, n_read)
        data_dict['energies+times']['u'] = m.get_data(HITIMESU, 0, n_read)
        data_dict['energies+times']['v'] = m.get_data(HITIMESV, 0, n_read)
        data_dict['eventids'] = m.get_data(EVENTIDS, 0, n_read)
        data_dict['planecodes'] = m.get_data(PLANECODES, 0, n_read)
        data_dict['segments'] = m.get_data(SEGMENTS, 0, n_read)
        data_dict['zs'] = m.get_data(ZS, 0, n_read)
        m.close()

        return data_dict

    def read_data(self):
        """
        return a dictionary of ndarrays, keyed by 'x', 'u', and 'v',
        each with shape (N, C, H, W) - could be anywhere from 1 to 3 views.
        """
        if self.filetype in self.hdf5_extensions:
            return self._read_hdf5()
        elif self.filetype in self.tfr_extensions:
            return self._read_tfr()
        else:
            raise ValueError('Invalid file type extension!')


def make_plots(data_dict, max_events, normed_img):
    """
    cases:
    * 'energies+times',
    * 'energies' and 'times' separately,
    * or 'energies' or 'times'
    If 2-deep tensor, assume energy is index 0, time is index 1
    """
    target_plane_codes = {9: 1, 18: 2, 27: 3, 44: 4, 49: 5}
    pkeys = []
    for k in data_dict.keys():
        if len(data_dict[k]) > 0:
            pkeys.append(k)
    print('Data dictionary present keys: {}'.format(pkeys))

    types = ['energy', 'time']
    views = ['x', 'u', 'v']   # TODO? build dynamically?

    # only working with two-deep imgs these days
    # plotting_two_tensors = True

    evt_plotted = 0
    for counter in range(len(data_dict['eventids'])):
        evtid = data_dict['eventids'][counter]
        segment = data_dict['segments'][counter] \
            if len(data_dict['segments']) > 0 else -1
        planecode = data_dict['planecodes'][counter] \
            if len(data_dict['planecodes']) > 0 else -1
        (run, subrun, gate, phys_evt) = decode_eventid(evtid)
        if evt_plotted > max_events:
            break
        print('Plotting entry {}: {}: {} - {} - {} - {} for segment {} / planecode {}'.format(
                  counter, evtid, run, subrun, gate, phys_evt, segment, planecode
              ))

        # run, subrun, gate, phys_evt = decode_eventid(evtid)
        fig_wid = 9
        fig_height = 6
        grid_height = 2
        fig = pylab.figure(figsize=(fig_wid, fig_height))
        if planecode in target_plane_codes.keys():
            fig.suptitle('{}/{}/{}/{}: seg {} / pcode {} / targ {}'.format(
                run, subrun, gate, phys_evt,
                segment, planecode, target_plane_codes[planecode[0]]
            ))
        else:
            fig.suptitle('{}/{}/{}/{}: seg {} / pcode {}'.format(
                run, subrun, gate, phys_evt, segment, planecode
            ))
        gs = pylab.GridSpec(grid_height, 3)

        for i, t in enumerate(types):
            datatyp = 'energies+times'
            # set the bounds on the color scale
            if normed_img:
                minv = 0 if t == 'energy' else -1
                maxv = 1
            else:
                maxes = []
                mins = []
                for v in views:
                    maxes.append(
                        np.abs(np.max(data_dict[datatyp][v][counter, i, :, :]))
                    )
                    mins.append(
                        np.abs(np.max(data_dict[datatyp][v][counter, i, :, :]))
                    )
                minv = np.max(mins)
                maxv = np.max(maxes)
                maxex = maxv if maxv > minv else minv
                minv = 0 if minv < 0.0001 else 0 if t == 'energy' else -maxv
                maxv = maxex
            for j, view in enumerate(views):
                gs_pos = i * 3 + j
                ax = pylab.subplot(gs[gs_pos])
                ax.axis('on')
                ax.xaxis.set_major_locator(pylab.NullLocator())
                ax.yaxis.set_major_locator(pylab.NullLocator())
                cmap = 'jet' if t == 'energy' else 'bwr'
                cbt = 'energy' if t == 'energy' else 'times'
                datap = data_dict[datatyp][view][counter, i, :, :]
                # make the plot
                im = ax.imshow(
                    datap,
                    cmap=pylab.get_cmap(cmap),
                    interpolation='nearest',
                    vmin=minv, vmax=maxv
                )
                cbar = pylab.colorbar(im, fraction=0.04)
                cbar.set_label(cbt, size=9)
                cbar.ax.tick_params(labelsize=6)
                pylab.title(t + ' - ' + view, fontsize=12)
                pylab.xlabel('plane', fontsize=10)
                pylab.ylabel('strip', fontsize=10)
        figname = 'evt_%d.pdf' % (counter)
        pylab.savefig(figname, bbox_inches='tight')
        pylab.close()
        evt_plotted += 1


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-f', '--file', dest='filename',
                      help='Dset file name', metavar='FILENAME',
                      default=None, type='string')
    parser.add_option('-n', '--nevents', dest='n_events', default=10,
                      help='Number of events', metavar='N_EVENTS',
                      type='int')
    parser.add_option('--imgw_x', dest='imgw_x', default=94,
                      help='Image width (x)', metavar='IMG_WIDTHX',
                      type='int')
    parser.add_option('--imgw_uv', dest='imgw_uv', default=47,
                      help='Image width (uv)', metavar='IMG_WIDTHUV',
                      type='int')
    parser.add_option('--n_planecodes', dest='n_planecodes', default=173,
                      help='Number of planecodes (onehot)',
                      metavar='N_PLANECODES', type='int')
    parser.add_option('--normed_img', dest='normed_img', default=False,
                      help='Image from normalized source',
                      metavar='NORMED_IMG', action='store_true')

    (options, args) = parser.parse_args()

    if not options.filename:
        print("\nSpecify file (-f):\n\n")
        print(__doc__)
        sys.exit(1)

    img_sizes = (options.imgw_x, options.imgw_uv)
    reader = MnvDataReader(
        filename=options.filename,
        n_events=options.n_events,
        img_sizes=img_sizes,
        n_planecodes=options.n_planecodes
    )
    dd = reader.read_data()

    make_plots(dd, options.n_events, options.normed_img)
