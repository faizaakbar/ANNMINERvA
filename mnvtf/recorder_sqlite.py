#!/usr/bin/env python
"""
Do persistence
"""
import os
import shutil
import logging
import gzip

from six.moves import range

from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float
from sqlalchemy import UniqueConstraint
from sqlalchemy import MetaData
from sqlalchemy import select

from mnvtf.evtid_utils import decode_eventid

LOGGER = logging.getLogger(__name__)


class MnvCategoricalSQLiteRecorder:
    """
    record segments or planecodes in a sqlite db
    """
    _allowed_number_of_classes = [11, 67, 173]

    def __init__(self, n_classes, db_base_name):
        LOGGER.info('Setting up {}...'.format(
            self.__class__.__name__
        ))
        if n_classes in self._allowed_number_of_classes:
            self.n_classes = n_classes
        else:
            raise ValueError(
                'Unsupported number of classes in '
                + self.__class__.__name__ + '!'
            )
        if db_base_name[-3:] == '.db':
            self.db_name = db_base_name
        else:
            self.db_name = db_base_name + '.db'
        self._configure_db()

    def write_data(self, eventid, pred, probs):
        return self._filldb(eventid, pred, probs)

    def read_data(self):
        """ test reader - careful calling this on anything but tiny dbs! """
        s = select([self.table])
        rp = self.connection.execute(s)
        results = rp.fetchall()
        return results

    def close(self):
        gzfile = self.db_name + '.gz'
        with open(self.db_name, 'rb') as f_in, gzip.open(gzfile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        if os.path.isfile(gzfile) and (os.stat(gzfile).st_size > 0):
            os.remove(self.db_name)
        else:
            raise IOError('Compressed file not produced!')

    def _setup_prediction_table(self):
        self.table = Table('zsegment_prediction', self.metadata,
                           Column('id', Integer(), primary_key=True),
                           Column('run', Integer()),
                           Column('subrun', Integer()),
                           Column('gate', Integer()),
                           Column('phys_evt', Integer()),
                           Column('segment', Integer()),
                           UniqueConstraint(
                               'run', 'subrun', 'gate', 'phys_evt'
                           ))
        for i in range(self.n_classes):
            name = 'prob%03d' % i
            col = Column(name, Float())
            self.table.append_column(col)

    def _configure_db(self):
        if os.path.isfile(self.db_name):
            LOGGER.info('found existing record file {}, removing'.format(
                self.db_name
            ))
            os.remove(self.db_name)
        LOGGER.info('using record file {}'.format(self.db_name))
        db = 'sqlite:///' + self.db_name
        self.metadata = MetaData()
        self.engine = create_engine(db)
        self.connection = self.engine.connect()
        self._setup_prediction_table()
        self.metadata.create_all(self.engine)

    def _filldb(self, eventid, pred, probs):
        """
        expect pred to have shape (batch, prediction) and probs to have
        shape (batch?, ?, probability)
        """
        result = None
        run, sub, gate, pevt = decode_eventid(eventid)
        if len(probs) == 11:
            ins = self.table.insert().values(
                run=run,
                subrun=sub,
                gate=gate,
                phys_evt=pevt,
                segment=pred,
                prob000=probs[0],
                prob001=probs[1],
                prob002=probs[2],
                prob003=probs[3],
                prob004=probs[4],
                prob005=probs[5],
                prob006=probs[6],
                prob007=probs[7],
                prob008=probs[8],
                prob009=probs[9],
                prob010=probs[10]
            )
        elif len(probs) == 67:
            ins = self.table.insert().values(
                run=run,
                subrun=sub,
                gate=gate,
                phys_evt=pevt,
                segment=pred,
                prob000=probs[0],
                prob001=probs[1],
                prob002=probs[2],
                prob003=probs[3],
                prob004=probs[4],
                prob005=probs[5],
                prob006=probs[6],
                prob007=probs[7],
                prob008=probs[8],
                prob009=probs[9],
                prob010=probs[10],
                prob011=probs[11],
                prob012=probs[12],
                prob013=probs[13],
                prob014=probs[14],
                prob015=probs[15],
                prob016=probs[16],
                prob017=probs[17],
                prob018=probs[18],
                prob019=probs[19],
                prob020=probs[20],
                prob021=probs[21],
                prob022=probs[22],
                prob023=probs[23],
                prob024=probs[24],
                prob025=probs[25],
                prob026=probs[26],
                prob027=probs[27],
                prob028=probs[28],
                prob029=probs[29],
                prob030=probs[30],
                prob031=probs[31],
                prob032=probs[32],
                prob033=probs[33],
                prob034=probs[34],
                prob035=probs[35],
                prob036=probs[36],
                prob037=probs[37],
                prob038=probs[38],
                prob039=probs[39],
                prob040=probs[40],
                prob041=probs[41],
                prob042=probs[42],
                prob043=probs[43],
                prob044=probs[44],
                prob045=probs[45],
                prob046=probs[46],
                prob047=probs[47],
                prob048=probs[48],
                prob049=probs[49],
                prob050=probs[50],
                prob051=probs[51],
                prob052=probs[52],
                prob053=probs[53],
                prob054=probs[54],
                prob055=probs[55],
                prob056=probs[56],
                prob057=probs[57],
                prob058=probs[58],
                prob059=probs[59],
                prob060=probs[60],
                prob061=probs[61],
                prob062=probs[62],
                prob063=probs[63],
                prob064=probs[64],
                prob065=probs[65],
                prob066=probs[66]
            )
        elif len(probs) == 173:
            ins = self.table.insert().values(
                run=run,
                subrun=sub,
                gate=gate,
                phys_evt=pevt,
                segment=pred,
                prob000=probs[0],
                prob001=probs[1],
                prob002=probs[2],
                prob003=probs[3],
                prob004=probs[4],
                prob005=probs[5],
                prob006=probs[6],
                prob007=probs[7],
                prob008=probs[8],
                prob009=probs[9],
                prob010=probs[10],
                prob011=probs[11],
                prob012=probs[12],
                prob013=probs[13],
                prob014=probs[14],
                prob015=probs[15],
                prob016=probs[16],
                prob017=probs[17],
                prob018=probs[18],
                prob019=probs[19],
                prob020=probs[20],
                prob021=probs[21],
                prob022=probs[22],
                prob023=probs[23],
                prob024=probs[24],
                prob025=probs[25],
                prob026=probs[26],
                prob027=probs[27],
                prob028=probs[28],
                prob029=probs[29],
                prob030=probs[30],
                prob031=probs[31],
                prob032=probs[32],
                prob033=probs[33],
                prob034=probs[34],
                prob035=probs[35],
                prob036=probs[36],
                prob037=probs[37],
                prob038=probs[38],
                prob039=probs[39],
                prob040=probs[40],
                prob041=probs[41],
                prob042=probs[42],
                prob043=probs[43],
                prob044=probs[44],
                prob045=probs[45],
                prob046=probs[46],
                prob047=probs[47],
                prob048=probs[48],
                prob049=probs[49],
                prob050=probs[50],
                prob051=probs[51],
                prob052=probs[52],
                prob053=probs[53],
                prob054=probs[54],
                prob055=probs[55],
                prob056=probs[56],
                prob057=probs[57],
                prob058=probs[58],
                prob059=probs[59],
                prob060=probs[60],
                prob061=probs[61],
                prob062=probs[62],
                prob063=probs[63],
                prob064=probs[64],
                prob065=probs[65],
                prob066=probs[66],
                prob067=probs[67],
                prob068=probs[68],
                prob069=probs[69],
                prob070=probs[70],
                prob071=probs[71],
                prob072=probs[72],
                prob073=probs[73],
                prob074=probs[74],
                prob075=probs[75],
                prob076=probs[76],
                prob077=probs[77],
                prob078=probs[78],
                prob079=probs[79],
                prob080=probs[80],
                prob081=probs[81],
                prob082=probs[82],
                prob083=probs[83],
                prob084=probs[84],
                prob085=probs[85],
                prob086=probs[86],
                prob087=probs[87],
                prob088=probs[88],
                prob089=probs[89],
                prob090=probs[90],
                prob091=probs[91],
                prob092=probs[92],
                prob093=probs[93],
                prob094=probs[94],
                prob095=probs[95],
                prob096=probs[96],
                prob097=probs[97],
                prob098=probs[98],
                prob099=probs[99],
                prob100=probs[100],
                prob101=probs[101],
                prob102=probs[102],
                prob103=probs[103],
                prob104=probs[104],
                prob105=probs[105],
                prob106=probs[106],
                prob107=probs[107],
                prob108=probs[108],
                prob109=probs[109],
                prob110=probs[110],
                prob111=probs[111],
                prob112=probs[112],
                prob113=probs[113],
                prob114=probs[114],
                prob115=probs[115],
                prob116=probs[116],
                prob117=probs[117],
                prob118=probs[118],
                prob119=probs[119],
                prob120=probs[120],
                prob121=probs[121],
                prob122=probs[122],
                prob123=probs[123],
                prob124=probs[124],
                prob125=probs[125],
                prob126=probs[126],
                prob127=probs[127],
                prob128=probs[128],
                prob129=probs[129],
                prob130=probs[130],
                prob131=probs[131],
                prob132=probs[132],
                prob133=probs[133],
                prob134=probs[134],
                prob135=probs[135],
                prob136=probs[136],
                prob137=probs[137],
                prob138=probs[138],
                prob139=probs[139],
                prob140=probs[140],
                prob141=probs[141],
                prob142=probs[142],
                prob143=probs[143],
                prob144=probs[144],
                prob145=probs[145],
                prob146=probs[146],
                prob147=probs[147],
                prob148=probs[148],
                prob149=probs[149],
                prob150=probs[150],
                prob151=probs[151],
                prob152=probs[152],
                prob153=probs[153],
                prob154=probs[154],
                prob155=probs[155],
                prob156=probs[156],
                prob157=probs[157],
                prob158=probs[158],
                prob159=probs[159],
                prob160=probs[160],
                prob161=probs[161],
                prob162=probs[162],
                prob163=probs[163],
                prob164=probs[164],
                prob165=probs[165],
                prob166=probs[166],
                prob167=probs[167],
                prob168=probs[168],
                prob169=probs[169],
                prob170=probs[170],
                prob171=probs[171],
                prob172=probs[172]
            )
        else:
            msg = 'Impossible number of outputs for db in filldb!'
            LOGGER.error(msg)
            raise Exception(msg)
        try:
            result = self.connection.execute(ins)
        except:
            import sys
            e = sys.exc_info()[0]
            msg = 'db error: {}'.format(e)
            LOGGER.error(msg)
        return result
