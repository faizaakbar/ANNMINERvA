#!/usr/bin/env python
"""
Do persistence
"""
import os
import logging

from six.moves import range

from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float
from sqlalchemy import UniqueConstraint
from sqlalchemy import MetaData
from sqlalchemy import select

import mnv_utils

LOGGER = logging.getLogger(__name__)


class MnvCategoricalSQLiteRecorder:
    """
    record segments or planecodes in a sqlite db
    """
    def __init__(self, n_classes, db_base_name):
        self.n_classes = n_classes
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
            name = 'prob%02d' % i
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
        run, sub, gate, pevt = mnv_utils.decode_eventid(eventid)
        if len(probs) == 11:
            ins = self.table.insert().values(
                run=run,
                subrun=sub,
                gate=gate,
                phys_evt=pevt,
                segment=pred,
                prob00=probs[0],
                prob01=probs[1],
                prob02=probs[2],
                prob03=probs[3],
                prob04=probs[4],
                prob05=probs[5],
                prob06=probs[6],
                prob07=probs[7],
                prob08=probs[8],
                prob09=probs[9],
                prob10=probs[10]
            )
        elif len(probs) == 67:
            ins = self.table.insert().values(
                run=run,
                subrun=sub,
                gate=gate,
                phys_evt=pevt,
                segment=pred,
                prob00=probs[0],
                prob01=probs[1],
                prob02=probs[2],
                prob03=probs[3],
                prob04=probs[4],
                prob05=probs[5],
                prob06=probs[6],
                prob07=probs[7],
                prob08=probs[8],
                prob09=probs[9],
                prob10=probs[10],
                prob11=probs[11],
                prob12=probs[12],
                prob13=probs[13],
                prob14=probs[14],
                prob15=probs[15],
                prob16=probs[16],
                prob17=probs[17],
                prob18=probs[18],
                prob19=probs[19],
                prob20=probs[20],
                prob21=probs[21],
                prob22=probs[22],
                prob23=probs[23],
                prob24=probs[24],
                prob25=probs[25],
                prob26=probs[26],
                prob27=probs[27],
                prob28=probs[28],
                prob29=probs[29],
                prob30=probs[30],
                prob31=probs[31],
                prob32=probs[32],
                prob33=probs[33],
                prob34=probs[34],
                prob35=probs[35],
                prob36=probs[36],
                prob37=probs[37],
                prob38=probs[38],
                prob39=probs[39],
                prob40=probs[40],
                prob41=probs[41],
                prob42=probs[42],
                prob43=probs[43],
                prob44=probs[44],
                prob45=probs[45],
                prob46=probs[46],
                prob47=probs[47],
                prob48=probs[48],
                prob49=probs[49],
                prob50=probs[50],
                prob51=probs[51],
                prob52=probs[52],
                prob53=probs[53],
                prob54=probs[54],
                prob55=probs[55],
                prob56=probs[56],
                prob57=probs[57],
                prob58=probs[58],
                prob59=probs[59],
                prob60=probs[60],
                prob61=probs[61],
                prob62=probs[62],
                prob63=probs[63],
                prob64=probs[64],
                prob65=probs[65],
                prob66=probs[66]
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
