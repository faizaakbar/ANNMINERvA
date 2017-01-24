#!/usr/bin/env python

from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float
from sqlalchemy import UniqueConstraint


def get_engine(dbname):
    """
    simple method to wrap the db name with the sqlite designator and create
    the engine
    """
    db = 'sqlite:///' + dbname + '.db'
    engine = create_engine(db)
    return engine


def get_connection(engine):
    connection = engine.connect()
    return connection


def get_11segment_prediction_table(metadata):
    table = Table('zsegment_prediction', metadata,
                  Column('id', Integer(), primary_key=True),
                  Column('run', Integer()),
                  Column('subrun', Integer()),
                  Column('gate', Integer()),
                  Column('phys_evt', Integer()),
                  Column('segment', Integer()),
                  Column('prob00', Float()),
                  Column('prob01', Float()),
                  Column('prob02', Float()),
                  Column('prob03', Float()),
                  Column('prob04', Float()),
                  Column('prob05', Float()),
                  Column('prob06', Float()),
                  Column('prob07', Float()),
                  Column('prob08', Float()),
                  Column('prob09', Float()),
                  Column('prob10', Float()),
                  UniqueConstraint('run', 'subrun', 'gate', 'phys_evt')
                  )
    return table


def get_67segment_prediction_table(metadata):
    table = Table('zsegment_prediction', metadata,
                  Column('id', Integer(), primary_key=True),
                  Column('run', Integer()),
                  Column('subrun', Integer()),
                  Column('gate', Integer()),
                  Column('phys_evt', Integer()),
                  Column('segment', Integer()),
                  UniqueConstraint('run', 'subrun', 'gate', 'phys_evt')
                  )
    for i in range(67):
        name = 'prob%02d' % i
        col = Column(name, Float())
        table.append_column(col)
    return table


def get_active_table(metadata, engine,
                     get_table_fn=get_11segment_prediction_table):
    table = get_table_fn(metadata)
    metadata.create_all(engine)
    return table
