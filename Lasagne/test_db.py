#!/usr/bin/env python

import predictiondb
from sqlalchemy import MetaData
from sqlalchemy import select


def test1():
    metadata = MetaData()
    eng = predictiondb.get_engine('prediction')
    con = predictiondb.get_connection(eng)
    tbl = predictiondb.get_active_table(metadata, eng)
    ins = tbl.insert().values(
        run=1,
        subrun=1,
        gate=1,
        phys_evt=1,
        segment=0,
        prob00=0.95,
        prob01=0.05,
        prob02=0.00,
        prob03=0.00,
        prob04=0.00,
        prob05=0.00,
        prob06=0.00,
        prob07=0.00,
        prob08=0.00,
        prob09=0.00,
        prob10=0.00)
    result = con.execute(ins)
    return result


def test2():
    metadata = MetaData()
    eng = predictiondb.get_engine('prediction')
    con = predictiondb.get_connection(eng)
    tbl = predictiondb.get_active_table(metadata, eng)
    s = select([tbl])
    rp = con.execute(s)
    results = rp.fetchall()
    print(results)
