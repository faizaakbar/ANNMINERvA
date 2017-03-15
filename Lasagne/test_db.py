#!/usr/bin/env python

import predictiondb
from sqlalchemy import MetaData
from sqlalchemy import select


def test1(dbname):
    metadata = MetaData()
    eng = predictiondb.get_engine(dbname)
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


def test2(dbname):
    metadata = MetaData()
    eng = predictiondb.get_engine(dbname)
    con = predictiondb.get_connection(eng)
    tbl = predictiondb.get_active_table(metadata, eng)
    s = select([tbl])
    rp = con.execute(s)
    results = rp.fetchall()
    print(results)


def test_make_file_of_unique_runs_and_subs(dbname):
    metadata = MetaData()
    eng = predictiondb.get_engine(dbname)
    con = predictiondb.get_connection(eng)
    tbl = predictiondb.get_active_table(metadata, eng)
    s = select([tbl.c.run, tbl.c.subrun])
    rp = con.execute(s)
    results = rp.fetchall()
    runs_subs = set()
    _ = [runs_subs.add(tuple(i)) for i in results]
    runs_subs = list(runs_subs)
    sorted_runs_subs = sorted(runs_subs, key=lambda x: x[1])
    f = open('runs_subs.txt', 'w')
    for i in sorted_runs_subs:
        f.write(str(i) + '\n')
    f.close()


def test3():
    metadata = MetaData()
    eng = predictiondb.get_engine('prediction')
    con = predictiondb.get_connection(eng)
    tbl = predictiondb.get_active_table(
        metadata,
        eng,
        predictiondb.get_67segment_prediction_table
    )
    ins = tbl.insert().values(
        run=1,
        subrun=1,
        gate=1,
        phys_evt=1,
        segment=0,
        prob00=0.80,
        prob01=0.05,
        prob02=0.00,
        prob03=0.00,
        prob04=0.00,
        prob05=0.01,
        prob06=0.00,
        prob07=0.00,
        prob08=0.00,
        prob09=0.00,
        prob10=0.01,
        prob11=0.01,
        prob12=0.00,
        prob13=0.00,
        prob14=0.00,
        prob15=0.01,
        prob16=0.01,
        prob17=0.00,
        prob18=0.00,
        prob19=0.00,
        prob20=0.01,
        prob21=0.00,
        prob22=0.00,
        prob23=0.00,
        prob24=0.00,
        prob25=0.01,
        prob26=0.00,
        prob27=0.00,
        prob28=0.00,
        prob29=0.00,
        prob30=0.01,
        prob31=0.00,
        prob32=0.00,
        prob33=0.00,
        prob34=0.00,
        prob35=0.01,
        prob36=0.00,
        prob37=0.00,
        prob38=0.00,
        prob39=0.00,
        prob40=0.01,
        prob41=0.00,
        prob42=0.00,
        prob43=0.00,
        prob44=0.00,
        prob45=0.01,
        prob46=0.00,
        prob47=0.00,
        prob48=0.00,
        prob49=0.00,
        prob50=0.01,
        prob51=0.00,
        prob52=0.00,
        prob53=0.00,
        prob54=0.00,
        prob55=0.01,
        prob56=0.00,
        prob57=0.00,
        prob58=0.00,
        prob59=0.00,
        prob60=0.01,
        prob61=0.00,
        prob62=0.00,
        prob63=0.00,
        prob64=0.00,
        prob65=0.01,
        prob66=0.00
    )
    result = con.execute(ins)
    return result


if __name__ == '__main__':
    test1('testdb')
    test2('testdb')
    test_make_file_of_unique_runs_and_subs('testdb')
