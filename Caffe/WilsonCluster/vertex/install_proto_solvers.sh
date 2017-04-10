#!/bin/bash

BASEDIR=/data/perdue/minerva/caffe
cp vertex_epsilon_adv.solver $BASEDIR/solvers
cp vertex_epsilon_adv.prototxt $BASEDIR/proto
cp vertex_epsilon_adv_test.prototxt $BASEDIR/proto
