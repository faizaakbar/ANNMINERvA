#!/bin/bash

python make_hdf5_fuelfiles.py \
  -b minosmatch_nukecczdefs_fullz_127x94_minerva1mc \
  -o minosmatch_nukecczdefs_genallz_pcodecap66_127x50x25_xuv_minerva1mc.hdf5 \
  -x \
  --trim_column_down_x 50 \
  --trim_column_down_uv 25 \
  --cap_planecode 66
