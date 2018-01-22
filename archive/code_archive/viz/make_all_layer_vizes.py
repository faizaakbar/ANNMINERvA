#!/usr/bin/env python
"""
Usage:
    python make_all_layer_vizes.py
"""
from __future__ import print_function
import os
import re

from layer_vis import make_vis

files = os.listdir('.')
filepat = re.compile(r"^vis_[0-9]+_(conv|pool)_[0-9]_[0-9]+_[0-9]+.npy$")
files = [f for f in files if re.match(filepat, f)]

eventids = set()
for f in files:
    nums = re.findall(r'[0-9]+', f)
    eventid = nums[-1]
    eventids.add(eventid)

for eventid in eventids:
    make_vis(eventid)
