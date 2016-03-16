#!/usr/bin/env python
"""
"""
from __future__ import print_function
import operator


def decode(pcode):
    """
    modules will be returned with "real" values (negative values are allowed)
    """
    if pcode == 0xFFFF:
        return (-999, -999, -999)
    target = pcode & 0x07
    if target != 0:
        return (0, 0, target)
    plane = (pcode >> 3) & 0x03
    module = ((pcode >> 5) & 0xFF) - 5
    return (module, plane, target)


def encode(module, plane, target):
    """
    here, modules should be input with real values; they will be shifted to
    be non-negative in the encoding
    """
    if module == -999 or plane == -999 or target == -999:
        return 0

    return target + (plane << 3) + ((module + 5) << 5)


def build_planecode_dict():
    planecodes = {}
    planecodes[-1] = (-999, -999, -999)
    # work in 0->119 module numbering scheme (real is -5->114)
    for i in range(-5, 115):

        # handle "target modules first"
        if i == -1:
            # module -1 is target 1
            planecodes[1] = (-1, 0, 1)
            continue
        if i == 4:
            # module 4 is target 2
            planecodes[2] = (4, 0, 2)
            continue
        if i == 9 or i == 10:
            # modules 9 and 10 are target 3
            planecodes[3] = (9, 0, 3)
            continue
        if i == 19:
            # module 19 is target 4
            planecodes[4] = (19, 0, 4)
            continue
        if i == 22:
            # module 22 is target 5
            planecodes[5] = (22, 0, 5)
            continue

        if i < 95:
            # targets, tracker and ecal modules have 2 planes
            for j in [1, 2]:
                encoded = encode(i, j, 0)
                planecodes[encoded] = (i, j, 0)
        else:
            # hcal only has plane 2
            encoded = encode(i, 2, 0)
            planecodes[encoded] = (i, 2, 0)

    return planecodes


def build_indexed_codes():
    """
    this dictionary takes the "plane-id-code" produced from the skimmer
    and changes it into the "one-hot index" value of a vector with length
    equal to the total number of possible codes (214)
    """
    planecodes = build_planecode_dict()
    recoded_codes = {}
    for k, v in planecodes.items():
        newcode = encode(v[0], v[1], v[2])
        recoded_codes[k] = newcode

    sorted_codes = sorted(recoded_codes.items(),
                          key=operator.itemgetter(1))
    indexed_codes = {}
    for i, t in enumerate(sorted_codes):
        indexed_codes[t[0]] = i

    return indexed_codes


def build_reversed_indexed_codes():
    """
    use this dictionary to map an index in the classifier output to a position
    in the detector
    """
    icodes = build_indexed_codes()
    pcodes = build_planecode_dict()
    rcodes = dict(zip(icodes.values(), icodes.keys()))
    index_to_detector = dict()
    for k, v in rcodes.items():
        index_to_detector[k] = pcodes[v]
    return index_to_detector


def get_views():
    import re
    views_by_plane = """
    UXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVX
    UXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVX
    UXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXUXVXXUXVXUXVXUXVXUXVXUXV
    """
    views_by_plane = re.sub(r"\s+", "", views_by_plane)
    views_by_plane_dict = dict()
    for i, c in enumerate(views_by_plane):
        views_by_plane_dict[i] = c

    return views_by_plane_dict
