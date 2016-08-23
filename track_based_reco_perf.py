#!/usr/bin/env python
import numpy as np


def reorder_matrix(ndarr):
    if ndarr.shape != (11, 11):
        return np.array([0], dtype=np.float)
    mid_arr = np.zeros((11, 11), dtype=np.float)
    new_arr = np.zeros((11, 11), dtype=np.float)
    mid_arr[0, :] = ndarr[0, :] 
    mid_arr[1, :] = ndarr[1, :] 
    mid_arr[2, :] = ndarr[6, :] 
    mid_arr[3, :] = ndarr[2, :] 
    mid_arr[4, :] = ndarr[7, :] 
    mid_arr[5, :] = ndarr[3, :] 
    mid_arr[6, :] = ndarr[8, :] 
    mid_arr[7, :] = ndarr[4, :] 
    mid_arr[8, :] = ndarr[9, :] 
    mid_arr[9, :] = ndarr[5, :] 
    mid_arr[10, :] = ndarr[10, :] 
    new_arr[:, 0] = mid_arr[:, 0]
    new_arr[:, 1] = mid_arr[:, 1]
    new_arr[:, 2] = mid_arr[:, 6]
    new_arr[:, 3] = mid_arr[:, 2]
    new_arr[:, 4] = mid_arr[:, 7]
    new_arr[:, 5] = mid_arr[:, 3]
    new_arr[:, 6] = mid_arr[:, 8]
    new_arr[:, 7] = mid_arr[:, 4]
    new_arr[:, 8] = mid_arr[:, 9]
    new_arr[:, 9] = mid_arr[:, 5]
    new_arr[:, 10] = mid_arr[:, 10]
    return new_arr

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160321_112200-05_127x94.txt
# 150k events...
arr_all_me1Bmc = [
    [ 659,292,61,48,6,3,127,54,59,2,32,],
    [ 402,5853,145,90,16,10,562,131,126,1,25,],
    [ 5,58,6058,232,21,11,621,689,288,3,51,],
    [ 1,4,37,5066,18,17,11,457,933,2,71,],
    [ 0,1,2,7,3666,171,3,1,422,210,259,],
    [ 0,0,0,7,158,4988,1,3,56,324,706,],
    [ 6,141,250,63,3,6,2260,83,80,1,13,],
    [ 0,1,98,253,10,1,18,2158,148,2,32,],
    [ 0,4,16,135,346,88,5,18,6961,33,228,],
    [ 0,0,0,3,91,163,0,0,9,394,53,],
    [ 1,5,3,4,28,288,7,5,38,11,101095,],
]
# 300k events...
arr_all_me1Bmc = [
    [ 1383,603,119,96,12,8,236,118,134,5,66,],
    [ 756,11680,262,181,29,19,1089,260,252,4,60,],
    [ 16,110,12216,431,30,23,1280,1359,579,5,98,],
    [ 5,10,84,10124,43,39,22,940,1835,2,138,],
    [ 1,2,4,24,7366,376,4,4,872,392,503,],
    [ 0,1,0,12,344,9916,1,8,106,624,1400,],
    [ 14,240,468,111,8,8,4331,174,156,1,28,],
    [ 1,7,198,488,23,11,30,4358,315,3,69,],
    [ 1,11,25,254,671,178,9,38,13902,61,434,],
    [ 0,0,1,5,183,339,0,0,21,759,102,],
    [ 4,8,3,11,61,575,10,8,76,22,202467,],
]
arr_all_me1Bmc = reorder_matrix(np.array(arr_all_me1Bmc,
                                         dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160321_112200-05_127x94.txt
# 150k events...
arr_pass_me1Bmc = [
    [ 270,194,31,28,3,2,60,27,28,1,12,],
    [ 248,4172,91,57,8,6,366,85,75,0,8,],
    [ 1,39,4398,154,12,7,445,434,175,2,19,],
    [ 0,0,30,3666,8,11,7,331,605,1,33,],
    [ 0,1,2,6,2602,127,3,1,307,148,134,],
    [ 0,0,0,4,118,3588,1,3,33,237,439,],
    [ 2,98,159,41,3,4,1609,45,52,0,3,],
    [ 0,0,73,175,5,1,12,1536,85,1,15,],
    [ 0,4,10,96,222,55,5,15,4903,15,105,],
    [ 0,0,0,2,70,125,0,0,3,288,32,],
    [ 1,2,1,1,16,214,7,4,23,7,62693,],
]
# 300k events...
arr_pass_me1Bmc = [
    [ 549,403,63,57,7,5,122,56,59,2,21,],
    [ 457,8355,161,119,15,10,716,160,138,1,14,],
    [ 10,83,8856,294,16,14,901,885,339,3,41,],
    [ 3,5,62,7385,24,25,17,679,1191,1,62,],
    [ 0,2,3,19,5244,279,4,3,609,269,266,],
    [ 0,1,0,7,253,7194,1,5,64,453,853,],
    [ 5,171,298,67,5,4,3118,105,95,0,11,],
    [ 0,4,146,336,15,5,23,3126,180,2,34,],
    [ 0,7,16,187,420,113,9,29,9784,28,220,],
    [ 0,0,0,4,139,255,0,0,13,546,60,],
    [ 3,3,1,4,41,420,9,6,50,15,125653,],
]
arr_pass_me1Bmc = reorder_matrix(np.array(arr_pass_me1Bmc,
                                          dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160406_117200-117209.txt
arr_all_me1Amc = [
    [ 1426,623,121,91,14,10,200,106,152,3,56,],
    [ 791,12100,277,172,18,24,1090,271,261,6,68,],
    [ 15,98,12497,410,47,34,1260,1302,601,0,121,],
    [ 1,14,96,9892,53,42,23,915,1808,8,122,],
    [ 0,1,1,16,7285,384,0,11,893,362,532,],
    [ 0,0,3,11,358,9859,2,2,110,563,1439,],
    [ 14,217,418,92,16,6,4275,176,135,2,35,],
    [ 2,15,201,546,20,13,30,4258,315,3,56,],
    [ 2,6,19,273,643,204,6,22,13984,46,449,],
    [ 0,1,0,3,190,313,0,0,11,712,97,],
    [ 2,8,5,3,49,569,9,18,95,25,202322,],
]
arr_all_me1Amc = reorder_matrix(np.array(arr_all_me1Amc,
                                         dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160406_117200-117209.txt
arr_pass_me1Amc = [
    [ 541,403,56,54,8,8,112,53,66,0,7,],
    [ 462,8691,176,114,8,13,697,155,141,1,21,],
    [ 8,61,9005,259,27,18,875,839,348,0,46,],
    [ 0,12,76,7253,30,28,14,645,1192,3,52,],
    [ 0,0,0,12,5188,251,0,10,644,242,286,],
    [ 0,0,2,8,247,7171,1,2,77,406,864,],
    [ 5,157,273,65,6,2,3062,109,78,1,12,],
    [ 0,10,153,380,4,8,18,2987,180,2,19,],
    [ 2,4,17,205,432,129,5,15,9812,22,208,],
    [ 0,0,0,2,141,216,0,0,8,515,54,],
    [ 1,7,3,2,29,409,5,14,60,17,124900,],
]
arr_pass_me1Amc = reorder_matrix(np.array(arr_pass_me1Amc,
                                          dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160630_10200-10249.txt
arr_all_minerva1mc_part1 = [
    [ 4604,1776,2728,737,98,86,1188,1653,1304,25,273,],
    [ 1241,56209,7885,1712,263,246,6530,4052,3385,68,773,],
    [ 50,522,69039,3581,422,374,6168,13464,5556,64,1416,],
    [ 16,71,451,61259,575,435,201,5619,13340,93,1542,],
    [ 2,1,23,152,44793,2584,13,45,5271,2404,4662,],
    [ 1,3,15,65,2230,60930,14,20,789,3339,10932,],
    [ 58,1139,5275,991,159,126,21523,2518,1782,33,415,],
    [ 8,65,1289,3768,195,157,171,26139,3029,37,638,],
    [ 5,11,108,1610,4174,1426,61,205,86739,346,4685,],
    [ 2,2,3,12,1185,2073,0,0,148,4457,926,],
    [ 21,18,52,74,440,3634,46,56,529,142,1290627,],
]
arr_all_minerva1mc_part1 = reorder_matrix(np.array(arr_all_minerva1mc_part1,
                                                   dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160630_10200-10249.txt
arr_pass_minerva1mc_part1 = [
    [ 684,980,4,2,0,0,92,4,0,0,0,],
    [ 421,36005,50,7,0,2,2565,25,5,0,1,],
    [ 5,295,39576,118,0,2,3229,5470,26,0,2,],
    [ 1,44,243,37242,0,3,115,2999,3724,0,4,],
    [ 0,0,11,73,25384,628,6,16,2592,1086,49,],
    [ 1,2,6,31,1106,35456,7,10,354,1717,1966,],
    [ 14,726,1730,8,0,1,13021,402,3,0,1,],
    [ 2,33,745,1513,0,1,88,15770,134,0,1,],
    [ 1,6,49,905,1410,50,34,114,50547,28,17,],
    [ 0,1,2,7,710,1038,0,0,80,2721,29,],
    [ 0,7,26,43,198,2095,33,29,241,79,590635,],
]
arr_pass_minerva1mc_part1 = reorder_matrix(np.array(arr_pass_minerva1mc_part1,
                                                    dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160701_10200-10249.txt
arr_all_minerva1mc_part2 = [
    [ 711,292,426,110,15,19,184,261,193,2,49,],
    [ 205,9293,1282,311,55,33,1179,674,566,11,133,],
    [ 4,92,11185,574,69,62,1009,2128,939,12,244,],
    [ 2,11,94,9931,87,69,29,874,2060,19,269,],
    [ 1,1,2,21,7339,400,3,7,883,364,736,],
    [ 1,1,4,7,339,9844,2,5,116,528,1747,],
    [ 13,170,866,142,31,21,3555,455,291,5,72,],
    [ 2,7,207,663,41,30,38,4272,491,7,100,],
    [ 2,0,14,301,725,243,16,39,14056,51,776,],
    [ 1,0,0,2,184,344,0,2,31,761,131,],
    [ 2,3,4,10,60,552,9,12,92,27,209572,],
]
arr_all_minerva1mc_part2 = reorder_matrix(np.array(arr_all_minerva1mc_part2,
                                                   dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160701_10200-10249.txt
arr_pass_minerva1mc_part2 = [
    [ 109,159,1,0,0,0,14,1,0,0,0,],
    [ 67,5969,7,1,0,0,456,2,1,0,0,],
    [ 0,51,6412,15,0,0,506,865,6,0,0,],
    [ 0,6,51,6107,1,0,16,455,613,0,0,],
    [ 0,1,2,12,4101,90,3,2,440,156,8,],
    [ 0,0,2,3,164,5816,1,3,53,292,342,],
    [ 2,99,274,2,0,0,2162,69,1,0,0,],
    [ 0,4,117,264,1,0,23,2558,22,0,0,],
    [ 0,0,11,171,227,8,8,25,8152,3,3,],
    [ 0,0,0,1,111,180,0,0,20,452,6,],
    [ 0,3,3,5,36,328,7,7,41,11,95473,],
]
arr_pass_minerva1mc_part2 = reorder_matrix(np.array(arr_pass_minerva1mc_part2,
                                                    dtype=np.float))

arr_all_memc = arr_all_me1Bmc + arr_all_me1Amc
arr_pass_memc = arr_pass_me1Bmc + arr_pass_me1Amc

arr_all_minerva1mc = arr_all_minerva1mc_part1 + arr_all_minerva1mc_part2
arr_pass_minerva1mc = arr_pass_minerva1mc_part1 + arr_pass_minerva1mc_part2

# We row normalize (divide by `axis=1` to get the _purity_ - it is saying 
# "okay, I reconstructed an event in target 1, what fraction of the events 
# really come from target 1? what fraction came from other z's?, etc."

pur_all_memc = np.zeros_like(arr_all_memc)
for i in range(np.shape(arr_all_memc)[0]):
    pur_all_memc[i, :] = arr_all_memc[i, :] / arr_all_memc.sum(axis=1)[i]

pur_all_minerva1mc = np.zeros_like(arr_all_minerva1mc)
for i in range(np.shape(arr_all_minerva1mc)[0]):
    pur_all_minerva1mc[i, :] = arr_all_minerva1mc[i, :] / arr_all_minerva1mc.sum(axis=1)[i]

pur_pass_memc = np.zeros_like(arr_pass_memc)
for i in range(np.shape(arr_pass_memc)[0]):
    pur_pass_memc[i, :] = arr_pass_memc[i, :] / arr_pass_memc.sum(axis=1)[i]

pur_pass_minerva1mc = np.zeros_like(arr_pass_minerva1mc)
for i in range(np.shape(arr_pass_minerva1mc)[0]):
    pur_pass_minerva1mc[i, :] = arr_pass_minerva1mc[i, :] / arr_pass_minerva1mc.sum(axis=1)[i]

# We column normalize (divide by `axis=0` to get the _efficiency_ - it is
# saying "okay, I have an event really coming from target 1 (reading the
# "y-axis"), was it reconstructed in target 1?, etc. (reading along the x for
# a given y)"

eff_all_memc = np.zeros_like(arr_all_memc)
for i in range(np.shape(arr_all_memc)[0]):
    eff_all_memc[:, i] = arr_all_memc[:, i] / arr_all_memc.sum(axis=0)[i]

eff_all_minerva1mc = np.zeros_like(arr_all_minerva1mc)
for i in range(np.shape(arr_all_minerva1mc)[0]):
    eff_all_minerva1mc[:, i] = arr_all_minerva1mc[:, i] / arr_all_minerva1mc.sum(axis=0)[i]

eff_pass_memc = np.zeros_like(arr_pass_memc)
for i in range(np.shape(arr_pass_memc)[0]):
    eff_pass_memc[:, i] = arr_pass_memc[:, i] / arr_pass_memc.sum(axis=0)[i]

eff_pass_minerva1mc = np.zeros_like(arr_pass_minerva1mc)
for i in range(np.shape(arr_pass_minerva1mc)[0]):
    eff_pass_minerva1mc[:, i] = arr_pass_minerva1mc[:, i] / arr_pass_minerva1mc.sum(axis=0)[i]
