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
# 300k events...
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
# 300k events...
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
# 300k events...
arr_all_minerva1mc_part1 = [
    [ 705,272,437,111,17,14,183,290,230,5,48,],
    [ 186,8921,1241,270,38,40,1117,686,526,14,123,],
    [ 10,70,10785,554,56,49,960,2191,912,6,195,],
    [ 2,10,69,9686,85,69,35,899,2102,17,241,],
    [ 0,0,6,28,7096,433,1,9,840,379,732,],
    [ 0,0,1,12,345,9800,2,2,118,573,1732,],
    [ 7,167,856,171,18,18,3396,433,292,5,60,],
    [ 1,6,209,544,38,26,29,4164,463,3,95,],
    [ 1,0,15,267,673,224,10,32,13574,56,709,],
    [ 0,0,0,0,157,279,0,0,24,701,143,],
    [ 4,2,4,15,79,606,7,9,74,22,204728,],
]
# 600k...
arr_all_minerva1mc_part1 = [
    [ 1412,542,885,222,35,28,403,556,460,9,99,],
    [ 392,17744,2471,525,77,85,2130,1345,1036,18,233,],
    [ 19,144,21797,1104,123,113,1925,4276,1767,18,434,],
    [ 6,19,136,19225,181,142,73,1764,4144,34,504,],
    [ 0,0,10,43,14259,830,3,16,1637,736,1433,],
    [ 0,0,5,18,698,19288,7,7,248,1140,3494,],
    [ 13,352,1691,331,42,34,6858,834,600,13,136,],
    [ 3,22,430,1183,68,45,58,8246,951,11,189,],
    [ 1,2,42,508,1299,469,16,61,27330,105,1464,],
    [ 0,0,0,2,324,595,0,0,46,1393,301,],
    [ 6,4,13,26,144,1168,12,15,156,35,409828,],
]
arr_all_minerva1mc_part1 = reorder_matrix(np.array(arr_all_minerva1mc_part1,
                                                   dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160630_10200-10249.txt
# 300k events...
arr_pass_minerva1mc_part1 = [
    [ 105,151,250,50,10,5,101,167,124,3,16,],
    [ 57,5690,710,154,19,27,633,398,272,6,51,],
    [ 3,41,6921,319,26,26,567,1377,497,3,83,],
    [ 1,7,40,6261,42,39,22,562,1215,6,114,],
    [ 0,0,2,17,4635,258,1,6,527,224,381,],
    [ 0,0,0,5,212,6300,1,1,72,346,955,],
    [ 2,100,513,90,10,11,2168,254,157,2,26,],
    [ 1,1,120,336,17,16,18,2696,253,1,34,],
    [ 0,0,10,158,401,116,5,19,8447,34,351,],
    [ 0,0,0,0,97,159,0,0,19,465,63,],
    [ 0,1,3,12,53,386,6,4,46,12,113723,],
]
# 600k...
arr_pass_minerva1mc_part1 = [
    [ 206,293,503,108,15,12,221,322,246,5,33,],
    [ 127,11351,1452,306,32,49,1205,802,566,6,102,],
    [ 3,84,13939,653,57,66,1157,2692,993,11,177,],
    [ 1,14,79,12478,95,82,44,1118,2368,13,233,],
    [ 0,0,5,24,9271,498,3,8,1014,448,757,],
    [ 0,0,1,8,444,12402,4,3,159,708,1927,],
    [ 3,217,1012,173,17,22,4291,490,319,6,60,],
    [ 2,9,266,704,34,27,37,5293,513,5,61,],
    [ 0,1,26,307,761,245,8,39,17000,58,719,],
    [ 0,0,0,2,212,354,0,0,34,908,140,],
    [ 0,2,8,20,89,764,11,6,93,24,227420,],
]
arr_pass_minerva1mc_part1 = reorder_matrix(np.array(arr_pass_minerva1mc_part1,
                                                    dtype=np.float))

arr_all_memc = arr_all_me1Bmc + arr_all_me1Amc
arr_pass_memc = arr_pass_me1Bmc + arr_pass_me1Amc

arr_all_minerva1mc = arr_all_minerva1mc_part1
arr_pass_minerva1mc = arr_pass_minerva1mc_part1

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
