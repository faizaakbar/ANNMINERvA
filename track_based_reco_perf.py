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

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160715-16_112200-112204.txt
arr_all_me1Bmc_part1 = np.zeros((11, 11), dtype=np.float)
# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160715-16_112200-112204.txt
arr_pass_me1Bmc_part1 = np.zeros((11, 11), dtype=np.float)

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160719_112205.txt
arr_all_me1Bmc_part2 = [
    [ 1885,790,159,121,17,15,299,170,178,5,83,],
    [ 1064,15673,370,221,23,21,1461,346,310,3,72,],
    [ 14,139,16137,549,48,31,1641,1635,702,8,154,],
    [ 2,20,136,13247,61,56,41,1233,2497,16,188,],
    [ 0,1,9,21,9706,520,1,13,1153,488,694,],
    [ 0,2,1,12,468,13156,2,7,151,754,1851,],
    [ 16,296,522,149,8,15,5589,219,208,3,35,],
    [ 1,21,301,713,19,13,25,5720,460,3,54,],
    [ 1,5,24,367,845,258,10,35,18222,64,610,],
    [ 0,0,0,3,203,470,0,1,32,1000,140,],
    [ 0,8,13,13,89,763,11,13,114,31,268064,],
]
arr_all_me1Bmc_part2 = reorder_matrix(np.array(arr_all_me1Bmc_part2,
                                               dtype=np.float))

# /minerva/data/users/perdue/RecoTracks/files/nukecc_20160719_112205.txt
arr_pass_me1Bmc_part2 = [
    [ 722,352,2,5,0,0,15,3,0,0,0,],
    [ 576,10681,18,3,0,0,647,3,1,0,0,],
    [ 6,86,10660,18,1,0,987,635,9,0,0,],
    [ 1,14,79,9102,0,3,26,794,851,0,1,],
    [ 0,1,7,13,6193,181,1,8,664,253,13,],
    [ 0,1,0,6,259,8690,1,3,88,470,456,],
    [ 6,205,272,3,0,1,3852,18,2,0,0,],
    [ 0,13,201,346,0,0,18,3925,28,0,0,],
    [ 1,5,18,258,362,20,4,20,12182,5,4,],
    [ 0,0,0,2,124,284,0,0,20,680,6,],
    [ 0,4,7,8,47,496,5,10,66,15,140430,],
]
arr_pass_me1Bmc_part2 = reorder_matrix(np.array(arr_pass_me1Bmc_part2,
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

arr_all_me1Bmc = arr_all_me1Bmc_part1 + arr_all_me1Bmc_part2
arr_pass_me1Bmc = arr_pass_me1Bmc_part1 + arr_pass_me1Bmc_part2

arr_all_minerva1mc = arr_all_minerva1mc_part1 + arr_all_minerva1mc_part2
arr_pass_minerva1mc = arr_pass_minerva1mc_part1 + arr_pass_minerva1mc_part2

# We row normalize (divide by `axis=1` to get the _purity_ - it is saying 
# "okay, I reconstructed an event in target 1, what fraction of the events 
# really come from target 1? what fraction came from other z's?, etc."

pur_all_me1Bmc = np.zeros_like(arr_all_me1Bmc)
for i in range(np.shape(arr_all_me1Bmc)[0]):
    pur_all_me1Bmc[i, :] = arr_all_me1Bmc[i, :] / arr_all_me1Bmc.sum(axis=1)[i]

pur_all_minerva1mc = np.zeros_like(arr_all_minerva1mc)
for i in range(np.shape(arr_all_minerva1mc)[0]):
    pur_all_minerva1mc[i, :] = arr_all_minerva1mc[i, :] / arr_all_minerva1mc.sum(axis=1)[i]

pur_pass_me1Bmc = np.zeros_like(arr_pass_me1Bmc)
for i in range(np.shape(arr_pass_me1Bmc)[0]):
    pur_pass_me1Bmc[i, :] = arr_pass_me1Bmc[i, :] / arr_pass_me1Bmc.sum(axis=1)[i]

pur_pass_minerva1mc = np.zeros_like(arr_pass_minerva1mc)
for i in range(np.shape(arr_pass_minerva1mc)[0]):
    pur_pass_minerva1mc[i, :] = arr_pass_minerva1mc[i, :] / arr_pass_minerva1mc.sum(axis=1)[i]

# We column normalize (divide by `axis=0` to get the _efficiency_ - it is
# saying "okay, I have an event really coming from target 1 (reading the
# "y-axis"), was it reconstructed in target 1?, etc. (reading along the x for
# a given y)"

eff_all_me1Bmc = np.zeros_like(arr_all_me1Bmc)
for i in range(np.shape(arr_all_me1Bmc)[0]):
    eff_all_me1Bmc[:, i] = arr_all_me1Bmc[:, i] / arr_all_me1Bmc.sum(axis=0)[i]

eff_all_minerva1mc = np.zeros_like(arr_all_minerva1mc)
for i in range(np.shape(arr_all_minerva1mc)[0]):
    eff_all_minerva1mc[:, i] = arr_all_minerva1mc[:, i] / arr_all_minerva1mc.sum(axis=0)[i]

eff_pass_me1Bmc = np.zeros_like(arr_pass_me1Bmc)
for i in range(np.shape(arr_pass_me1Bmc)[0]):
    eff_pass_me1Bmc[:, i] = arr_pass_me1Bmc[:, i] / arr_pass_me1Bmc.sum(axis=0)[i]

eff_pass_minerva1mc = np.zeros_like(arr_pass_minerva1mc)
for i in range(np.shape(arr_pass_minerva1mc)[0]):
    eff_pass_minerva1mc[:, i] = arr_pass_minerva1mc[:, i] / arr_pass_minerva1mc.sum(axis=0)[i]
