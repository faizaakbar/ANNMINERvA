#!/bin/bash
BASE_DIR=/Users/perdue/Documents/MINERvA/AI/hdf5/201801
HDF5LIST="
hadmultkineimgs_127x94_me1Amc.hdf5
hadmultkineimgs_127x94_me1Bmc.hdf5
hadmultkineimgs_127x94_me1Cmc.hdf5
hadmultkineimgs_127x94_me1Dmc.hdf5
hadmultkineimgs_127x94_me1Emc.hdf5
hadmultkineimgs_127x94_me1Gmc.hdf5
hadmultkineimgs_127x94_me1Lmc.hdf5
hadmultkineimgs_127x94_me1Mmc.hdf5
hadmultkineimgs_127x94_me1Nmc.hdf5
hadmultkineimgs_127x94_me1Omc.hdf5
hadmultkineimgs_127x94_me1Pmc.hdf5
mnvimgs_127x94_me1Adata.hdf5
mnvimgs_127x94_me1Bdata.hdf5
mnvimgs_127x94_me1Cdata.hdf5
mnvimgs_127x94_me1Ddata.hdf5
mnvimgs_127x94_me1Edata.hdf5
mnvimgs_127x94_me1Fdata.hdf5
mnvimgs_127x94_me1Gdata.hdf5
mnvimgs_127x94_me1Ldata.hdf5
mnvimgs_127x94_me1Mdata.hdf5
mnvimgs_127x94_me1Ndata.hdf5
mnvimgs_127x94_me1Odata.hdf5
mnvimgs_127x94_me1Pdata.hdf5
"

for file in $HDF5LIST
do
  python write_evt_numbers_set_to_file.py $BASE_DIR/$file
  echo "finished $BASE_DIR/$file ..."
done
