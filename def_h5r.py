import h5py
import numpy as np
import os, re

def return_ind(filenamez, siten):
# return  the index of site value in the hdf5 files
    hf = h5py.File(filenamez, 'r')  # input 1, file name
    lat_t = siten[1]
    lon_t = siten[2]  # input 2, location
    dset_lon = hf['Global_Projection/cell_lon']
    dset_lat = hf['Global_Projection/cell_lat']
    lon_ind = np.where(np.abs(dset_lon[...] - lon_t) < 0.36)
    lat_ind = np.where(np.abs(dset_lat[...] - lat_t) < 0.36)
    region_ind = np.intersect1d(lon_ind, lat_ind)
    if len(region_ind) > 0:
        print(region_ind)
        print(filenamez)
    return region_ind, filenamez
# then only read this orbit, so the TB value reading is optimized.
# def last_ob()
rootdir = "/media/Seagate Expansion Drive/Data_Xy/CloudMusic/2015.04.29/"
for files in os.listdir(rootdir):
    if files.find('.iso') == -1 and files.find('_A_') != -1:  # A orbit file (eliminate iso file)
        [inds, names] = return_ind(rootdir + files, ['947', 65.12422, -146.73390])

