import numpy as np
import h5py, os, re
import gmtpy


def hdf2xyz(fname, field):
    hf0 = h5py.File(fname, 'r')
    attribute = hf0.keys()
    print attribute[0]
    if attribute[0] == field[0]:
        z = np.array(hf0[field[0]])
        y = np.array(hf0[field[1]])
        x = np.array(hf0[field[2]])
        hf0.close()
        xyz_1 = np.array([x, y, z]).T
        print xyz_1.shape
        xyz_file = xyz_1
        return xyz_file
    else:
        print 'there is no attribute %s' % field[0]
        return -1


h5_dir = '/media/Seagate Expansion Drive/Data_Xy/Cloudh5/radar0616/'
for h5_file in os.listdir(h5_dir):
    h5_name = h5_dir + h5_file
    print h5_name
    h5_xyz = hdf2xyz(h5_name, ['cell_sigma0_vv_aft', 'sig_cell_lat', 'sig_cell_lon'])
    if np.any(h5_xyz != -1):
        save_name = h5_file[0:-3] + '_grid.txt'
        print(save_name)
        np.savetxt(save_name, h5_xyz, fmt='%8.4f')
