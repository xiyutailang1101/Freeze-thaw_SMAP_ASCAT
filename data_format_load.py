import numpy as np
import basic_xiyu as bxy
import glob
import h5py


def npz_interest_ascat_pixel(data_path='prepare_files/npz/ascat/ascat_interest_pixel_series_2017.npz',
                             data_path1='prepare_files/npz/smap/smap_all_series_A_2017.npz'):
    npz0 = np.load(data_path)
    t0 = npz0['utc_line_nodes'][0]
    t0_tuple = bxy.time_getlocaltime(t0[t0>0], ref_time=[2000, 1, 1, 0])
    npz1 = np.load(data_path1)
    return t0_tuple, npz1


def npy_ascat_one_station_estimate():
    pp = np.load('prepare_files/npy_ascat_one_station/file0.npy')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                return pp


def other_import():
    h0 = h5py.File('un_used0.h5', 'a')
    match0 = glob.glob('ascat_interest_pixel_series_201*')
    return h0, match0


if __name__ == "__main__":
    npy_ascat_one_station_estimate()
    npz_interest_ascat_pixel('prepare_files/npz/ascat/ascat_interest_pixel_series_2017.npz')