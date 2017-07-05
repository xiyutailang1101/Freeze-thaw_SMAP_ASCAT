import numpy as np
import h5py
from datetime import datetime
import os
import fnmatch
import re

def gt_le(x, a, b, mode='bool'):
    if mode == 'bool':
        return (a < x) & (x < b)
    elif mode == 'int':
        return np.where((a < x) & (x < b))[0]


def rm_mis(x, threshold):
    x[x < threshold] = np.nan
    return x

def cal_dis(lat0, lon0, lats, lons):
    '''
    Returen distance between two points with  lat/lon coordinates
    :param lat0:
    :param lon0:
    :param lats:
    :param lons:
    :return:
    '''
    lamda0 = (lon0/180.0)*np.pi
    lamdas = (lons/180.0)*np.pi
    phi0 = (lat0/180.0)*np.pi
    phis = (lats/180.0)*np.pi
    x = (lamdas-lamda0) * np.cos((phis+phi0)/2)
    y = phis - phi0
    return 6371*np.sqrt(x**2 + y**2)

def check_h5(filename):
    hf = h5py.File(filename)
    n_key = hf.keys()[2]
    print hf.keys(), '\n', n_key
    for key in hf[hf.keys()[2]].keys():
        print key
    print hf[n_key]['cell_row'][-1]
    hf.close()


def dis_inter(dis, value):
    weighs = 1/(dis)**2
    return np.sum(weighs*value)/np.sum(weighs)


def get_doy(date_string):
    """
    :param
        date_string: in the form of 'yyyymmdd'
    :return:
        doy_num: the num of doy
    """
    doy_num_list = []
    for strz in date_string:
        t = datetime.strptime(strz, '%Y%m%d').timetuple()
        doy_num_list.append(t.tm_yday + (t.tm_year - 2015) * 365)
    doy_num_np = np.array(doy_num_list)
    return doy_num_np


def find_by_date(date_str, path):
    """
    :param date_str:
    :return:
    """
    # regex = fnmatch.translate(date_str+'*.npy')
    # reobj = re.compile(regex)
    file_list = os.listdir(path)
    f_out = []
    for f0 in file_list:
        if fnmatch.fnmatch(f0, '*'+date_str+'*'):
            f_out.append(f0)
            file_list.remove(f0)
    return f_out

