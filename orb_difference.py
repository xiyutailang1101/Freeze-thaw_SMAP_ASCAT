__author__ = 'xiyu'
import numpy as np
import matplotlib as plt
import re
import os
import h5py
import log_write, site_infos


def search_ob_file(date, file_root):
    # search file by date
    # return a list contains the ascend and descend orbit file
    file_name = []
    for data_a in os.listdir(file_root):
        if re.search(date, data_a):  # data of the date
            file_name.append(file_root + data_a)
    if len(file_name) > 0:
        file_name = sorted(file_name)  # A - B
    # need write the existence of data to log
        return file_name
    else:
        print 'not enough data in %s' % date
        return -1


def cal_difference(a_orbit, d_orbit, site_info, attribute):
    va, fa = site_read(a_orbit, site_info, attribute)
    vd, fd = site_read(d_orbit, site_info, attribute)
    return va - vd, fa, fd


def site_read(file_name, site_info, attribute):
    hf_a = h5py.File(file_name, 'r')
    if attribute[0] not in hf_a.keys():
        hf_a.close()
        return -9999, -9999
    else:
        lat = np.array(hf_a[attribute[1]])
        lon = np.array(hf_a[attribute[2]])
        dis = (lat - site_info[1])**2 + (lon - site_info[2])**2
        min_ind = np.argmin(dis)
        value = np.array(hf_a[attribute[0]][min_ind])
        if attribute[3] in hf_a.keys():
            flag = np.array(hf_a[attribute[3]][min_ind])
            hf_a.close()
            return value, flag
        else:
            return value, -9999



def cal_main(site_no):
    # start the loops, use date list
    # site_no = '950'
    data_root = site_infos.get_data_root() + '/s' + site_no + '/'
    date_list = []
    for filename in os.listdir(data_root):  # the folder contains h5 results for a site
        date_list.append(filename[-11:-3])  # the date list
    date_list = set(date_list)
    date_list = sorted(date_list)
    site_info = site_infos.change_site(site_no)
    attributes = ['cell_sigma0_vv_aft', 'sig_cell_lat', 'sig_cell_lon', 'cell_sigma0_qual_flag_vv']
    attributes2 = ['cell_tb_v_aft', 'tb_cell_lat', 'tb_cell_lon', ' ']
    ob_diference = np.zeros([4, len(date_list)])
    print ob_diference.shape
    n = 0
    for date_obj in date_list:  # loop dates in date list
        file_list = search_ob_file(date_obj, data_root)
        # if date_obj != '20150501':
        #     continue
        if n == 0:
            print 'date    ', 'radar       ', 'tb:    ', site_no
        if len(file_list) > 1:
            # radar data
            differ1, flag_radar_A, flag_radar_D = cal_difference(file_list[0], file_list[1], site_info, attributes)
            # tb data
            differ2, flag_tb_A, flag_tb_D = cal_difference(file_list[0], file_list[1], site_info, attributes2)
            ob_diference[0][n] = differ1
            ob_diference[1][n] = differ2
            ob_diference[2][n] = flag_radar_A
            ob_diference[3][n] = flag_radar_D
            print date_obj, differ1, differ2, flag_radar_A, flag_radar_D
        else:
            ob_diference[0][n], ob_diference[1][n] = -9999, -9999
        n += 1
    return ob_diference