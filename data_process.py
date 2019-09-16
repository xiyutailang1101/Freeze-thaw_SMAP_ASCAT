"""
Deal the data that have been read
"""
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as pylt
import sys
import os
import h5py
import log_write
import site_infos
import read_site
import datetime
import math
import peakdetect
import csv
import plot_funcs
import basic_xiyu as bxy
from mpl_toolkits.basemap import Basemap
import test_def
from datetime import timedelta
import re
from osgeo import gdal, osr
import glob
from matplotlib.patches import Path, PathPatch
import multiprocessing


def get_ft_thresh(series, date_str, onset):
    """
    Calculate reference of f & t, f is based on the average backscatter of beginning to 7 days before thaw onset
    t is the average from 10 days after thaw onset to end of May
    :param
        series: series is 2-row data, one is value, the other is flag
    :param
        onset: thaw onset
    :return:
        freeze and thaw reference
    """
    f_series, ab = read_site.get_abnormal(series, [1, 11, 13])
    f_series[f_series == -9999] = np.nan
    if onset in date_str:
        pos = date_str.index(onset)
    fr = np.nanmean(f_series[0][0: pos - 7])
    th = np.nanmean(f_series[0][pos + 7: -1])
    return fr, th


def get_doy(date_string, fm='%Y%m%d'):
    """

    :param
        date_string: in the form of 'yyyymmdd'
    :return:
        doy_num: the num of doy from 2015.01.01
    """
    doy_num_list = []
    if type(date_string) is not list:
        date_string = [date_string]
    for strz in date_string:
        t = datetime.datetime.strptime(strz, fm).timetuple()
        doy_num_list.append(t.tm_yday + (t.tm_year - 2015) * 365)
    doy_num_np = np.array(doy_num_list)
    return doy_num_np


def get_date_obj(date_string):
    """

    :param
        date_string: in the form of 'yyyymmdd'
    :return:
        doy_num: the num of doy
    """
    obj_list = []
    for strz in date_string:
        t = datetime.datetime.strptime(strz, '%Y%m%d')
        obj_list.append(t)
    date_obj_np = np.array(obj_list)
    return date_obj_np


def cal_emi(sno, tb, doy, hrs='emi', t_unit='days'):
    """
    <discription>:
        1. emi: Based on mode, to calculate emissivity or read in situ measurement at a certain timing
    :param sno:
    :param tb:
    :param doy:
    :param hrs: two models, 'emi' for emissivity cal. Integrate for read data with the interval in hour, e.g., read data per 6 hour
    :return:
    """
    ind = np.array([])
    if hrs is False:
        tb_doy = get_doy(doy) + 0.75
        date_sno = sno[0][:]
        t_5 = sno[1][:]
        ind = np.argwhere(np.in1d(date_sno, tb_doy))
        # consider the complement sets: elements in tb_doy but not in date_no, {tb_doy - date_no}
        missing_date = np.setdiff1d(tb_doy,
                                    np.intersect1d(date_sno, tb_doy))  # this missing date of in situ mearsurement
        miss_ind = np.in1d(tb_doy, missing_date).nonzero()  # or try ind = np.in1d(date_sno, tb_doy).nonzero()
        tb_doy = np.delete(tb_doy, miss_ind[0])
        tb = np.delete(tb, miss_ind[0])
        emissivity = tb / (t_5[ind].T + 273.12)
        emissivity = emissivity[0][:]
        return emissivity
    else:
        # time pass of satellite
        if type(hrs) is int:
            tb_doy = doy + np.round(hrs) / 24.0
            if t_unit == 'sec' or t_unit == 'secs':
                tb_doy = doy + np.round(hrs) * 3600
        else:
            if t_unit == 'secs' or t_unit == 'sec':
                tb_doy = [doy[i0] + hrs[i0] * 3600 for i0 in range(0, doy.size)]
            else:
                tb_doy = [doy[i0] + hrs[i0] / 24.0 for i0 in range(0, doy.size)]
            pause = 0
        date_sno = sno[0][:]
        data = sno[1][:]
        index0 = np.array([])
        for t in tb_doy:
            ind0 = np.where(date_sno == date_sno[np.argmin(np.abs(date_sno - t))])  # date_sno: from station txt file
            if ind0[0].size < 1:
                pause = 0
            elif ind0[0].size > 1:
                index0 = np.append(index0, ind0[0][0])
            else:
                index0 = np.append(index0, ind0[0])
        # ind2 = np.int_(np.sort(ind, axis=0))

        # remove the unreal sudden change
        index = index0.astype(int)
        # i_valid_change = np.where(np.abs(np.diff(data[index]))<75)
        data_valid = data[index]  # [i_valid_change[0] + 1]
        date_valid = date_sno[index]  # [i_valid_change[0] + 1]
        return data_valid, date_valid


def cal_emi_v2(sno, tb, doy, hrs='emi'):
    """
    <discription>:
        1. emi: Based on mode, to calculate emissivity or read in situ measurement at a certain timing
    :param sno:
    :param tb:
    :param doy: actually is the secs, updated 20190304
    :param hrs: two models, 'emi' for emissivity cal. Integrate for read data with the interval in hour, e.g., read data per 6 hour
    :return:
    """
    ind = np.array([])
    if hrs is False:
        tb_doy = get_doy(doy) + 0.75
        date_sno = sno[0][:]
        t_5 = sno[1][:]
        ind = np.argwhere(np.in1d(date_sno, tb_doy))
        # consider the complement sets: elements in tb_doy but not in date_no, {tb_doy - date_no}
        missing_date = np.setdiff1d(tb_doy,
                                    np.intersect1d(date_sno, tb_doy))  # this missing date of in situ mearsurement
        miss_ind = np.in1d(tb_doy, missing_date).nonzero()  # or try ind = np.in1d(date_sno, tb_doy).nonzero()
        tb_doy = np.delete(tb_doy, miss_ind[0])
        tb = np.delete(tb, miss_ind[0])
        emissivity = tb / (t_5[ind].T + 273.12)
        emissivity = emissivity[0][:]
        return emissivity
    else:
        # time pass of satellite
        if type(hrs) is int:
            tb_doy = doy + np.round(hrs) * 3600
        else:
            tb_doy = [doy[i0] + hrs[i0] / 24.0 for i0 in range(0, doy.size)]
            tb_doy = doy
            pause = 0
        date_sno = sno[0]
        data = sno[1:]
        index0 = np.array([])
        for t in tb_doy:
            ind0 = np.where(date_sno == date_sno[np.argmin(np.abs(date_sno - t))])  # date_sno: from station txt file
            if ind0[0].size < 1:
                pause = 0
            elif ind0[0].size > 1:
                index0 = np.append(index0, ind0[0][0])
            else:
                index0 = np.append(index0, ind0[0])
        # ind2 = np.int_(np.sort(ind, axis=0))

        # remove the unreal sudden change
        index = index0.astype(int)
        # i_valid_change = np.where(np.abs(np.diff(data[index]))<75)
        data_valid = data[:, index]  # [i_valid_change[0] + 1]
        date_valid = date_sno[index]  # [i_valid_change[0] + 1]
        return data_valid, date_valid


def get_4var(filename, site_no, prj='np', dat='tb_v_aft'):
    hf = h5py.File(filename, 'r')
    # the lat and lon of all available tb pixel
    attribute = site_infos.get_attribute(prj, dat)
    lat = hf[attribute[1]].value
    lon = hf[attribute[2]].value
    var = hf[attribute[0]].value
    if attribute[0] not in hf:
        print filename
        print 'no tb attribute'
    # find the nearest 4
    site_info = site_infos.change_site(site_no)
    if prj == 'tbn':
        # important ind - coord relation: near0 - ref, near1 - side0, near2 - side1
        dis = (lat - site_info[1]) ** 2 + (lon - site_info[2]) ** 2
        near_ind = np.argsort(dis)
        near0, near1, near2 = near_ind[0], near_ind[1], near_ind[2]
        # coordinate and value of center and 2 direction side
        ref = np.array([lon[near0], lat[near0]])
        side0, side1 = np.array([lon[near1], lat[near1]]), np.array([lon[near2], lat[near2]])
        # vector of two sides of envelope square
        vec0 = np.array([side0[0] - ref[0], side0[1] - ref[1]])
        vec1 = np.array([side1[0] - ref[0], side1[1] - ref[1]])
        modul0, modul1 = np.sqrt(vec0[0] ** 2 + vec0[1] ** 2), np.sqrt(vec1[1] ** 2 + vec1[0] ** 2)
        cos0_1 = np.dot(vec0, vec1) / modul0 / modul1

        #  select the diagonal point to the ref from the remained points
        diags = np.array(
            [lon[near_ind[3:]] - ref[0], lat[near_ind[3:]] - ref[1]])  # vectors of other points as [xn, yn].T
        diags_mod = np.sqrt(diags[0, :] ** 2 + diags[1, :] ** 2)  # modules of the above vectors
        # indicator_ref = np.dot(vec_ref, diags)/(diags_mod * ref_mod)  # cosines of diags vectors with reference
        if 0 < cos0_1 < 0.174:  # angle of 0-1 within 80 to 90 degree
            print "the two sides of envelop were found"
            indicator1 = np.dot(vec0, diags)
            indicator2 = np.dot(vec1, diags)
            # ind_ref = np.where(indicator_ref > 0)  # cosines within 0 to pi
            ind1 = np.where(indicator1 > 0)
            ind2 = np.where(indicator2 > 0)
            ind_diag = np.intersect1d(ind1, ind2)
            if not ind_diag:
                print 'not enough pixels to cover the station'
                return -1, -1, -1
            diag = diags[:, ind_diag[0]] + ref
            # if ind_ref.size > 2:
            #     diag_gt_0 = diags[ind_ref]
            #     order_cos = np.sort(diag_gt_0)
            # else:
            p_4 = np.array([ref, side0, side1, diag])
            ind_lt = np.where(lat == diag[1])
            ind_lg = np.where(lon == diag[0])
            ind_diag = np.intersect1d(ind_lt, ind_lg)
            v_4 = np.array([var[near0], var[near1], var[near2], var[ind_diag]])
        else:
            print "one side and the diagonal were founded"
            # cal the angles between diagonal and other vector, select the one smaller than 1/4 pi as diagonal. the 2nd
            # side (side1) actually is the diagonal
            # find the diagonal
            if modul0 < modul1:
                diag_vec, diag_mod = vec1, modul1
            else:
                diag_vec, diag_mod = vec0, modul0
            dig_cosine = np.dot(diag_vec, diags) / (diags_mod * diag_mod)
            ind_side = np.where(dig_cosine > 0.5)  # the other side has a angle with side that is lt 1/3 pi
            diag = diags[:, ind_side[0]].T + ref  # diag here represents the side
            p_4 = np.array([ref, side0, diag[0], side1])  # in form of [lonN, latN]
            ind_lt = np.where(lat == diag[0][1])
            ind_lg = np.where(lon == diag[0][0])
            ind_diag = np.intersect1d(ind_lt, ind_lg)
            if ind_diag.size is 0:
                print 'not enough pixels to cover the station'
                return -1, -1, -1
            v_4 = np.array([var[near0], var[near1], var[ind_diag], var[near2]])
        # obtain the coordinates and value of envelop
        lon_order = np.argsort(p_4[:, 0])  # ordered by longitude, sort the points as upL, lowL, upR, lowR
        loc4 = p_4[lon_order]  # upL, lowL, upR, lowR: in (x, y)
        var4 = v_4[lon_order]
        return 1, loc4, var4
    else:  # when using the GM projection, first find the two longitude that cover the station.
        lon_u = np.sort(np.unique(lon))
        ind_r = np.searchsorted(lon_u, site_info[2])
        if ind_r > 0:
            ind_l = ind_r - 1
        else:
            print 'cannot find the pixel around this station longitude'
        lat_u = np.sort(np.unique(lat))
        ind_up = np.searchsorted(lat_u, site_info[1])
        if ind_up > 0:
            ind_low = ind_up - 1
        else:
            print 'cannot find the pixel around this station in latitude'
        loc4 = np.array([[lon_u[ind_l], lat_u[ind_up]], [lon_u[ind_l], lat_u[ind_low]], [lon_u[ind_r], lat_u[ind_up]],
                         [lon_u[ind_r], lat_u[ind_low]]])
        ind0 = np.intersect1d(np.where(lat == loc4[0][1]), np.where(lon == loc4[0][0]))
        ind1 = np.intersect1d(np.where(lat == loc4[1][1]), np.where(lon == loc4[1][0]))
        ind2 = np.intersect1d(np.where(lat == loc4[2][1]), np.where(lon == loc4[2][0]))
        ind3 = np.intersect1d(np.where(lat == loc4[3][1]), np.where(lon == loc4[3][0]))
        var4 = [var[ind0[0]], var[ind1[0]], var[ind2[0]], var[ind3[0]]]

    print var4
    return 1, loc4, var4


def interpolate2d(filename, site_no, prj='tbn'):
    stats_4var, locs, vars = get_4var(filename, site_no, prj)
    site_info = site_infos.change_site(site_no)  # 1:lat, 2:lon
    x, y = site_info[2], site_info[1]
    if stats_4var == -1:
        print "cannot find enough tb pixels around the station"
        return -9999
    # get the function of each side of envelope square
    line_x = (locs[2][1] - locs[0][1]) * x - (locs[2][0] - locs[0][0]) * y - locs[0][0] * locs[2][1] + locs[2][0] * \
                                                                                                       locs[0][1]
    line_y = (locs[1][1] - locs[0][1]) * x - (locs[1][0] - locs[0][0]) * y - locs[0][0] * locs[1][1] + locs[1][0] * \
                                                                                                       locs[0][1]
    d2xline = line_x ** 2 / (
    (locs[2][1] - locs[0][1]) ** 2 + (locs[2][0] - locs[0][0]) ** 2)  # squared distance to x line
    d2yline = line_y ** 2 / (
    (locs[1][1] - locs[0][1]) ** 2 + (locs[1][0] - locs[0][0]) ** 2)  # squared distance to y line
    dx = math.sqrt((locs[0][0] - locs[2][0]) ** 2 + (locs[0][1] - locs[2][1]) ** 2)  # length of x line
    dy = math.sqrt((locs[0][0] - locs[1][0]) ** 2 + (locs[0][1] - locs[1][1]) ** 2)  # length of y line
    d = (locs[0][0] - site_info[2]) ** 2 + (locs[0][1] - site_info[1]) ** 2  # square of station to up-left
    disx1 = math.sqrt(d - d2xline)
    disx = np.array([dx - disx1, disx1])
    disy2 = math.sqrt(d - d2yline)
    disy = np.array([[disy2], [dy - disy2]])
    fmn = np.array([[vars[1], vars[0]], [vars[3], vars[2]]])
    # var = 1/(dx * dy) * disx * fmn * disy
    var = 1 / (dx * dy) * np.dot(np.dot(disx, fmn), disy)
    return var


def interp_radius(filename, site_no, prj='np', dat='smap_tb', disref=0.5):
    s_info = site_infos.change_site(site_no)
    attributes = 'North_Polar_Projection'
    atts = site_infos.get_attribute('np', sublayer=dat)
    hf_l0 = h5py.File(filename, 'r')
    # the lat and lon of all available tb pixel
    # print filename[48: ]
    if attributes not in hf_l0:
        hf_l0.close()
        stat = -1
        print '%s was not included in \n %s' % (attributes, filename[48:])
        return [-9999 + i * 0 for i in range(len(atts[1]))], -9999, -9999, -1
    hf = hf_l0[attributes]  # open second layer
    lat = hf['cell_lat'].value
    lon = hf['cell_lon'].value
    dis = bxy.cal_dis(s_info[1], s_info[2], lat, lon)
    if site_no == '960':
        disref = 19
    inner = np.where(dis < disref)
    # if site_no == '949':
    #     inner = np.where((dis >18) & (dis<19))
    dis_inner = dis[inner[0]]
    if site_no == '1177':
        # inner_id = inner[0]
        # if dis_inner.size > 1:
        #     print '%d pixels:' % dis_inner.size, dis_inner, ' within the r=%d km' % disref
        print dis_inner
        inner = inner[0][dis_inner > 20]
        dis_inner = dis_inner[dis_inner > 20]
    interpolated_value = []
    if dis_inner.size > 0:  # not empty
        inner_id = inner[0]
        print site_no, 'has neighbor: ', inner_id.size
        v_list = np.array([hf[atti].value for atti in atts[1]])
        for atti in atts[1]:
            var = hf[atti].value
            for i0, val0 in enumerate(var):
                if val0 == -9999:
                    if atti == 'cell_tb_v_aft':
                        var[i0] = hf['cell_tb_v_fore'].value[i0]
                    elif atti == 'cell_tb_h_aft':
                        var[i0] = hf['cell_tb_h_fore'].value[i0]
            var_inner = var[inner_id]
            v_interp = bxy.dis_inter(dis_inner, var_inner)
            interpolated_value.append(v_interp)
        return interpolated_value, dis_inner, [lon[inner_id], lat[inner_id]], 0
    else:
        return [-9999 + i * 0 for i in range(len(atts[1]))], -9999, -9999, -1


def read_all_pixel(filename, site_no, prj='np', dat='smap_tb', disref=0.5):
    # the lat/lon of pixel
    path0 = 'result_07_01/txtfiles/SMAP_pixel/'
    txt_list = glob.glob(path0 + 'subcent_' + site_no + '*.txt')
    attributes = 'North_Polar_Projection'
    atts = site_infos.get_attribute('np', sublayer=dat)
    hf_l0 = h5py.File(filename, 'r')
    if attributes not in hf_l0:
        hf_l0.close()
        stat = -1
        print '%s was not included in \n %s' % (attributes, filename[48:])
        return -1, -1, -1
    hf = hf_l0[attributes]  # open second layer
    lat = hf['cell_lat'].value
    lon = hf['cell_lon'].value
    # read TB within each pixels
    # initial
    dis_list = np.array([-1, -1, -1, -1])
    tb_pixels = np.zeros([len(txt_list), len(atts[1])]) - 9999
    for n_p, pixel0_path in enumerate(txt_list):
        txt_fname = pixel0_path.split('/')[-1]
        pixel_dis = txt_fname.split('_')[-1].split('.')[0]  # the distance from center to site, as id for each pixel
        dis_list[n_p] = float(pixel_dis)
        pixel_info = np.loadtxt(pixel0_path, delimiter=',')
        s_info = [0, pixel_info[1, 4], pixel_info[0, 4]]
        dis = bxy.cal_dis(s_info[1], s_info[2], lat, lon)
        inner = np.where(dis < 1)
        dis_inner = dis[inner[0]]
        interpolated_value = []
        if dis_inner.size > 0:  # not empty
            inner_id = inner[0]
            print site_no, 'has neighbor: ', inner_id.size
            v_list = np.array([hf[atti].value for atti in atts[1]])
            for atti in atts[1]:
                var = hf[atti].value
                for i0, val0 in enumerate(var):
                    if val0 == -9999:
                        if atti == 'cell_tb_v_aft':
                            var[i0] = hf['cell_tb_v_fore'].value[i0]
                        elif atti == 'cell_tb_h_aft':
                            var[i0] = hf['cell_tb_h_fore'].value[i0]
                var_inner = var[inner_id]
                v_interp = bxy.dis_inter(dis_inner, var_inner)
                interpolated_value.append(v_interp)
            tb_pixels[n_p, :] = interpolated_value
    return tb_pixels, dis_list, 0  # value, distance, coordinate, status
    # else:
    #     # interpolated_value.append([-9999 + i*0 for i in range(len(atts[1]))])
    #     return [-9999 + i*0 for i in range(len(atts[1]))], -9999, -9999, -1




    # s_info = site_infos.change_site(site_no)
    attributes = 'North_Polar_Projection'
    atts = site_infos.get_attribute('np', sublayer=dat)
    hf_l0 = h5py.File(filename, 'r')
    # the lat and lon of all available tb pixel
    # print filename[48: ]
    if attributes not in hf_l0:
        hf_l0.close()
        stat = -1
        print '%s was not included in \n %s' % (attributes, filename[48:])
        return [-9999 + i * 0 for i in range(len(atts[1]))], -9999, -9999, -1
    hf = hf_l0[attributes]  # open second layer
    lat = hf['cell_lat'].value
    lon = hf['cell_lon'].value
    dis = bxy.cal_dis(s_info[1], s_info[2], lat, lon)
    if site_no == '960':
        disref = 19
    inner = np.where(dis < disref)
    # if site_no == '949':
    #     inner = np.where((dis >18) & (dis<19))
    dis_inner = dis[inner[0]]
    if site_no == '1177':
        # inner_id = inner[0]
        # if dis_inner.size > 1:
        #     print '%d pixels:' % dis_inner.size, dis_inner, ' within the r=%d km' % disref
        print dis_inner
        inner = inner[0][dis_inner > 20]
        dis_inner = dis_inner[dis_inner > 20]
    interpolated_value = []
    if dis_inner.size > 0:  # not empty
        inner_id = inner[0]
        print site_no, 'has neighbor: ', inner_id.size
        v_list = np.array([hf[atti].value for atti in atts[1]])
        for atti in atts[1]:
            var = hf[atti].value
            for i0, val0 in enumerate(var):
                if val0 == -9999:
                    if atti == 'cell_tb_v_aft':
                        var[i0] = hf['cell_tb_v_fore'].value[i0]
                    elif atti == 'cell_tb_h_aft':
                        var[i0] = hf['cell_tb_h_fore'].value[i0]
            var_inner = var[inner_id]
            v_interp = bxy.dis_inter(dis_inner, var_inner)
            interpolated_value.append(v_interp)
        return interpolated_value, dis_inner, [lon[inner_id], lat[inner_id]], 0
    else:
        return [-9999 + i * 0 for i in range(len(atts[1]))], -9999, -9999, -1
    return 0


def interpolation_spatial(values, lon, lat, site_no, disref=0.25):
    s_info = site_infos.change_site(site_no)
    dis = bxy.cal_dis(s_info[1], s_info[2], lat, lon)
    nn = np.where(dis < disref)  # the nearest neighbours within a radius distance "dis_ref"
    dis_nn = dis[nn[0]]
    if dis_nn.size > 0:  # not empty
        nn_id = nn[0]
        print site_no, 'has neighbor: ', nn_id.size
        values_nn = values[nn_id]
        value_interp = bxy.dis_inter(dis_nn, values_nn)  # perform 2d spatial interpolation
        return value_interp, dis_nn, [lon[nn_id], lat[nn_id]], 0
    else:
        return 0, 0, 0, -1


def interp_geo(x, y, z, orb, site_no, disref=0.5, f=None, incidence=None):
    f_bad = np.where(f > 0)
    if any(f_bad[0] > 0):
        print 'not usable at %s' % site_no
    s_info = site_infos.change_site(site_no)
    v_interp = []
    v_ind = []
    for orbit in [0, 1]:
        i_orb = np.where(orb == orbit)
        if len(i_orb) > 0:
            xo, yo = x[i_orb], y[i_orb]
            dis = (yo - s_info[1]) ** 2 + (xo - s_info[2]) ** 2
            if dis.size < 1:
                continue
            inner = np.where(dis < disref ** 2)[0]
            if disref == 0:  # using the nearest one
                inner = np.argmin(dis)
                dis_inner = dis[inner]
                z_inner = z[:, i_orb[0][inner]]
                v_interp.append(z_inner)
                v_ind.append(i_orb[0][inner])
                status = 1
            else:
                dis_inner = dis[inner]
                z_inner = z[:, i_orb[0][inner]]
                di = 1.0 / dis_inner
                v1 = np.sum(di * z_inner, axis=1)
                vn = np.sum(di)
                if vn > 0:
                    v_interp.append(v1 * 1.0 / vn)
                    v_ind.append(i_orb[0][inner])
                    status = 1
                else:
                    v_interp.append(-99)
                    v_ind.append(i_orb[0][inner])
                    status = -1

    return v_interp, -99, v_ind, status


def step_func(y2, t, doy):
    """
    iterate based on the timing of thawing, to generate a squared value
    :param y2: The observations of satellite, e.g., sigma
    :param t: contains all possible timing of spring thawing
    :param doy: the day of year corresponding to y2
    :return:
    """
    sums = []
    date_sums = []
    t = np.linspace(t[0], t[1], num=t[1] - t[0] + 1)
    for timing in t:
        ind = np.where(doy == timing)
        if ind[0].size > 0:
            sig_w = np.nanmean(y2[0: ind[0]])
            sig_s = np.nanmean(y2[ind[0]:])
            ft = np.zeros(y2.size)
            ft[0: ind[0]] = sig_w
            ft[ind[0]:] = sig_s
            sums.append(np.nansum((y2 - ft) ** 2))
            date_sums.append(timing)
        else:
            "print not such a date"
    ts = np.argmin(sums)
    ind = np.where(doy == date_sums[ts])
    sig_w = np.nanmean(y2[0: ind[0]])
    sig_s = np.nanmean(y2[ind[0]:])
    ft = np.zeros(y2.size)
    ft[0: ind[0]] = sig_w
    ft[ind[0]:] = sig_s
    return date_sums[ts], ft, np.array([sums, date_sums])


def list_mean(num_list):
    """
    <Introduction>
        Return the mean value of numbers saved in a list
    <:param> num_list: must contains number type vars
    :return: the mean value
    """
    np_list = np.array(num_list)
    np_mean = np.nanmean(np_list)

    return np_mean


def rm_temporal(series1, series2):
    """
    <description>
    :param series:
    :return:
    """
    diff = series1 - series2
    d_mean = np.nanmean(diff)
    d_std = np.nanstd(diff)
    crit = np.where(diff > d_mean + 3 * d_std)  # warnings if there are nan value in diff
    # crit = np.where(series2 < 220)
    series2[crit] = np.nan
    rm_odd(series2)
    return diff


def rm_odd(series, s_date, iter):
    id0 = np.where(~np.isnan(series))[0]  # id0.size = diff.size+1
    diff = np.diff(series[id0])
    d_std = np.nanstd(diff)
    crit = np.where(np.abs(diff) > 3 * d_std)
    series_out = series
    if crit[0].size > 0:
        series[crit[0].astype(int) + 1] = np.nan
        for idx in id0[crit[0].astype(int) + 1]:
            series[idx] = (np.nanmean(series[idx - 2: idx]) + np.nanmean(series[idx + 1: idx + 3])) / 2.0
        if iter < 3:
            iter += 1
            rm_odd(series, s_date, iter)
        else:
            return diff
    else:
        return diff


def rm_diff(series):
    return 0


def n_smooth(series, n):
    w = np.ones(n, ) / n
    r = np.convolve(series, w, mode='same')
    valid = np.linspace((n - 1) / 2, series.size - (n + 1) / 2, series.size - n + 1, dtype=int)
    return r, valid


def n_convolve(series, n):  # this is useless!!!!!!!!!!!!!!!
    win = np.ones(n, )
    k = series.size + win.size - 1
    n = win.size
    m = series.size
    w = np.zeros(k)
    for i in range(0, k):
        wsize = range(max(0, i + 1 - n), min(i, m - 1))  # the index saved in moving window
        tp = series[wsize]
        num_nan = np.count_nonzero(np.isnan(tp))  # find nans in the window
        mw = win / (n - num_nan)
        tp[np.isnan(tp)] = 0
        for j in wsize:
            n1 = 0
            w[i] += tp[n1] * mw[i - j]
            n1 += 1
    return w


def n_convolve2(series, n):
    '''
    <intro>
        Moving average function
    :param series:
    :param n:
    :return:
    '''
    win = np.ones(n, )
    n = win.size
    m = series.size
    k = m + n - 1
    w = np.zeros(k)
    for i in range(0, k):
        if np.isnan(series[min(i, m - 1)]):
            w[i] = np.nan
            continue
        wsize = range(max(0, i + 1 - n), min(i, m - 1) + 1)  # the index saved in moving window
        num_nan = np.count_nonzero(np.isnan(series[wsize]))  # find nans in the window
        mw = win / (n - num_nan)  # set new mean value for this window
        for j in wsize:
            if np.isnan(series[j]):
                tp = 0  # nan value set as 0
            else:
                tp = series[j]
            w[i] += tp * mw[i - j]
    valid = np.linspace(n - 1, m - 1, m - n + 1, dtype=int)
    return w, valid


def n_convolve3(series, n):
    '''
    <intro>
        count the number of positive number
    :param series:
    :param n:
    :return:
    '''
    win = np.ones(n, )
    n = win.size
    m = series.size
    k = m + n - 1
    w = np.zeros(k)
    for i in range(0, m):
        if np.isnan(series[min(i, m - 1)]):
            w[i] = np.nan
            continue
        wsize = np.arange(max(0, i + 1 - n), min(i, m - 1) + 1)  # the index saved in moving window
        print 'overlap: ', wsize
        num_nan = np.count_nonzero(np.isnan(series[wsize]))  # find nans in the window
        mw = series[-np.sort(-wsize)]
        for j in wsize:
            if np.isnan(series[j]):
                tp = 0  # nan value set as 0
            else:
                tp = series[j]
            print 'the signal i:%d, kernel j: %d' % (i, j)

            w[i] += tp * max(0, 1.0 / mw[i - j])
    valid = np.linspace(0, m, m, dtype=int)
    return w, valid


def gauss_conv(series, sig=1, size=17, fill_v=[-999, 0, -99, -9999], sorted=True, n=4, sig2=False):
    """

    :param series:
    :param sig:
    :return:
        ig: the true/false index of value that is not a nan
    """
    # print 'the standard deviation used is', sig
    # i_fill = [i for i, x in enumerate(series) if x in fill_v]
    size = 6 * sig + 1

    # for fill0 in fill_v:
    #     series[series == fill0] = np.nan
    # if sorted is not True:
    #     series = series[sorted]
    # ig = ~np.isnan(series)  # ig, the the valid value in the series

    for fill0 in fill_v:
        series[series == fill0] = -99999
    ig = series > -99999

    x = np.linspace(-size / 2 + 1, size / 2, size)
    filterz = ((-x) / sig ** 2) * np.exp(-x ** 2 / (2 * sig ** 2))
    if sig2:
        size2 = 6 * sig2 + 1
        x2 = np.linspace(-size2 / 2 + 1, size2 / 2, size2)
        filterz2 = ((-x2) / sig2 ** 2) * np.exp(-x2 ** 2 / (2 * sig2 ** 2))
        filter_re = filterz2[filterz2>0]
        loc_0 = np.where(filterz==0)[0][0]  # location of 0 in the original filter
        filterz3 = filterz.copy()
        # print 'loc_0: ', loc_0
        # print 'filter_re.size', filter_re.size
        # print filter_re
        filterz3[loc_0-filter_re.size: loc_0] = filter_re
        filterz3[0: loc_0-filter_re.size] = 0
        filterz = filterz3
        # plot a figure to check the shape
        # f0 = pylt.figure()
        # ax0 = f0.add_subplot(1, 1, 1)
        # ax0.plot(x, filterz3)
        # pylt.savefig('check_long_short_convolution.png')
    if n < 4:
        sig_new = 3
        filterz2 = ((-x) / sig_new ** 2) * np.exp(-x ** 2 / (2 * sig_new ** 2))
        filterz[x < 0] = filterz2[x < 0]

        fig_g, axg = plt.pyplot.subplots()
        axg.plot(x, filterz)
        fig_g.text(0.8, 0.8, 'sigma = %d' % sig)
        plt.pyplot.savefig('kenerls3.png', dpi=120)
        plt.pyplot.close()
    if series[ig].size < 1:
        # print 'gauss_conv, the non-nan elements does not exit'
        return np.array([-1]), np.array([-1])
    else:
        f1 = np.convolve(series[ig], filterz, 'same')
    # local maximum and minimum
    return f1, ig


def gauss_cov_snr(series, peaks_iter, t_series, t_window=[[80, 150], [245, 350]], sig=1, size=17, s_type='other'):
    ons1 = -1
    ons2 = -1
    for sig in np.arange(5, 11, 1):
        size = 6 * sig + 1
        gsize = 6 * sig / 2
        ig = ~np.isnan(series)
        x = np.linspace(-size / 2 + 1, size / 2, size)
        filterz = ((-x) / sig ** 2) * np.exp(-x ** 2 / (2 * sig ** 2))
        f1 = np.convolve(series[ig], filterz, 'same')
        convs = f1[gsize: -gsize]
        t_convs = t_series[ig][gsize: -gsize]
        max_convs, min_convs = peakdetect.peakdet(convs, peaks_iter, t_convs)
        # relation: max/min -- win_thaw/freeze
        if s_type != 'tb':
            ons1 = find_infleclt_signal(max_convs, t_window[0], convs, t_convs)  # thaw
            ons2 = find_infleclt_signal(min_convs, t_window[1], convs, t_convs)  # freeze
        elif s_type == 'tb':
            ons1 = find_infleclt_signal(min_convs, t_window[0], convs, t_convs)  # thaw
            ons2 = find_infleclt_signal(max_convs, t_window[1], convs, t_convs)  # freeze
        if (ons1 != -1) & (ons2 != -1):
            return np.array([ons1, ons2]), f1, ig, convs, max_convs, min_convs, sig
    return np.array([ons1, ons2]), -1, -1, -1. - 1, -1


def edge_detect(t_series, edge_series, s, s2=3, order=1, seriestype='tb', is_sort=True, w=4, long_short=False, sig2=3):
    """
    :param t_series: from overpass second
    :param edge_series: ft indicators
    :param s: sigma of gaussian filter
    :param order: 1st detrivative of gaussian
    :param seriestype: the name of indicator.
    :param w: window in unit of std
    :return:
    max_npr_valid, default value is np.array([[-1., t_series[0], -1.]])
    min_npr_valid,
    np.array([t_valid, conv_valid])
    """
    snr_threshold = 0
    if order == 1:  # first order of gaussian
        if seriestype == 'tb':
            peaks_iter = 1e-1
        elif seriestype == 'npr':
            peaks_iter = 1e-4
        elif seriestype == 'sig' or seriestype == 'sigma':
            peaks_iter = 5e-2
        else:
            peaks_iter = 1e-4
        g_size = 6*s/2
        if is_sort is not True:  # sorted the t
            i_sort = np.argsort(t_series)
            t_series = t_series[i_sort]
            edge_series = edge_series[i_sort]
        if long_short:
            g_npr, i_gaussian = gauss_conv(edge_series, sig=s, n=w, sig2=s2)
        else:
            g_npr, i_gaussian = gauss_conv(edge_series, sig=s, n=w)  # option: ffnpr-t_h; var_npv-t_h
        if g_npr.size < 2:
            return np.array([[-999, -999, -999]]), np.array([[-999, -999, -999]]), \
                   np.zeros([2, t_series.size])[g_size: -g_size] - 999
        conv_valid = g_npr[g_size: -g_size]  # valid interval: g_size: -g_size
        max_gnpr, min_gnpr = peakdetect.peakdet(conv_valid, peaks_iter, t_series[i_gaussian][g_size: -g_size])
        # calculate the winter mean convolution as well as the snr
        t_valid = t_series[i_gaussian][g_size: -g_size]
        if t_valid.size < 1:
            t_valid = np.array([np.min(t_series[i_gaussian])])
        max_npr_valid = max_gnpr
        min_npr_valid = min_gnpr
        if max_npr_valid.size < 1:
            max_npr_valid = np.array([[-1., t_series[0], -1.]])
        if min_npr_valid.size < 1:
            min_npr_valid = np.array([[-1., t_series[0], -1.]])
        return max_npr_valid, min_npr_valid, np.array([t_valid, conv_valid])


def find_infleclt_signal(tabs, window, convs, t_convs):
    onset = 0
    snr_threshold = 0
    t_date = t_convs - 365
    convs_win = convs[(t_date > 10) & (t_date < 70)]  # winter window: 10~70
    mean_wind, std_win = np.nanmean(convs_win), np.nanstd(convs_win)
    snr_threshold = [mean_wind - 3 * std_win, mean_wind + 2 * std_win]
    snr_thershold = [100, -100]
    dayz = tabs[:, 1] - 365
    window_trans = (dayz > window[0]) & (dayz < window[1])
    v_z = tabs[:, -1][window_trans]
    onset_all = dayz[window_trans][(v_z > snr_threshold[1]) | (v_z < snr_threshold[0])]
    if onset_all.size > 0:
        onset = onset_all[0]
    else:
        onset = -1

    # np.set_printoptions(precision=5, suppress=True)
    # print 'Filter width: ', w0
    # trans2winter[i0] = np.abs(np.nanmean(extremum[window_trans])/conv_std_win)
    return onset


def gauss2nd_conv(series, sig=1):
    size = 6 * sig + 1
    ig = ~np.isnan(series)
    x = np.linspace(-size / 2 + 1, size / 2, size)
    # filterz = (np.sqrt(1/(2*sig**2*np.pi)))*((-x)/sig**2)*np.exp(-x**2/(2*sig**2))
    filterz = (1.0 * x ** 2 / sig ** 4 - 1.0 / sig ** 2) * np.exp(-1.0 * x ** 2 / (2.0 * sig ** 2))
    # fig_g, axg = plt.pyplot.subplots()
    # axg.plot(x, filterz)
    # fig_g.text(0.8, 0.8, 'sigma = %d' % sig)
    # plt.pyplot.savefig('kenerls3.png', dpi=120)
    # plt.pyplot.close()
    if series[ig].size < 1:
        return -1, -1
    else:
        f1 = np.convolve(series[ig], filterz, 'same')
    # local maximum and minimum
    # save the result as png
    return f1, ig


def find_inflect(maxtab, mintab, typez='tri', typen='sig', t_win=[60, 150]):
    """
    <discription>
        find the maximum of maxtab, minimum of mintab, depend on the season. find maximum in thawing season
    :param maxtab:
    :param mintab:
    :param typez: tri: find a thawing, a freezing and another thawing
           typez: th: find a thawing onset
           typez: fr: find a freezing onset
    :return:
    """
    onsets = []
    if typez == 'tri':
        isea1 = np.where(maxtab[:, 1] < 200)  # index of season
        min_dates = mintab[:, 1]
        isea2 = np.where((min_dates > 250) & (min_dates < 350))
        isea3 = np.where(maxtab[:, 1] > 400)

        ipeak1 = np.argmax(maxtab[isea1, 2])
        onsets.append(maxtab[ipeak1, :])

        mintab_winter = mintab[isea2]
        ipeak2 = np.argmin(mintab_winter[:, 2])
        onsets.append(mintab_winter[ipeak2])

        maxtab_summer2 = maxtab[isea3]
        ipeak3 = np.argmax(maxtab_summer2[:, 2])
        onsets.append(maxtab_summer2[ipeak3])
    elif typez == 'all':
        isea1 = np.where(maxtab[:, 1] < 200)  # index of season
        isea2 = np.where((mintab[:, 1] > 250) & (mintab[:, 1] < 350))
        isea3 = np.where((maxtab[:, 1] > 400) & (maxtab[:, 1] < 550))
        isea4 = np.where(mintab[:, 1] > 625)

        ipeak1 = np.argmax(maxtab[isea1, 2])
        onsets.append(maxtab[ipeak1, :])

        mintab_winter = mintab[isea2]
        if mintab_winter.size < 1:
            onsets.append(np.array([0, 0, 0]))
        else:
            ipeak2 = np.argmin(mintab_winter[:, 2])
            onsets.append(mintab_winter[ipeak2])

        maxtab_summer2 = maxtab[isea3]
        ipeak3 = np.argmax(maxtab_summer2[:, 2])
        onsets.append(maxtab_summer2[ipeak3])

        mintab_winter2 = mintab[isea4]
        if mintab_winter2.size < 1:
            onsets.append(np.array([0, 0, 0]))
        else:
            ipeak4 = np.argmin(mintab_winter2[:, 2])
            onsets.append(mintab_winter2[ipeak4])

    elif typez == 'fr':
        ipeak = np.argmin(mintab[:, 2])
        onsets.append(mintab[ipeak])
    elif typez == 'annual':
        if mintab.size == 0:
            mintab, maxtab = np.array([[0, 0, -99]]), np.array([[0, 0, -99]])  # no min value
        max_day, min_day = maxtab[:, 1], mintab[:, 1]
        if max_day[0] > 200:
            max_day -= 365
            min_day -= 365
        if typen == 'tb':
            thaw = find_infleclt_anual(mintab, [60, 150], 0)
            freeze = find_infleclt_anual(maxtab, [240, 335], -1)
        else:
            thaw = find_infleclt_anual(maxtab, [t_win[0], t_win[1]], -1)
            freeze = find_infleclt_anual(mintab, [240, 350], 0)
        #
        # isea1 = np.where((min_day < 135) & (min_day > 60))[0]  # for thawing index
        # isea2 = np.where((max_day > 260) & (max_day < 335))[0]  # for freezing index
        # ipeak1 = np.argsort(mintab[isea1, 2])
        # t_ind = isea1[ipeak1[0]]
        # thaw = np.mean(min_day[t_ind])
        # #
        # ipeak2 = np.argsort(maxtab[isea2, 2])
        # if ipeak2.size < 1:
        # # f_ind = isea2[ipeak2[0]]
        #     f_ind = -1
        # else:
        #     f_ind = isea2[ipeak2[-1]]
        # freeze = np.mean(max_day[f_ind])
        return [np.fix(thaw), np.fix(freeze)]
    return np.array(onsets)


def find_infleclt_anual(tabs, window, m_no):
    """
    :param tabs:
    :param window:
    :param m_no: 0 or -1
    :return:
    """
    dayz = tabs[:, 1]
    isea = bxy.gt_le(dayz, window[0], window[1], 'int')
    ipeak = np.argsort(tabs[isea, 2])
    if ipeak.size < 1:
        onset = window[-1]
    else:
        onset = np.nanmean(dayz[isea[ipeak[m_no]]])
    return onset


def find_inflect_cp(maxtab, mintab, typez='tri'):
    """
    <discription>
        find the maximum of maxtab, minimum of mintab, depend on the season. find maximum in thawing season
    :param maxtab:
    :param mintab:
    :param typez: tri: find a thawing, a freezing and another thawing
           typez: th: find a thawing onset
           typez: fr: find a freezing onset
    :return:
    """
    onsets = []
    if typez == 'tri':
        isea1 = np.where(maxtab[:, 1] < 200)  # index of season
        min_dates = mintab[:, 1]
        isea2 = np.where((min_dates > 250) & (min_dates < 350))
        isea3 = np.where(maxtab[:, 1] > 400)

        ipeak1 = np.argmax(maxtab[isea1, 2])
        onsets.append(maxtab[ipeak1, :])

        mintab_winter = mintab[isea2]
        ipeak2 = np.argmin(mintab_winter[:, 2])
        onsets.append(mintab_winter[ipeak2])

        maxtab_summer2 = maxtab[isea3]
        ipeak3 = np.argmax(maxtab_summer2[:, 2])
        onsets.append(maxtab_summer2[ipeak3])
    elif typez == 'all':
        isea1 = np.where(maxtab[:, 1] < 200)  # index of season
        isea2 = np.where((mintab[:, 1] > 250) & (mintab[:, 1] < 350))
        isea3 = np.where((maxtab[:, 1] > 400) & (maxtab[:, 1] < 550))
        isea4 = np.where(mintab[:, 1] > 625)

        ipeak1 = np.argmax(maxtab[isea1, 2])
        onsets.append(maxtab[ipeak1, :])

        mintab_winter = mintab[isea2]
        if mintab_winter.size < 1:
            onsets.append(np.array([0, 0, 0]))
        else:
            ipeak2 = np.argmin(mintab_winter[:, 2])
            onsets.append(mintab_winter[ipeak2])

        maxtab_summer2 = maxtab[isea3]
        ipeak3 = np.argmax(maxtab_summer2[:, 2])
        onsets.append(maxtab_summer2[ipeak3])

        mintab_winter2 = mintab[isea4]
        if mintab_winter2.size < 1:
            onsets.append(np.array([0, 0, 0]))
        else:
            ipeak4 = np.argmin(mintab_winter2[:, 2])
            onsets.append(mintab_winter2[ipeak4])

    elif typez == 'fr':
        ipeak = np.argmin(mintab[:, 2])
        onsets.append(mintab[ipeak])
    elif typez == 'annual':
        t_day, f_day = maxtab[:, 1], mintab[:, 1]
        if t_day[0] > 365:
            t_day -= 365
            f_day -= 365
        isea1 = np.where((t_day < 135) & (t_day > 60))[0]  # for thawing index
        isea2 = np.where((f_day > 260) & (f_day < 335))[0]  # for freezing index
        ipeak1 = np.argsort(mintab[isea1, 2])
        t_ind = isea1[ipeak1[0]]
        thaw = np.mean(t_day[t_ind])
        ipeak2 = np.argsort(maxtab[isea2, 2])
        if ipeak2.size < 1:
            # f_ind = isea2[ipeak2[0]]
            f_ind = -1
        else:
            f_ind = isea2[ipeak2[-1]]
        freeze = np.mean(f_day[f_ind])
        return [np.fix(thaw), np.fix(freeze)]
    return np.array(onsets)


def flag_miss(series, tag):
    type1 = np.array([1, 2, 3])
    if type(series) is not type1:
        series_np = np.array(series)
    w = 0

    return w


def p20_select(series):
    """
    select the mean of 10% ~ 30% as minimum; mean of 70% to 90% as maximum
    :param series:
    :return:
    """
    sz = series.size
    min = np.nanmean(series[sz / 10: sz * 3 / 10 + 1])
    max = np.nanmean(series[sz * 7 / 10: sz * 9 / 10 + 1])
    return np.array[min, max]


def mk_grid():
    return 0


def tb_1st(onset, convs):
    """
    find the zero crossing after/before thawing/freezing edge
    :param onset: the onsets detected by convs
    :param convs: the convolution result
    :return:
    """
    thaw_day = onset[0]
    window0 = np.where((convs[0] > thaw_day) & (convs[0] < 150))[0]  # ind in the window
    ind_zero = np.argmin(np.abs(convs[1][window0]))  # ind of the zero crossing
    day0 = convs[0][window0][ind_zero]  # the day when tb is lowest

    freeze_day = onset[1]
    window1 = np.where((convs[0] > 250) & (convs[0] < freeze_day))
    ind_zero = np.argmin(np.abs(convs[1][window1]))
    day1 = convs[0][window1][ind_zero]

    return np.array([day0, day1])


def edge_2nd(onset, convs):
    thaw_day = onset[0]
    window0 = np.where((convs[0] > thaw_day) & (convs[0] < 160))[0]  # ind in the window
    ind_zero = np.argmin(convs[1][window0])
    day0 = convs[0][window0][ind_zero]

    freeze_day = onset[1]
    window1 = np.where((convs[0] > freeze_day) & (convs[0] < 365))[0]
    ind_zero = np.argmax(convs[1][window1])
    day1 = convs[0][window1][ind_zero]
    return np.array([day0, day1])


def date_match(date1, date2):
    """
    find the index of elements in date2 which are also in date1
    :param date1:
    :param date2:
    :return:
    """
    imatch = np.in1d(date1, date2)

    return imatch


def edge_detection(ffnpr, t_h, typez='tri', minz=5e-4, wind=[17], figname='test_average'):
    g_size = 8
    fig = plt.pyplot.figure(figsize=[8, 8])
    pos = 510
    for w in wind:
        pos += 1
        valid_dateh = t_h[(w - 1) / 2: t_h.size - (w + 1) / 2 + 1]  # date --> smooth[valid]
        ffnpr_s, valid = n_convolve2(ffnpr, w)
        ffnpr_s = ffnpr_s[valid]  # x: valid_dateh, y: ffnpr_s[valid]
        g_npr, i_gaussian = gauss_conv(ffnpr_s)  # ffnpr, t_h
        g_npr_valid_n = (g_npr[g_size: -g_size] - np.nanmin(g_npr[g_size: -g_size])) \
                        / (np.nanmax(g_npr[g_size: -g_size]) - np.nanmin(g_npr[g_size: -g_size]))  # normalized
        max_gnpr, min_gnpr = peakdetect.peakdet(g_npr[g_size: -g_size], minz, valid_dateh[i_gaussian][g_size: -g_size])
        onset = find_inflect(max_gnpr, min_gnpr, typez=typez)
        # plot
        # plt_npr_gaussian(['NPR', valid_dateh, ffnpr_s],
        #                  [None, valid_dateh[i_gaussians][g_size: -g_size], g_nprs_valid_n],
        #                  vline=onset_s, figname='onset_based_on_nprs'+site_no+'.png')
        ax0 = fig.add_subplot(pos)
        ax0.set_ylim([0, 0.06])
        _, l0 = plot_funcs.pltyy(valid_dateh, ffnpr_s, figname, 'NPR_smoothed',
                                 t2=valid_dateh[i_gaussian][g_size: -g_size], s2=g_npr_valid_n, symbol=['k', 'g-'],
                                 handle=[fig, ax0], nbins2=6)
        # ax0.legend(l0, ['NPR', 'convolved\nNPR'], loc=2, prop={'size': 8})
        # l1, = ax1.plot(valid_dateh, ffnpr_s)
        # l1_le = plot_funcs.plt_more(ax1, valid_dateh[i_gaussian][g_size: -g_size], g_npr_valid_n, marksize=2, line_list=[l1])
        # ax1.set_ylim([200, 280])
        # ax1.locator_params(axis='y', nbins=6)
        # ax1.set_ylabel('T$_B$ (K)')
        # ax1.legend(l4_l1, ['T$_{BV}$', 'T$_{BH}$'], loc=2, prop={'size': 10})

    plt.pyplot.close()

    return 0


def output_csv(head, vars, value, fname='csv_out'):
    """

    :param head:
    :param id:
    :param value:
     the onsets and a variable, which can be kernel width, kernel size, moving average size
    :return:
    """
    value[value > 365] -= 365
    with open(fname + '.csv', 'a+') as csvfile:
        Ar = csv.reader(csvfile, delimiter=',')
        Aw = csv.writer(csvfile, delimiter=',')
        if len(list(Ar)) < 1:
            Aw.writerow(head)
        for v in value:
            vars.append(v)
        Aw.writerow(vars)
    return 0


def site_onset(onset, insitu, id, ind_bool, ind_size, buff=30):
    """
    :param onset:
        timing include the indices of in situ measurement, date of onest, and value of onset (typically is NPR)
    :param insitu:
        in situ measurement like daily temperature, daily moisture
    :return:
    """
    num = onset.shape[0]  # the number of rows (number of onsets) in onset.
    # an empty table
    atv = insitu[0][ind_bool][ind_size: -ind_size]
    lenv = atv.shape[0] - int(onset[-1, 0]) + buff
    lenv += (num - 2) * (buff * 2 + 1)
    A1 = np.zeros([len(insitu) + 1, min(lenv, (num - 1) * (buff * 2 + 1))])
    j = 1
    for att in insitu:  # include the date
        A0 = np.array([])
        for i in range(1, num, 1):
            att_valid = att[ind_bool][ind_size: -ind_size]
            A0 = np.append(A0, att_valid[int(onset[i, 0]) - buff: int(onset[i, 0]) + 1 + buff])
            # print onset[i, 0]
        A1[j] = A0
        j += 1
    for i in range(1, num, 1):  # add a onset flag, if 1 it is the onset
        A1[0][A1[1] == onset[i, 1]] = 1.11
    # set the txt write format
    # A1.shape[0]
    formt = ['%1.2f', '%1d', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.8f', '%1.8f']
    np.savetxt('onset_test' + id + '.txt', np.transpose(A1), delimiter=',', fmt=formt,
               header='onset, date, mv, Tsoil, Tair, swe, npr, npr_7day')
    return A1


def sm_onset(datez, sm, t, mode='annual'):
    """

    :param datez:
    :param sm:
    :param t:
    :param mode:
    :return:
        onset: [sm0, sm1]; onset_temperature: [t_in0, t_in1]; day_temperature2: t_out0
        where "in" means reach the transition zone, "out" means leave the zone
    """
    if mode == 'annual':
        sm_win = sm[np.where((datez > 15) & (datez < 60))]
        sm_ref = np.nanmean(sm_win)  # reference moisture: DOY 15 to 60
        if sm_ref == 0:
            sm_ref == 0.01
        i_th = np.where((datez < 150) & (datez > 75))[0]  # for thawing index
        i_frz = np.where((datez > 250) & (datez < 330))[0]

        sm_th = sm[i_th]
        d_sm0 = np.diff(sm_th)  # the daily change in thawing period
        t_th = t[i_th]
        rcs = (sm_th - sm_ref) / sm_ref  # change of soil moisture during thawing
        n = 0
        sm_window = 5
        for rc in rcs:
            if rc:
                rc_weeky, sm_weeky = rcs[n: n + 7], sm_th[n: n + sm_window]
                d_sm_weeky = d_sm0[n: n + 6]
                if sm_weeky.size < sm_window:  # array length protection, we consider the next 7 days's value after the possible onsets
                    break
                d5 = (np.max(sm_weeky) - sm_weeky[0])
                d3 = np.max(sm_weeky[0: 3]) - sm_weeky[0]
                d_sm = sm_weeky - sm_ref
                day_th = np.where(rc_weeky > 0.2)[0]
                if (d5 > 10) | (d3 > 5):  # (np.mean(sm_weeky)-sm_ref>5) (day_th.size > 3) &
                    if t[i_th[n]] < -1:
                        print 'temperature is too low for thawing: %.2f' % t[i_th[n]]
                        n += 1
                        continue
                    else:
                        break
            n += 1
        if n >= i_th.size:
            n -= 1
        print 'Thawing onset: the souil temperature is %.2f' % t[i_th[n]]
        t_0 = [-10, 10]
        weeky_insitu = np.array([datez[i_th[n] + t_0[0]: i_th[n] + t_0[1]],
                                 t[i_th[n] + t_0[0]: i_th[n] + t_0[1]],
                                 sm[i_th[n] + t_0[0]: i_th[n] + t_0[1]]])
        np.set_printoptions(precision=3, suppress=True)
        print 'Measurements in the thawing week (-3, 3): \n', \
            weeky_insitu
        onset = [datez[i_th[n]]]
        # temperature constraint
        # greater than -1 celsius for 4 or more days
        i2 = -1
        t_in = 0  # 0: tsoil is not in the (-1, 1) zone
        onset_temperature = [-1]
        day_temperature2 = [-1]  # date that tsoil leave zone and reach zone again.
        for t0 in t_th:
            i2 += 1
            if (t0 > -1) & (t_in == 0):
                # check if the date is beyond the thawing period
                if i2 + 3 > t_th.size:
                    print 'cannot find the -1 degC timing'
                elif np.mean(t_th[i2: i2 + 4]) > -1:  # 4-day > -1 degC
                    onset_temperature = [datez[i_th[i2]]]
                    t_in = 1
            elif t0 > 1:
                if i2 + 3 > t_th.size:
                    print 'cannot find the 1 degC timing'
                elif np.mean(t_th[i2: i2 + 1]) > 0:
                    day_temperature2 = [datez[i_th[i2]]]
                    t_in = 0
                    break


        # freezing detect
        sm_fr = sm[i_frz]
        t_fr = t[i_frz]
        rcd = (sm_fr[1:] - sm_fr[0: -1]) / sm_fr[0: -1]
        frz_time = 5
        n = 0
        for rc in rcd:
            if t_fr[n] < 1:
                if rc < -0.05:
                    rc_weeky, sm_weeky = rcd[n: n + 7], sm_fr[n: n + frz_time]
                    print 'sm change is %.1f - %.1f' % (sm_fr[n + 1], sm_fr[n])
                    for sm_i in sm_weeky:
                        print sm_i
                    day_frz = np.where(rc_weeky < 0.01)[0]
                    d3 = sm_weeky[0] - np.min(sm_weeky[0: 3])
                    print sm_weeky.size
                    d10 = sm_weeky[0] - np.min(sm_weeky)
                    if (sm_weeky[0] < 10) | (
                        np.mean(sm_fr[n:]) < 10):  # ((day_frz.size>3) | (np.mean(sm_fr[n+1: n+8]) < 10)):
                        # ((d10 > 10)|(sm_weeky[0]<10))&(np.mean(sm_fr[n: ]) < 10):
                        if t[i_frz[n + 1]] > 1:
                            print 'the soil temperature is two high: %f.1' % t[i_frz[n + 1]]
                            n += 1
                            continue
                        else:
                            break
                            # elif (t[i_frz[n+1]] < 1)&((sm[i_frz[n+1]] - sm[i_frz[n+7]]) > 10):
                            #     print t[n], rcd[n], sm[n]
                            #     break

            n += 1

        if n + 1 >= i_frz.size:
            n -= 1
        onset.append(datez[i_frz[n + 1]])
        # temperature constraint
        # greater than -1 celsius for 4 or more days
        i2 = -1

        t_frz = t[i_frz]
        for t0 in t_frz:
            i2 += 1
            if t0 < 1:
                # check if the date is beyond the thawing period
                if i2 + 3 > t_frz.size:
                    print 'cant find the -1 degC timing'
                elif all(t_frz[i2: i2 + 3] < 1):  # 4-day > -1 degC
                    onset_temperature.append(datez[i_frz[i2]])
                    break
    return onset, onset_temperature, day_temperature2


def h5_write(fname, keyz, valz, groupz=['12_5km']):
    h0 = h5py.File(fname, 'a')
    h1 = h0.create_group(groupz[0])
    for i in range(0, len(keyz), 1):
        h1[keyz[i]] = valz[i]
    h0.close()
    return 0


def angle_compare(sigma_1, fname):
    # out = read_site.fix_angle_ascat(txtname)
    #
    # orb_no = np.array(out[4])
    # ob_as = np.where(orb_no<0.5)
    # ob_des = np.where(orb_no>0.5)
    # x_time = np.array(out[0])[ob_as]
    # x_time_np = np.array(x_time)
    # sigma0_np = np.array(np.array(out[2])[ob_as])  # mid-observation
    # if sigma0_np[0] < -1e5:
    #     sigma0_np *= 1e-6
    # # edge detect
    # g_size = 8
    # g_sig, ig2 = data_process.gauss_conv(sigma0_np, sig=3)  # non-smoothed
    # g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
    #                 /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
    #
    # # get the peak
    # max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, x_time_np[g_size: -g_size])
    # #print txt_file, n_warning
    # onset = data_process.find_inflect(max_gsig_s, min_gsig_s, typez='annual')

    # [[x_time, sigma0_np], [x_time_np[ig2][g_size: -g_size], g_sig_valid]]
    plot_funcs.pltyy(sigma_1[0][0], sigma_1[0][1], fname, '$\sigma^0$',
                     t2=sigma_1[1][0], s2=sigma_1[1][1], label_y2='E(t)',
                     symbol=['k', 'g-'], nbins2=6, ylim=[-15, -7])


def time_mosaic(intervals, pass_h):
    id_time = []
    for interv in intervals:
        id_time.append((pass_h > interv[0]) & (pass_h < interv[1]))
    return id_time


def pass_zone_plot(input_x, input_y, value, pre_path, fname=[], z_min=-20, z_max=-4, prj='merc', odd_points=False,
                   odd_index=False, title_str=' ', txt=False):
    """
    :param input_x:
    :param input_y:
        the cordinate value, x is longitude and y is latitude, in 2d grid system
    :param value: the value to be plot on map
    :param odd_points: the lat/lon of pixels of interest, marked in the map
    :return:
    """
    fig = plt.pyplot.figure(figsize=[8, 8])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    lons_1d, lats_1d, value_1d = input_x.ravel(), input_y.ravel(), value.ravel()
    ax.set_title(title_str)
    if prj == 'merc':
        m = Basemap(llcrnrlon=-170, llcrnrlat=54, urcrnrlon=-140, urcrnrlat=72,
                    resolution='i', area_thresh=1000., projection=prj,
                    lat_ts=50., ax=ax)
    elif prj == 'aea':
        # m = Basemap(width=3e6, height=3e6, resolution='l', projection=prj, lat_ts=62, lat_0=62, lon_0=-150., ax=ax)
        m = Basemap(width=15e5, height=2e6, resolution='i', projection=prj, lat_1=55, lat_2=65, lon_0=-154., lat_0=63,
                    ellps='WGS84', ax=ax)
    elif prj == 'ease':
        m = Basemap(llcrnrlon=-170, llcrnrlat=54, urcrnrlon=-140, urcrnrlat=72, resolution='i', area_thresh=10000,
                    lat_ts=30, projection='cea')
    # m.fillcontinents(color='coral', lake_color='black')
    im = m.pcolormesh(input_x, input_y, value, vmin=z_min, vmax=z_max, latlon=True)
    # m.drawmapboundary(fill_color='aqua')
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='cyan')
    print 'water color set'
    m.drawcoastlines()
    m.drawcountries()
    parallel = np.arange(50, 80, 5)
    meridian = np.arange(-170, -130, 5)
    m.drawparallels(parallel, labels=[1, 0, 0, 0], fontsize=18)
    m.drawmeridians(meridian, labels=[0, 0, 0, 1], fontsize=18)
    cb = m.colorbar(im)
    if fname == 'thaw_14_7' or fname == 'freeze_14_7':
        cb.set_label('Days')
    else:
        cb.set_label('DOY 2016')

    # mask the ocean

    ##getting the limits of the map:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    map_edges = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    ##getting all polygons used to draw the coastlines of the map
    polys = [p.boundary for p in m.landpolygons]

    ##combining with map edges
    polys = [map_edges] + polys[:]

    ##creating a PathPatch
    codes = [
        [Path.MOVETO] + [Path.LINETO for p in p[1:]]
        for p in polys
        ]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [c for cs in codes for c in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path, facecolor='white', lw=0)

    ##masking the data:
    ax.add_patch(patch)


    # if len(odd_points) > 0:
    if odd_points is not False:
        if odd_points.size < 100:
            if type(odd_points[0]) is list:
                x = np.array([odd_points[i][0] for i in range(0, len(odd_points))])
                y = np.array([odd_points[i][1] for i in range(0, len(odd_points))])
                m.scatter(odd_points[0], odd_points[1], marker='x', color='k', latlon=True)
            elif odd_points.ndim > 1:
                print 'the odd point size is ', odd_points.size
                for i_p0, od_p0 in enumerate(odd_points):
                    index = odd_index[i_p0]
                    # x, y = od_p0[0], od_p0[1]
                    x, y = lons_1d[index], lats_1d[index]
                    m.scatter(x, y, 2, marker='.', color='k', latlon=True)
            else:
                x, y = odd_points[0], odd_points[1]
                m.scatter(x, y, 20, marker='*', color='k', latlon=True)
            if txt is not False:
                print 'add the annotation to each odd points'
                for i0 in range(0, odd_points.shape[0]):  # txt.size
                    # x00, y00 = m(lons_1d[txt[i0]], lats_1d[txt[i0]])
                    x00, y00 = m(odd_points[i0][0], odd_points[i0][1])
                    annotations = '%s' % (txt[i0])  # value_1d[txt[i0]]
                    # m.scatter(lons_1d[txt[i0]], lats_1d[txt[i0]], marker='x', color='k', latlon=True)
                    # m.scatter(odd_points[i0][0], odd_points[i0][1], marker='.', color='k', latlon=True)
                    # print 'pixel %d located at: %d' % (i0, txt[i0]), lons_1d[txt[i0]], lats_1d[txt[i0]]
                    pylt.annotate(annotations, xy=(x00, y00), xytext=(x00 + 0.2, y00), xycoords='data',
                                  textcoords='data',
                                  fontsize=6)
                    # m.scatter(x[-7:], y[-5:], marker='s', color='k', latlon=True)
        elif odd_points.size == 2:
            x = odd_points[1]
            y = odd_points[0]
            m.scatter(x, y, marker='x', color='k', latlon=True)
            print 'odd point: ', x, y
        else:
            for s0 in odd_points:
                x, y = s0[-2], s0[-3]
                m.scatter(x, y, marker='x', color='k', latlon=True)
    plt.rcParams.update({'font.size': 18})
    plt.pyplot.savefig(pre_path + fname + '.png', dpi=300)
    # add_site(m, pre_path+'spatial_ascat'+fname+'_site')
    plt.pyplot.close()
    return 0


def add_site(m, fname):
    site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968', '1090', '1175', '1177', '2210',
                '1089', '1233', '2212', '2211']
    s_lat, s_lon, s_name = [], [], []
    for no in site_nos:
        s_name.append(site_infos.change_site(no)[0])
        s_lat.append(site_infos.change_site(no)[1])
        s_lon.append(site_infos.change_site(no)[2])
    m.scatter(s_lon, s_lat, 5, marker='*', color='k', latlon=True)
    plt.pyplot.savefig(fname + '.png', dpi=120)


def ascat_nn(x, y, z, orb, site_no, disref=0.5, f=None, center=False, pass_sec=False):
    """
    return the indices of pixels within a distance from centroid to site.
    :return: min_inds: 0: id of NN at ascending, 1: id of NN at des
    """
    f_bad = np.where(f > 0)
    if any(f_bad[0] > 0):
        print 'not usable at %s' % site_no
    if center is not False:
        s_info = [str(site_no), 0, 0]
        s_info[1], s_info[2] = center[0], center[1]
    else:
        s_info = site_infos.change_site(site_no)
    min_inds = [[], []]  # 0: ascending neighbour, 1: descending
    dis_list = [[], []]
    for orbit in [0, 1]:
        i_orb = np.where(orb == orbit)
        if i_orb[0].size > 0:
            # xo, yo, secs = x[i_orb], y[i_orb], pass_sec[i_orb]
            xo, yo = x[i_orb], y[i_orb]
            # based on different passing time:

            dis = bxy.cal_dis(s_info[1], s_info[2], yo, xo)
            min_ind = dis < disref
            min_inds[orbit] = i_orb[0][min_ind]
            dis_list[orbit] = dis[min_ind]
            if sum(min_ind) < 1:
                print 'No neighbour data in %d orbit for site %s' % (orbit, s_info[0])
                continue
    orbname = ['AS', 'DES']
    for i in [0]:
        print 'Distance', orbname[i], ':', dis_list[i]
    # return min_inds, np.append(dis_list[0], dis_list[1])
    return min_inds, dis_list


def ascat_ipt(dis, value, time, orb):
    """
    :param dis:
    :param value:
    :param time:
    :return: the interpolated values (e.g.sigma, incidence) and the corresponded pass hours
    """
    ipt_out = []
    t_order = np.argsort(time)
    # ordered input
    time2 = np.sort(time)
    value2 = value[:, t_order]
    dis2 = dis[t_order]
    orb = orb[t_order]
    i = 0
    for t0 in time2:
        if i >= time2.size:
            break
        t_win = [time2[i], time2[i] + 1]
        i_win = (time2 >= t_win[0]) & (time2 < t_win[1])
        if sum(i_win) < 2:
            v = [var[i_win][0] for var in value2]
            ipt_out.append([time2[i]] + v + [orb[i]])
        else:
            v = []
            for var in value2:
                v.append(bxy.dis_inter(dis2[i_win], var[i_win]))
            ipt_out.append([time2[i]] + v + [orb[i]])
        i += sum(i_win)
    return ipt_out


def ascat_ang_norm():
    siteno = ['947',
              '1175', '950', '2065', '967', '2213', '949', '950', '960', '962', '968', '1090', '1177', '2081', '2210',
              '1089', '1233', '2212', '2211']
    angle_range = np.arange(25, 65, 0.1)
    prefix = site_infos.get_data_path('_07_01')
    for site in siteno:
        txtname = './result_05_01/ascat_point/ascat_s' + site + '_2016.npy'
        txt_table = np.load(txtname)
        id_orb = txt_table[:, -1] == 0
        out = txt_table[id_orb, :].T
        xt, y1t, y2t = np.array(out[0]), np.array(out[1]), np.array(out[2])
        x_inc, y_sig = out[5: 8], out[2: 5]
        tx = out[0]
        # 4 periods
        p1, p2, p3, p4 = tx < 80, (tx > 90) & (tx < 120), (tx > 160) & (tx < 250), tx > 270
        p_no = 0
        p_con = p1 | p3
        for p in [p1, p2, p3, p4, p_con]:
            p_no += 1
            fig = pylt.figure(figsize=[4, 3])
            ax = fig.add_subplot(111)
            inci1 = x_inc[:, p]
            sig1 = y_sig[:, p]
            if any(inci1[0] > 1e2):
                inci1 *= 1e-2
            if any(sig1[0] < -1e4):
                sig1 *= 1e-6
            ax.plot(inci1[0], sig1[0], 'bo')
            plot_funcs.plt_more(ax, inci1[1], sig1[1])
            plot_funcs.plt_more(ax, inci1[2], sig1[2], symbol='go')
            # linear regression
            x = inci1.reshape(1, -1)[0]
            y = sig1.reshape(1, -1)[0]
            a, b = np.polyfit(x, y, 1)
            f = np.poly1d([a, b])
            # r squared
            y_mean = np.sum(y) / y.size
            sstot = np.sum((y - y_mean) ** 2)
            ssres = np.sum((y - f(x)) ** 2)
            r2 = 1 - ssres / sstot
            fig.text(0.15, 0.15, '$y = %.2f x + %.f$\n $r^2 = %.4f$' % (a, b, r2))
            plot_funcs.plt_more(ax, x, f(x), symbol='r-', fname=prefix + 'Incidence_angle_p' + str(p_no))
            fig.clear()
            pylt.close()
        fig2 = pylt.figure(figsize=[4, 3])
        ax = fig2.add_subplot(111)
        n = 0
        for p in [p1, p3]:
            inci1 = x_inc[:, p]
            sig1 = y_sig[:, p]
            if any(inci1[0] > 1e2):
                inci1 *= 1e-2
            if any(sig1[0] < -1e4):
                sig1 *= 1e-6
            # linear regression
            x = inci1.ravel()
            y = sig1.ravel()
            a, b = np.polyfit(x, y, 1)
            f = np.poly1d([a, b])
            # r squared
            y_mean = np.sum(y) / y.size
            sstot = np.sum((y - y_mean) ** 2)
            ssres = np.sum((y - f(x)) ** 2)
            r2 = 1 - ssres / sstot
            if n < 1:
                ax.plot(inci1[0], sig1[0], 'ro', markersize=5)
                plot_funcs.plt_more(ax, inci1[1], sig1[1], symbol='go', marksize=5)
                plot_funcs.plt_more(ax, inci1[2], sig1[2], symbol='bo', marksize=5)
                # ax.set_ylim([-16, -6])
                plot_funcs.plt_more(ax, angle_range, f(angle_range), symbol='r-', marksize=5)
                fig2.text(0.15, 0.15, '$y = %.2f x + %.f$\n $r^2 = %.4f$' % (a, b, r2))
            else:
                plot_funcs.plt_more(ax, inci1[0], sig1[0], symbol='r^', marksize=5)
                plot_funcs.plt_more(ax, inci1[1], sig1[1], symbol='g^', marksize=5)
                plot_funcs.plt_more(ax, inci1[2], sig1[2], symbol='b^', marksize=5)
                plot_funcs.plt_more(ax, angle_range, f(angle_range), symbol='b-', marksize=5)
                fig2.text(0.45, 0.75, '$y = %.2f x + %.f$\n $r^2 = %.4f$' % (a, b, r2))
            n += 1
        ax.set_ylim([-18, -4])
        pylt.savefig(prefix + 'Incidence_angle_' + site + '.png', dpi=120)
        pylt.close()
    return 0


def ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g=5, pp=False, norm=False,
                      txt_path='./result_05_01/ascat_point/', is_sub=False, order=1, sate='B', daily_mean=True
                      , min_dis=19, time_window=[], x_unit='day'):  # orbit default is 0, ascending, night pass
    # initial assign
    total_days = 500
    print 'the station number is', site_no
    # read series sigma
    site = site_no
    site_nos = site_infos.get_id(site, mode='single')
    for si0 in site_nos:
        if is_sub is False:
            txtname = glob.glob(txt_path + 'ascat_s' + si0 + '*' + sate + '.npy')[0]
            txtname_B = glob.glob(txt_path + 'ascat_s' + si0 + '*' + 'B' + '.npy')[0]
            txtname_A = glob.glob(txt_path + 'ascat_s' + si0 + '*' + 'A' + '.npy')[0]
            # txtname = glob.glob(txt_path+'ascat_s'+si0+'*')[0]
        # txtname = './result_05_01/ascat_point/ascat_s'+si0+'_2016.npy'
        print 'the npy was read: ', txtname
        ascat_all = np.load(txtname)
        if ascat_all.size < 1:
            return [-1, -1, -1], [-1, -1, -1], [-1, -1], -1, [-1, -1], [-1, -1]
        if orb_no > 1:
            id_orb = ascat_all[:, -2] < orb_no
        else:
            id_orb = ascat_all[:, -2] == orb_no
        ascat_ob = ascat_all[id_orb]
        # READ THE DATA 08/22/2018
        sig_m = ascat_ob[:, 3]
        inc_m = ascat_ob[:, 9]

        # check the sampling rate
        ascat_ob[:, 14]
        # newly updated 0515/2018, 08/21/2018
        times_ascat = bxy.time_getlocaltime(ascat_ob[:, 14], ref_time=[2000, 1, 1, 0])
        # time_0 = times_ascat[:, 0]
        # set a initial timing (unit: s)
        utc_time = ascat_ob[:, 14]
        sec_end = np.max(utc_time)
        local_time0 = bxy.time_getlocaltime([np.min(utc_time)], ref_time=[2000, 1, 1, 0])
        # local_time1 = bxy.time_getlocaltime([np.max(utc_time)], ref_time=[2000, 1, 1, 0])
        if orb_no == 0:  # ascending
            ini_hr = 18
        else:
            ini_hr = 4
        sec_ini = bxy.get_secs([local_time0[0], local_time0[1], local_time0[2], ini_hr, 0, 0], [2000, 1, 1, 0, 0])
        # sec_end = bxy.get_secs([local_time1[0], local_time1[1], local_time1[2], ini_hr, 0, 0], [2000, 1, 1, 0, 0])
        # sec_ini = np.min(utc_time) - 4*3600
        sec_span = 8 * 3600
        sec_step = 8 * 3600
        sigma_out = np.zeros((3, sig_m.size)) - 999
        i2 = 0
        is_mean = daily_mean
        write_no = 0
        # initial a time series array, change sigma_out
        series0 = bxy.get_secs([local_time0[0], local_time0[1], local_time0[2], 0, 0, 0], [2000, 1, 1, 0, 0])
        series_sec = np.arange(series0, series0 + total_days * 24 * 3600, 24 * 3600)
        sigma_out = np.zeros([3, total_days * 10]) - 999
        sigma_out[0, 0: series_sec.size] = series_sec  # daily mean, the x axis is integral day with unit of secs
        while sec_ini < sec_end:
            t_current = bxy.time_getlocaltime([sec_ini, sec_ini + sec_span], ref_time=[2000, 1, 1, 0])
            daily_idx = (utc_time > sec_ini) & (utc_time < sec_ini + sec_span) & (ascat_ob[:, -1] < min_dis) \
                        & (ascat_ob[:, 6] < 2) & (ascat_ob[:, 9] > 30)
            daily_sigma, daily_sec, dis_daily, daily_inc = \
                sig_m[daily_idx], utc_time[daily_idx], ascat_ob[:, -1][daily_idx], inc_m[daily_idx]
            t_temp = bxy.time_getlocaltime(daily_sec, ref_time=[2000, 1, 1, 0])

            # write no data day
            if t_temp.size < 1:
                fname = 'ascat_no_data_%s.txt' % si0
                with open(fname, 'a-') as writer0:
                    if write_no < 1:
                        writer0.writelines('%s \n' % txtname)
                        write_no += 1
                        writer0.writelines('no thaw onset was find at: %d \n' % t_current[-2][0])
                i2 += 1
            else:
                if is_mean is True:
                    for t0 in [0]:
                        # value0 = np.mean(daily_sigma[t_temp[-1] == t0])
                        # value1 = np.mean(daily_inc[t_temp[-1] == t0])
                        # doy0_new = np.mean(t_temp[-2][t_temp[-1] == t0]) + t0/24.0
                        value0, value1, doy0_new = np.mean(daily_sigma), np.mean(daily_inc), np.mean(daily_sec)
                        # sigma_out[0, i2], sigma_out[1, i2], sigma_out[2, i2] = doy0_new, value0, value1
                        sigma_out[1, i2], sigma_out[2, i2] = value0, value1
                        i2 += 1
                else:
                    # re-sampled hourly
                    u_v, u_i = np.unique((daily_sec / 3600).astype(int), return_index=True)  # seconds integral hour
                    temp_v = np.zeros([u_i.size, 3]) - 999
                    for i3 in range(0, u_i.size):
                        temp_v[0, i3] = u_v[i3] * 3600
                        if i3 < u_i.size - 1:
                            temp_v[i3, 1], temp_v[i3, 2] \
                                = np.mean(daily_sigma[u_i[i3]: u_i[i3 + 1]]), np.mean(daily_inc[u_i[i3]: u_i[i3 + 1]])
                        else:
                            temp_v[i3, 1], temp_v[i3, 2] \
                                = np.mean(daily_sigma[u_i[i3]:]), np.mean(daily_inc[u_i[i3]:])
                        sigma_out[:, i2] = temp_v[i3]
                        i2 += 1
            sec_ini += sec_step
        out_valid = (sigma_out[0] > -999) & (sigma_out[1] > -999) & (sigma_out[2] > -999)
        # angular normalization
        # tx = doy_tp[1]+1 -365
        # tx_tuple = bxy.time_getlocaltime(ascat_ob[p, 5: 8], ref_time=[2000, 1, 1, 0])
        plot_funcs.quick_plot(sigma_out[0, out_valid], sigma_out[1, out_valid], lsymbol='-o')
        ascat_sec = ascat_ob[:, 14]
        win0 = bxy.get_secs([2016, 1, 1, 0, 0, 0], [2000, 1, 1, 0, 0, 0])
        win1 = bxy.get_secs([2016, 3, 1, 0, 0, 0], [2000, 1, 1, 0, 0, 0])
        p = (ascat_sec > win0) & (ascat_sec < win1)
        x, y = ascat_ob[p, 8: 11].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
        a, b = np.polyfit(x, y, 1)
        print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)

        # angular dependency for each ob mode
        # p_win, p_su = (tx>0) & (tx<60), (tx>150)&(tx<260)
        # x_trip_win, x_trip_su = ascat_ob[p_win, 5: 8], ascat_ob[p_su, 5: 8]
        # y_trip_win, y_trip_su = ascat_ob[p_win, 2: 5], ascat_ob[p_su, 2: 5]
        # a_array, b_array, d_array = np.zeros(6)-1, np.zeros(6)-1, np.zeros(3) - 99
        # for i1 in range(0, 3):
        #     a_w, b_w = np.polyfit(x_trip_win[:, i1], y_trip_win[:, i1], 1)
        #     a_s, b_s = np.polyfit(x_trip_su[:, i1], y_trip_su[:, i1], 1)
        #     a_array[i1], b_array[i1] = a_w, b_w
        #     a_array[i1+3], b_array[i1+3] = a_s, b_s
        #     d_array[i1] = np.mean(y_trip_su[:, i1]) - np.mean(y_trip_win[:, i1])
        # a_coef_name = 'ascat_linear_a.txt'
        # b_coef_name = 'ascat_linear_b.txt'
        # with open(a_coef_name, 'a') as t_file:
        #     t_file.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, a_array[0], a_array[1], a_array[2],
        #                                                       a_array[3], a_array[4], a_array[5], d_array[0], d_array[1], d_array[2]))
        # with open(b_coef_name, 'a') as t_file:
        #     t_file.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, b_array[0], b_array[1], b_array[2],
        #                                                       b_array[3], b_array[4], b_array[5], d_array[0], d_array[1], d_array[2]))

        # sig_m = ascat_ob[:, 2]  # changed using the forwards mode to increase the winter summer difference
        # inc_m = ascat_ob[:, 5]
        sig_m, inc_m = sigma_out[1, out_valid], sigma_out[2, out_valid]  # 2018
        sig_mn = sig_m - (inc_m - 45) * a

        # edge detect
        tx = sigma_out[0, out_valid]
        sig_g = sigma_g  # gaussian stds
        g_size = 6 * sig_g / 2
        if order == 1:
            g_sig, ig2 = gauss_conv(sig_mn, sig=sig_g, size=2 * g_size + 1)  # non-smoothed
        elif order == 2:
            g_sig, ig2 = gauss2nd_conv(sig_mn, sig=sig_g)
        g_sig_valid = 2 * (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size])) \
                      / (np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size])) - 1
        g_sig_valid_non = g_sig[g_size: -g_size]  # NON-normalized
        if norm == True:
            g_sig_valid = g_sig[g_size: -g_size]  # NON-normalized
        max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid_non, 1e-1, tx[g_size: -g_size])
        onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual')
        t_tuples_out = bxy.time_getlocaltime(tx, ref_time=[2000, 1, 1, 0])
        # the return values
        conv = [tx[ig2][g_size: -g_size], g_sig_valid, g_sig_valid_non]

        # change the unit of x axis into doy 2016
        if x_unit == 'sec':
            print 'change x_unit to ', x_unit
            temp = bxy.time_getlocaltime(tx, ref_time=[2000, 1, 1, 0])
            leap_num = temp[0] - 2016
            leap_num[leap_num < 0] = 0
            doy_2016 = temp[-2] + (temp[0] - 2016) * 365 + leap_num
            tx = doy_2016

        # set a period
        if len(time_window) > 0:
            win_valid2 = (conv[0] > time_window[0]) & (conv[0] < time_window[1])
            win_valid = (tx > time_window[0]) & (tx < time_window[1])

            return [conv[0][win_valid2], conv[1][win_valid2], conv[2][win_valid2]], \
                   [tx[win_valid], sig_mn[win_valid], inc_m[win_valid]], \
                   onset, \
                   t_tuples_out[-1], \
                   [t_tuples_out[-2] + 365, t_tuples_out[-1]], \
                   [max_gsig_s, min_gsig_s]
        else:
            return [tx[ig2][g_size: -g_size], g_sig_valid, g_sig_valid_non], \
                   [tx, sig_mn, inc_m], \
                   onset, \
                   t_tuples_out[-1], \
                   [t_tuples_out[-2] + 365, t_tuples_out[-1]], \
                   [max_gsig_s, min_gsig_s]


def smap_melt_initiation(npr_series, time_secs, winter_zone, summer_zone, year0, gk=10, one_pixel_return=False):
    npr_series[npr_series < 0] = -9999
    conv_npr_pixel, thaw_secs_npr, maximum, minimum = get_onset(time_secs, npr_series, year0=year0,
                                              thaw_window=[
                                                  bxy.get_total_sec('%d0101' % year0, reftime=[2000, 1, 1, 12]) +
                                                  doy0 * 3600 * 24 for doy0 in [60, 150]],
                                              k=gk, type='npr')
    melt_signal0 = npr_series[time_secs == thaw_secs_npr]
    # smap_pack = np.array([time_secs, npr_series, conv_npr_pixel, maximum, minimum])
    if npr_series.size < 1:
        mean_winter, mean_summer = -999
    else:
        if np.isinf(npr_series).any():
            print 'inf values in npr series'
        npr_series[np.isnan(npr_series)] = -999
        mean_winter = np.nanmean(npr_series[(time_secs > winter_zone[0]) &
                                            (time_secs < winter_zone[1]) & (npr_series > 0)])
        mean_summer = np.nanmean(npr_series[(time_secs > summer_zone[0]) &
                                            (time_secs < summer_zone[1]) & (npr_series > 0)])
    # data_process.zero_find([s_secs, s_measurements[:, 0, i_no, 0].ravel()], th=5)
    peak_sec = zero_find(conv_npr_pixel, th=5)
    peak_signal = npr_series[time_secs == peak_sec]
    if one_pixel_return:
        # return conv_npr_pixel, thaw_secs_npr, melt_signal0, np.array([mean_winter, mean_summer, peak_signal])
        return conv_npr_pixel, thaw_secs_npr, maximum, minimum, np.array([mean_winter, mean_summer, peak_signal])
    else:
        return 0, thaw_secs_npr, melt_signal0, np.array([mean_winter, mean_summer, peak_signal])


def ascat_melt(times0, sigma0_correct, thaw_window):
    max_value_ascat, min_value_ascat_na, conv_ascat \
        = test_def.edge_detect(times0, sigma0_correct,
                               10, seriestype='sig', is_sort=False)
    max_value_ascat_na, min_value_ascat, conv_ascat_na \
        = test_def.edge_detect(times0, sigma0_correct,
                               7, seriestype='sig', is_sort=False)
    sigma0_pack = np.array([times0, sigma0_correct, conv_ascat])
    thaw_ascat = max_value_ascat[(max_value_ascat[:, 1] > thaw_window[0]) &  # column 1 [:, 1] the secs of an edge
                                 (max_value_ascat[:, 1] < thaw_window[1])]
    return sigma0_pack


def get_ascat_series(site_no, orb_no=0, inc_plot=False, sigma_g=5, pp=False, norm=False,
                     txt_path='./result_05_01/ascat_point/', is_sub=False,
                     order=1):  # orbit default is 0, ascending, night pass
    """
    added 2018/06/14
        return ascat time series from the ascat .npy files, number of attributes: 14, passing seconds included
    :param site_no:
    :param orb_no:
    :param inc_plot:
    :param sigma_g:
    :param pp:
    :param norm:
    :param txt_path:
    :param is_sub:
    :param order:
    :return:
    """
    # read series sigma
    site = site_no
    site_nos = site_infos.get_id(site, mode='single')
    for si0 in site_nos:
        if is_sub is False:
            txtname = glob.glob(txt_path + 'ascat_s' + si0 + '*')[0]
        # txtname = './result_05_01/ascat_point/ascat_s'+si0+'_2016.npy'
        ascat_all = np.load(txtname)
        if ascat_all.size < 1:
            return [-1, -1, -1], [-1, -1, -1], [-1, -1], -1, [-1, -1], [-1, -1]
        id_orb = ascat_all[:, -1] == orb_no
        ascat_ob = ascat_all[id_orb]

        # # transform utc time to local time
        # sec_ascat = bxy.timetransform(ascat_ob[:, 1], '20000101 00:00:00', '%Y%m%d %X', tzone=True)
        # doy_tp = np.modf((sec_ascat)/3600.0/24.0)
        # doy = doy_tp[1]+1
        # passhr = np.round(doy_tp[0]*24.0)

        # newly updated 0515/2018
        times_ascat = bxy.time_getlocaltime(ascat_ob[:, 1], ref_time=[2000, 1, 1, 0])
        doy_tp2 = times_ascat[-2] + (times_ascat[0] - 2015) * 365
        passhr = times_ascat[-1] * 1.0

        # angular normalization
        # tx = doy_tp[1]+1 -365
        tx = doy_tp2 + 1 - 365
        p = (tx > 0) & (tx < 60)
        x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
        a, b = np.polyfit(x, y, 1)
        print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)
        f = np.poly1d([a, b])

        # angular dependency for each ob mode
        p_win, p_su = (tx > 0) & (tx < 60), (tx > 150) & (tx < 260)
        x_trip_win, x_trip_su = ascat_ob[p_win, 5: 8], ascat_ob[p_su, 5: 8]
        y_trip_win, y_trip_su = ascat_ob[p_win, 2: 5], ascat_ob[p_su, 2: 5]
        a_array, b_array, d_array = np.zeros(6) - 1, np.zeros(6) - 1, np.zeros(3) - 99
        for i1 in range(0, 3):
            a_w, b_w = np.polyfit(x_trip_win[:, i1], y_trip_win[:, i1], 1)
            a_s, b_s = np.polyfit(x_trip_su[:, i1], y_trip_su[:, i1], 1)
            a_array[i1], b_array[i1] = a_w, b_w
            a_array[i1 + 3], b_array[i1 + 3] = a_s, b_s
            d_array[i1] = np.mean(y_trip_su[:, i1]) - np.mean(y_trip_win[:, i1])
        a_coef_name = 'ascat_linear_a.txt'
        b_coef_name = 'ascat_linear_b.txt'
        # with open(a_coef_name, 'a') as t_file:
        #     t_file.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, a_array[0], a_array[1], a_array[2],
        #                                                       a_array[3], a_array[4], a_array[5], d_array[0], d_array[1], d_array[2]))
        # with open(b_coef_name, 'a') as t_file:
        #     t_file.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, b_array[0], b_array[1], b_array[2],
        #                                                       b_array[3], b_array[4], b_array[5], d_array[0], d_array[1], d_array[2]))

        sig_m = ascat_ob[:, 3]
        inc_m = ascat_ob[:, 6]
        # sig_m = ascat_ob[:, 2]  # changed using the forwards mode to increase the winter summer difference
        # inc_m = ascat_ob[:, 5]
        sig_mn = sig_m - (ascat_ob[:, 6] - 45) * a
        # daily average:
        tdoy = tx
        u_doy = np.unique(tdoy)
        sig_d, i0 = np.zeros([u_doy.size, 2]), 0
        pass_hr_d = np.zeros(u_doy.size)
        inc_d = np.zeros(u_doy.size)
        for td in u_doy:
            sig_d[i0][0] = td
            sig_d[i0][1] = np.mean(sig_mn[tdoy == td])
            inc_d[i0] = np.mean(inc_m[tdoy == td])  # daily incidence angle
            pass_hr_d[i0] = np.mean(passhr[tdoy == td])
            i0 += 1
        tx = sig_d[:, 0]
        sig_mn = sig_d[:, 1]
        pass_hr_d = np.round(pass_hr_d)
        # one more constraints, based on incidence angle
        # id_inc = bxy.gt_le(inc_d, 30, 35)
        # tx, sig_mn = tx[id_inc], sig_mn[id_inc]
        sig_d[:, 0] -= 1
        sig_d[:, 0] += 365
        return sig_d, pass_hr_d


def ascat_order2(site_no, orb_no=0, inc_plot=False, sigma_g=5, pp=False, norm=False,
                 txt_path='./result_05_01/ascat_point/', is_sub=False):
    # read sigmas
    txtname = glob.glob(txt_path + 'ascat_s' + site_no + '*')[0]
    ascat_all = np.load(txtname)
    if ascat_all.size < 1:
        return [-1, -1, -1], [-1, -1, -1], [-1, -1], -1, [-1, -1], [-1, -1]
    id_orb = ascat_all[:, -1] == orb_no
    ascat_ob = ascat_all[id_orb]
    # transform utc time to local time
    doy_passhr = bxy.time_getlocaltime(ascat_all, 0)
    doy = doy_passhr[1]
    doy[doy_passhr[0] < 2016] -= 365
    passhr = doy_passhr[2]
    # angular normalization
    tx = doy
    p = (tx > 20) & (tx < 90)
    x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
    a, b = np.polyfit(x, y, 1)
    print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)
    f = np.poly1d([a, b])

    sig_m = ascat_ob[:, 3]
    inc_m = ascat_ob[:, 6]
    sig_mn = sig_m - (ascat_ob[:, 6] - 45) * a
    # daily average:
    tdoy = tx
    u_doy = np.unique(tdoy)
    sig_d, i0 = np.zeros([u_doy.size, 2]), 0  # doy and mean sigma
    pass_hr_d = np.zeros(u_doy.size)  # mean pass time
    inc_d = np.zeros(u_doy.size)  # mean incidence
    for td in u_doy:
        sig_d[i0][0] = td
        sig_d[i0][1] = np.mean(sig_mn[tdoy == td])
        inc_d[i0] = np.mean(inc_m[tdoy == td])
        pass_hr_d[i0] = np.mean(passhr[tdoy == td])
        i0 += 1
    tx = sig_d[:, 0]
    sig_mn = sig_d[:, 1]
    pass_hr_d = np.round(pass_hr_d)

    # edge detect
    sig_g = sigma_g  # gaussian stds
    g_size = 6 * sig_g / 2
    g_sig, ig2 = gauss2nd_conv(sig_mn, sig=sig_g)  # non-smoothed
    g_sig_valid = 2 * (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size])) \
                  / (np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size])) - 1
    g_sig_valid_non = g_sig[g_size: -g_size]  # NON-normalized
    if norm == True:
        g_sig_valid = g_sig[g_size: -g_size]  # NON-normalized
    max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid_non, 1e-1, tx[g_size: -g_size])
    onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual')

    # print 'site no is %s' % si0
    print 'station ID is %s' % site_no
    if inc_plot is True:
        fig = pylt.figure(figsize=[8, 3])
        ax = fig.add_subplot(111)
        x, y = u_doy, inc_d
        ax.plot(x, y, 'o')
        pylt.savefig('./result_07_01/inc_mid_' + site_no + '.png')
        fig.clear()
        pylt.close()
        # ons_site = sm_onset(sm5_date-365, sm5_daily, t5_daily)
    # onset based on ascat

    # actual pass time
    # sec_ascat = bxy.timetransform(ascat_ob[:, 1], '20000101 00:00:00', '%Y%m%d %X', tzone=True)
    # doy, passhr = np.modf((sec_ascat)/3600.0/24.0)[1] + 1, np.round(np.modf((sec_ascat)/3600.0/24.0)[0]*24)
    return [tx[ig2][g_size: -g_size] + 365, g_sig_valid, g_sig_valid_non], \
           [tx, sig_mn, inc_d], \
           onset, \
           pass_hr_d, \
           [u_doy + 365, pass_hr_d], \
           [max_gsig_s, min_gsig_s]  # g_sig[g_size: -g_size] g_sig_valid
    return 0


def ascat_alaska_onset(ob='AS', norm=False, std=3, version='old', target00=[142, 237]):
    if version == 'new':
        f_doc = 'all_year_observation'
        doy0 = get_doy('20151001') - 365
        doy0 = get_doy('20151001') - 365
        doy_range = np.arange(doy0, doy0 + 500) - 1
    else:
        doy_range = np.arange(0, 365)
        f_doc = 'all_year_observation_old'
    if norm:
        isnorm = 'norm'
        folder_path = './result_05_01/ascat_resample_norms/ascat_resample_' + ob + '/'
    else:
        isnorm = 'orig'
        folder_path = './result_05_01/ascat_resample_' + ob + '/'
    print 'the sigma data is %s, %s' % (isnorm, ob)
    d0 = datetime.date(2016, 1, 1)  # change from 2016 1 1
    # initial a base grid
    base0 = np.load(folder_path + 'ascat_20160101_resample.npy')
    sigma_3d = np.zeros([base0.shape[0], base0.shape[1], doy_range.size])
    inc_3d = np.zeros([base0.shape[0], base0.shape[1], doy_range.size])
    onset_2d = np.zeros([base0.shape[0], base0.shape[1]])
    lg_grid, lat_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                        np.load('./result_05_01/other_product/lat_ease_grid.npy')
    fpath0 = './result_05_01/onset_result/%s/' % f_doc  # saving path
    if not os.path.exists(fpath0 + 'ascat_all_2016_' + isnorm + '_' + ob + '.npy'):
        # find the daily sigma and mask files of AK
        i = 0
        for doy in doy_range:
            d_daily = d0 + timedelta(doy)
            d_str = d_daily.strftime('%Y%m%d')
            ak_file = bxy.find_by_date(d_str, folder_path)
            for f1 in ak_file:
                if f1.find('resample') > 0:
                    v = np.load(folder_path + f1)
                    sigma_3d[:, :, i] = v
                    # if f1.find('incidence') > 0:
                    #     agl = np.load(folder_path+f1)
                    #     inc_3d[:, :, i] = agl
            i += 1
        np.save(fpath0 + 'ascat_all_2016_' + isnorm + '_' + ob, sigma_3d)
        # np.save(fpath0+'ascat_inc_2016_'+ob, inc_3d)
    else:
        sigma_3d = np.load(fpath0 + 'ascat_all_2016_' + isnorm + '_' + ob + '.npy')
        # inc_3d = np.load(fpath0+'ascat_inc_2016_'+ob+'.npy')

    # build mask for 30 day that can overpass can cover all area of Alaska
    mask0 = sigma_3d[:, :, 0] != 0
    for m in range(0, 30):
        maski = sigma_3d[:, :, m] != 0
        mask0 = np.logical_or(mask0, maski)
    sigma_land = sigma_3d[mask0, :]
    # lat_land, lg_land = lat_grid[mask0], lg_grid[mask0]
    target0 = target00  # row and col of target
    np.save('./result_05_01/other_product/mask_ease2_125N', mask0)
    row_test, col_test = np.where(mask0)[0], np.where(mask0)[1]  # test of target point
    nodata_count, land_count = 0, 0
    onset_1d = [[], []]
    for s1 in sigma_land:
        # edge detection
        g_size = 3 * std
        i_sig_valid = (s1 != 0)
        doy_i = doy_range[i_sig_valid]
        sigma_i = s1[i_sig_valid]
        if sigma_i.size >= 120:
            g_sig, ig2 = gauss_conv(sigma_i, sig=std, size=6 * std + 1)  # non-smoothed
            g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size])) \
                          / (np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
            g_sig_valid = g_sig[g_size: -g_size]
            max_gsig, min_gsig = peakdetect.peakdet(g_sig_valid, 1e-1, doy_i[g_size: -g_size])
            onset = find_inflect(max_gsig, min_gsig, typez='annual')
            onset_1d[0].append(onset[0])
            onset_1d[1].append(onset[1])
            if (row_test[land_count] == target0[0]) & (col_test[land_count] == target0[1]):  # special point test & plot
                print 'point for test: ', target0
                print 'location of odd point ', lg_grid[target0[0], target0[1]], lat_grid[target0[0], target0[1]]
                print 'col and row of odd point ', row_test[land_count], col_test[land_count], onset[0]
                fig = pylt.figure()
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                plot_funcs.pltyy(doy_i, sigma_i, fpath0 + 'odd_point', '$\sigma_0$ (dB)',
                                 t2=doy_i[g_size: -g_size], s2=g_sig_valid,
                                 ylim=[-20, -4], symbol=['bo', 'g-'], handle=[fig, ax])
        else:
            onset_1d[0].append(np.array(0))
            onset_1d[1].append(np.array(0))
            nodata_count += 1

        land_count += 1
        # print 'no data count: %f' % nodata_count
    print '%d pixels have no valid data\n' % nodata_count
    onset_2d[mask0] = onset_1d[0]
    np.save(fpath0 + 'ascat_onset_0_2016_' + isnorm + '_' + ob + '_w' + str(std), onset_2d)
    onset_2d[mask0] = onset_1d[1]
    np.save(fpath0 + 'ascat_onset_1_2016_' + isnorm + '_' + ob + '_w' + str(std), onset_2d)

    # mask_1d = mask0.reshape(1, -1)
    # sigma_2d = sigma_3d.reshape(1, -1, 365)
    # i_land = np.where(mask_1d == True)
    # for l0 in i_land[1]:
    #     sig_land0 = sigma_2d[0, l0]
    pause = True


def ascat_plot_series_v20(site_no, orb_no=0, inc_plot=False, sigma_g=5, pp=False, norm=False,
                          txt_path='./result_05_01/ascat_point/', is_sub=False,
                          order=1):  # orbit default is 0, ascending, night pass
    """
    The version for 2nd revision. The unit of x_time is still doy from 2015.
    """
    # read series sigma
    site = site_no
    site_nos = site_infos.get_id(site, mode='single')
    for si0 in site_nos:
        if is_sub is False:
            txtname = glob.glob(txt_path + 'ascat_s' + si0 + '*')[0]
        # txtname = './result_05_01/ascat_point/ascat_s'+si0+'_2016.npy'
        ascat_all = np.load(txtname)
        if ascat_all.size < 1:
            return [-1, -1, -1], [-1, -1, -1], [-1, -1], -1, [-1, -1], [-1, -1]
        id_orb = ascat_all[:, -1] == orb_no
        ascat_ob = ascat_all[id_orb]

        # # transform utc time to local time
        # sec_ascat = bxy.timetransform(ascat_ob[:, 1], '20000101 00:00:00', '%Y%m%d %X', tzone=True)
        # doy_tp = np.modf((sec_ascat)/3600.0/24.0)
        # doy = doy_tp[1]+1
        # passhr = np.round(doy_tp[0]*24.0)

        # newly updated 0515/2018
        times_ascat = bxy.time_getlocaltime(ascat_ob[:, 1], ref_time=[2000, 1, 1, 0])
        doy_tp2 = times_ascat[-2] + (times_ascat[0] - 2015) * 365 + np.max(
            np.array([(times_ascat[0] - 2016), np.zeros(times_ascat[0].size)]), axis=0)
        passhr = times_ascat[-1] * 1.0

        # angular normalization
        # tx = doy_tp[1]+1 -365
        tx = doy_tp2 + 1 - 365
        p = (tx > 0) & (tx < 60)
        x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
        a, b = np.polyfit(x, y, 1)
        print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)
        f = np.poly1d([a, b])

        # angular dependency for each ob mode
        p_win, p_su = (tx > 0) & (tx < 60), (tx > 150) & (tx < 260)
        x_trip_win, x_trip_su = ascat_ob[p_win, 5: 8], ascat_ob[p_su, 5: 8]
        y_trip_win, y_trip_su = ascat_ob[p_win, 2: 5], ascat_ob[p_su, 2: 5]
        a_array, b_array, d_array = np.zeros(6) - 1, np.zeros(6) - 1, np.zeros(3) - 99
        for i1 in range(0, 3):
            a_w, b_w = np.polyfit(x_trip_win[:, i1], y_trip_win[:, i1], 1)
            a_s, b_s = np.polyfit(x_trip_su[:, i1], y_trip_su[:, i1], 1)
            a_array[i1], b_array[i1] = a_w, b_w
            a_array[i1 + 3], b_array[i1 + 3] = a_s, b_s
            d_array[i1] = np.mean(y_trip_su[:, i1]) - np.mean(y_trip_win[:, i1])
        a_coef_name = 'ascat_linear_a.txt'
        b_coef_name = 'ascat_linear_b.txt'
        with open(a_coef_name, 'a') as t_file:
            t_file.write(
                '%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, a_array[0], a_array[1], a_array[2],
                                                                       a_array[3], a_array[4], a_array[5], d_array[0],
                                                                       d_array[1], d_array[2]))
        with open(b_coef_name, 'a') as t_file:
            t_file.write(
                '%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, b_array[0], b_array[1], b_array[2],
                                                                       b_array[3], b_array[4], b_array[5], d_array[0],
                                                                       d_array[1], d_array[2]))

        sig_m = ascat_ob[:, 3]
        inc_m = ascat_ob[:, 6]
        # sig_m = ascat_ob[:, 2]  # changed using the forwards mode to increase the winter summer difference
        # inc_m = ascat_ob[:, 5]
        sig_mn = sig_m - (ascat_ob[:, 6] - 45) * a
        # daily average:
        tdoy = tx
        u_doy = np.unique(tdoy)
        sig_d, i0 = np.zeros([u_doy.size, 2]), 0
        pass_hr_d = np.zeros(u_doy.size)
        inc_d = np.zeros(u_doy.size)
        for td in u_doy:
            sig_d[i0][0] = td
            sig_d[i0][1] = np.mean(sig_mn[tdoy == td])
            inc_d[i0] = np.mean(inc_m[tdoy == td])  # daily incidence angle
            pass_hr_d[i0] = np.mean(passhr[tdoy == td])
            i0 += 1
        tx = sig_d[:, 0]
        sig_mn = sig_d[:, 1]
        pass_hr_d = np.round(pass_hr_d)
        # one more constraints, based on incidence angle
        # id_inc = bxy.gt_le(inc_d, 30, 35)
        # tx, sig_mn = tx[id_inc], sig_mn[id_inc]

        # edge detect
        sig_g = sigma_g  # gaussian stds
        g_size = 6 * sig_g / 2
        if order == 1:
            g_sig, ig2 = gauss_conv(sig_mn, sig=sig_g, size=2 * g_size + 1)  # non-smoothed
        elif order == 2:
            g_sig, ig2 = gauss2nd_conv(sig_mn, sig=sig_g)
        g_sig_valid = 2 * (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size])) \
                      / (np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size])) - 1
        g_sig_valid_non = g_sig[g_size: -g_size]  # NON-normalized
        if norm == True:
            g_sig_valid = g_sig[g_size: -g_size]  # NON-normalized
        max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid_non, 1e-1, tx[g_size: -g_size])
        onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual')

        # new updated 20161130
        # onset, g_npr, i_gaussian, g_sig_valid_non, max_gsig_s, min_gsig_s, sig \
        #     = gauss_cov_snr(sig_mn, 1e-1, tx+365)
        # tp = [x for x in onset]
        # onset = tp
        # g_size = 6*sig/2
        # print 'site no is %s' % si0

        print 'station ID is %s' % si0
        if inc_plot is True:
            # tx = doy_tp[1]+1 -365
            # p = (tx > 20) & (tx < 90)
            # x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
            plot_funcs.inc_plot_ascat(ascat_ob, site_no)
            # ons_site = sm_onset(sm5_date-365, sm5_daily, t5_daily)
        # onset based on ascat

        # actual pass time
        # sec_ascat = bxy.timetransform(ascat_ob[:, 1], '20000101 00:00:00', '%Y%m%d %X', tzone=True)
        # doy, passhr = np.modf((sec_ascat)/3600.0/24.0)[1] + 1, np.round(np.modf((sec_ascat)/3600.0/24.0)[0]*24)
        return [tx[ig2][g_size: -g_size] + 365, g_sig_valid, g_sig_valid_non], \
               [tx, sig_mn, inc_d], \
               onset, \
               pass_hr_d, \
               [u_doy + 365, pass_hr_d], \
               [max_gsig_s, min_gsig_s]  # g_sig[g_size: -g_size] g_sig_valid


def smap_alaska_onset(mode='tb', std=3, version='old'):
    if version == 'new':
        f_doc = 'all_year_observation'
        f_doc2 = 'all'
        doy0 = get_doy('20151001') - 365
        doy_range = np.arange(doy0, doy0 + 500) - 1
    else:
        f_doc = 'all_year_observation_old'
        f_doc2 = 'all'
        doy_range = np.arange(0, 365)
    fpath0 = './result_05_01/onset_result/%s/' % f_doc
    ob = 'AS'
    folder_path = './result_05_01/smap_resample_' + ob + '/%s/' % f_doc2
    base0 = np.load(folder_path + 'smap_20160105_tbv_resample.npy')
    d0 = datetime.date(2016, 1, 1)
    tbv_3d = np.zeros([base0.shape[0], base0.shape[1], doy_range.size])
    tbh_3d = np.zeros([base0.shape[0], base0.shape[1], doy_range.size])
    onset_2d = np.zeros([base0.shape[0], base0.shape[1]])
    if not os.path.exists(fpath0 + 'smap_all_2016_tbv_' + ob + '.npy'):
        # find the daily sigma and mask files of AK
        print 'file not found: ', fpath0 + 'smap_all_2016' + '_' + mode + '_' + ob + '.npy'
        i = 0
        for doy in doy_range:
            d_daily = d0 + timedelta(doy)
            d_str = d_daily.strftime('%Y%m%d')
            ak_file = bxy.find_by_date(d_str, folder_path)
            for f1 in ak_file:
                if f1.find('v_resample') > 0:
                    v = np.load(folder_path + f1)
                    tbv_3d[:, :, i] = v
                if f1.find('h_resample') > 0:  # other attributs
                    agl = np.load(folder_path + f1)
                    tbh_3d[:, :, i] = agl
            i += 1
        np.save(fpath0 + 'smap_all_2016_tbv_' + ob, tbv_3d)
        np.save(fpath0 + 'smap_all_2016_tbh_' + ob, tbh_3d)
        # np.save('./result_05_01/onset_result/smap_all_2016_tbh_'+ob, tbh_3d)
    else:
        tbv_3d = np.load(fpath0 + 'smap_all_2016_tbv_' + ob + '.npy')
        tbh_3d = np.load(fpath0 + 'smap_all_2016_tbh_' + ob + '.npy')
    # read mask for land
    mask0 = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
    landid = mask0 == 1
    v_land = tbv_3d[landid, :]
    if mode == 'npr':
        tbv_3d_ma = np.ma.masked_array(tbv_3d, mask=[tbv_3d < 100])
        tbh_3d_ma = np.ma.masked_array(tbh_3d, mask=[tbh_3d < 100])
        npr = (tbv_3d_ma - tbh_3d_ma) / (tbv_3d_ma + tbh_3d_ma)
        v_land = npr[landid]
        npr_test = npr[51, 68, :]
    mask_row, mask_col = np.where(landid)[0], np.where(landid)[1]
    test_id = np.where((mask_row == 51) & (mask_col == 68))
    nodata_count = 0
    land_count = 0
    onset_1d = [[], []]
    # npr_test = npr[21, 52, :]
    for s1 in v_land:
        # edge detection
        g_size = 3 * std
        if mode == 'npr':
            i_tbv_valid = (s1 != 0) & (s1 != -9999) & (s1 != 1) & (s1 != -1)
        else:
            i_tbv_valid = ((s1 != 0) & (s1 != -9999))
        doy_i = doy_range[i_tbv_valid]
        tbv_i = s1[i_tbv_valid]
        if tbv_i.size >= 120:
            g_sig, ig2 = gauss_conv(tbv_i, sig=std, size=2 * g_size + 1)  # non-smoothed
            g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size])) \
                          / (np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
            if mode == 'npr':
                max_gsig, min_gsig = peakdetect.peakdet(g_sig[g_size: -g_size], 1e-4, doy_i[g_size: -g_size])
                onset = find_inflect(max_gsig, min_gsig, typez='annual')
            else:
                max_gsig, min_gsig = peakdetect.peakdet(g_sig_valid, 1e-1, doy_i[g_size: -g_size])
                onset = find_inflect(max_gsig, min_gsig, typez='annual', typen='tb')
            if land_count == test_id[0][0]:
                pause = 0
                fig = pylt.figure()
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                # print onset
                # print 's1 size: ', s1.size
                # print "real series's of s1 size is: ", tbv_i.size
                # print 'the temporal change of npr is: ', tbv_i[g_size: -g_size][58: 61]
                # id_test = int(max_gsig[max_gsig[:, 1] == onset[0]][0, 0])

                # iv_valid = ((v != 0) & (v != -9999))
                # iv_valid = (v != 0) & (v > -999)
                # ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d, Days: %d' % (loni, lati, rowi, coli, onset[0], onset[1], sum(v<0)))
                plot_funcs.pltyy(doy_i, tbv_i, 'test00xx', '$NPR$ (%)',
                                 t2=doy_i[ig2][g_size: -g_size], s2=g_sig[g_size: -g_size],
                                 ylim=[0, 0.1], symbol=['bo', 'g-'], handle=[fig, ax])
            onset_1d[0].append(onset[0])
            onset_1d[1].append(onset[1])
        else:
            onset_1d[0].append(np.array(0))
            onset_1d[1].append(np.array(0))
            nodata_count += 1
        land_count += 1
        # print 'no data count: %f' % nodata_count
    print '%d pixels have no valid data\n' % nodata_count
    onset_2d[landid] = onset_1d[0]
    np.save(fpath0 + 'smap_onset_0_2016' + '_' + mode + '_' + ob + '_w' + str(std), onset_2d)
    onset_2d[landid] = onset_1d[1]
    np.save(fpath0 + 'smap_onset_1_2016' + '_' + mode + '_' + ob + '_w' + str(std), onset_2d)
    return 0


def make_mask(onset, mask_value=[0, -999, -9999]):
    mask0 = np.load('/home/xiyu/PycharmProjects/R3/result_05_01/other_product/mask_ease2_125N.npy')
    # set mask to onset value
    onset[np.isnan(onset)] = -999
    for m0 in mask_value:
        onset[onset == m0] = -999
    value_array = np.zeros(mask0.shape)
    value_array[mask0 > 0] = onset
    return value_array


def ascat_onset_map(ob, odd_point=[], product='ascat', mask=False, std=4, mode=['_norm_'], version='old',
                    f_win=[0, 0], t_win=[0, 0], custom=[], points_index=np.array([]), resolution=360, input_onset=[]):
    anc_direct = './result_05_01/other_product/'
    if version == 'new':
        result_doc = 'all_year_observation'
    elif version == 'old':
        result_doc = 'all_year_observation_old'
    fpath1 = './result_08_01/'

    if product == 'ascat':
        prefix = './result_05_01/onset_result/%s/' % result_doc
        lons_grid, lats_grid = np.load('./result_05_01/onset_result/lon_ease_grid.npy'), \
                               np.load('./result_05_01/onset_result/lat_ease_grid.npy')
        if len(odd_point) > 0:
            if type(odd_point[0]) is not list:
                odds = np.array(odd_point).T
                print odds[1]
                dis_odd = bxy.cal_dis(odds[1], odds[0], lats_grid.ravel(), lons_grid.ravel())
                index = np.argmin(dis_odd)
                row = int(index / lons_grid.shape[1])
                col = index - (index / lons_grid.shape[1] * lons_grid.shape[1])
                print row, col, lons_grid[row, col], lats_grid[row, col]
        for key in ob:
            for m in mode:
                onset_0_file = prefix + 'ascat_onset_0' + '_2016' + m + key + '_w' + str(std) + '.npy'
                onset0 = np.load(onset_0_file)
                onset_1_file = prefix + 'ascat_onset_1' + '_2016' + m + key + '_w' + str(std) + '.npy'
                onset1 = np.load(onset_1_file)
                if mask is True:
                    # './result_05_01/other_product/snow_mask_125s.npy'
                    mask_snow = np.load(anc_direct + 'snow_mask_125s.npy')
                    onset0 = np.ma.masked_array(onset0, mask=[mask_snow == 0])
                    onset1 = np.ma.masked_array(onset1, mask=[mask_snow == 0])
                fpath1 = 'result_08_01/'
                pass_zone_plot(lons_grid, lats_grid, onset0, fpath1,
                               fname='onset_0' + m + key + '_w' + str(std),
                               z_max=180, z_min=50, odd_points=odd_point)
                pass_zone_plot(lons_grid, lats_grid, onset1, fpath1,
                               fname='onset_1' + m + key + '_w' + str(std),
                               z_max=360, z_min=240, odd_points=odd_point)
    elif product == 'grid_test':
        for d_str in ['20151102']:
            h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % d_str
            h0 = h5py.File(h5_name)
            lons_grid = h0['cell_lon'].value
            lats_grid = h0['cell_lat'].value

            indicators = [product]
            # change onset vlaue
            th_name = 'test_onset0_%s.npy' % std
            fr_name = 'test_onset1_%s.npy' % std
            onset0 = np.load(th_name)
            onset1 = np.load(fr_name)  # test_onset1.npy
            onset0_14 = np.load('test_onset0_14.npy')
            onset1_14 = np.load('test_onset1_14.npy')
            out_bound = lons_grid > -141.0
            onset0[out_bound], onset1[out_bound], onset0_14[out_bound], onset1_14[out_bound] = 0, 0, 0, 0

            if len(odd_point) == 4:
                odd_lon = odd_point[2]
                odd_lat = odd_point[3]
                odd_onset = onset0[odd_point[0], odd_point[1]]
            else:
                print 'add all location with labels'
            tbv0 = h0[u'cell_tb_v_aft'].value
            tbv0[tbv0 < 0] = 0
            mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
            onset0 = np.ma.masked_array(onset0, mask=[(onset0 == 0) | (mask == 0)])
            onset1 = n
            p.ma.masked_array(onset1, mask=[(onset1 == 0) | (mask == 0)])
            onset0_14 = np.ma.masked_array(onset0_14, mask=[(onset0_14 == 0) | (mask == 0)])
            onset1_14 = np.ma.masked_array(onset1_14, mask=[(onset1_14 == 0) | (mask == 0)])
            # mask the snow cover
            mask_snow = np.load('./result_05_01/other_product/snow_mask_360_2.npy')
            onset0 = np.ma.masked_array(onset0, mask=[mask_snow != 0])
            onset1 = np.ma.masked_array(onset1, mask=[mask_snow != 0])
            onset0_14 = np.ma.masked_array(onset0_14, mask=[mask_snow != 0])
            onset1_14 = np.ma.masked_array(onset1_14, mask=[mask_snow != 0])
            # onset0 = np.ma.masked_array(onset0, mask=[onset0==0])
            # for ind in indicators:
            #     onset0 = np.load(fpath1+'smap_onset_0_2016_'+ind+'_AS'+'_w'+str(std)+'.npy')
            #     onset1 = np.load(fpath1+'smap_onset_1_2016_'+ind+'_AS'+'_w'+str(std)+'.npy')
            #     if mask is True:
            #         mask0 = np.load(anc_direct+'snow_mask_360s.npy')
            #         onset0 = np.ma.masked_array(onset0, mask=[mask0==0])
            #         onset1 = np.ma.masked_array(onset1, mask=[mask0==0])
            #     fpath1 = 'result_08_01/'
            thawname = 'npr_thaw_s%d_doy%d_%d' % (std, t_win[0], t_win[1])
            # thawtitle = 'Thawing window: DOY %d--%d' % (t_win[0], t_win[1])
            thawtitle = 's of the normal distribution: %d' % std
            frname = 'npr_freeze_s%d_doy%d_%d' % (std, f_win[0], f_win[1])
            # frtitle = 'Freezing window: DOY %d--%d' % (f_win[0], f_win[1])
            frtitle = 's of the normal distribution: %d' % std

            # pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname=thawname, z_max=60, z_min=150, prj='aea',
            #                odd_points=odd_point[:, [0, 2, 3, 4]], title_str=thawtitle)  # fpath1
            # pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname=frname, z_max=250, z_min=340, prj='aea',
            #                odd_points=odd_point[:, [1, 2, 3, 4]], title_str=frtitle)

            pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname=thawname, z_max=30, z_min=150, prj='aea',
                           odd_points=np.array([odd_lon, odd_lat]), title_str=thawtitle)  # fpath1
            pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname=frname, z_max=250, z_min=340, prj='aea',
                           odd_points=np.array([odd_lon, odd_lat]), title_str=frtitle)
            pass_zone_plot(lons_grid, lats_grid, onset1_14 - onset1, fpath1, fname='freeze_14_7', z_max=-20, z_min=20,
                           prj='aea',
                           odd_points=np.array([odd_lon, odd_lat]), title_str='Freezing onsets bias')
            pass_zone_plot(lons_grid, lats_grid, onset0_14 - onset0, fpath1, fname='thaw_14_7', z_max=-20, z_min=20,
                           prj='aea',
                           odd_points=np.array([odd_lon, odd_lat]), title_str='Thawing onsets bias')
    elif product == 'customize':
        for d_str in ['20151102']:
            if resolution == 125:
                lons_grid, lats_grid = np.load('./result_05_01/onset_result/lon_ease_grid.npy'), \
                                       np.load('./result_05_01/onset_result/lat_ease_grid.npy')
            elif resolution == 360:
                h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % d_str
                h0 = h5py.File(h5_name)
                lons_grid = h0['cell_lon'].value
                lats_grid = h0['cell_lat'].value
            lons_1d, lats_1d = lons_grid.ravel(), lats_grid.ravel()

            indicators = [product]
            # change onset vlaue
            th_name = custom[0]
            fr_name = custom[1]
            conv_name = custom[2]
            onset0 = np.load(th_name)
            onset1 = np.load(fr_name)  # test_onset1.npy
            onset2 = np.load(conv_name)
            # check shape
            onset0 = check_shape(onset0)
            onset1 = check_shape(onset1)
            onset2 = check_shape(onset2)

            if 'level' in conv_name:
                onset2 = np.abs(onset2)
            out_bound = lons_grid > -141.0
            onset0[out_bound], onset1[out_bound], onset2[out_bound] = 0, 0, 0
            south_bound = (lats_grid < 61) & (onset0 > 120)
            onset0[south_bound] = 0

            if len(odd_point) == 4:
                odd_lon = odd_point[2]
                odd_lat = odd_point[3]
            else:
                print 'add all location with labels'
                odd_lon = np.array([-1, -1])
                odd_lat = np.array([-1, -1])
            if points_index.size > 3:
                odd_lon = lons_1d[points_index]
                odd_lat = lats_1d[points_index]

            tbv0 = h0[u'cell_tb_v_aft'].value
            tbv0[tbv0 < 0] = 0
            mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
            melt_ini = 0
            onset0 = np.ma.masked_array(onset0, mask=[(onset0 == 0) | (mask == 0) | (onset0 < 60) | (onset0 > 140)])
            onset1 = np.ma.masked_array(onset1, mask=[(onset1 < melt_ini) | (mask == 0) | (onset1 < 30)])
            # onset2 = np.ma.masked_array(onset2, mask=[(onset2<-20)|(onset1<30)])
            onset2 = np.ma.masked_array(onset2, mask=[
                ((onset2 == -999) | (onset2 == 0)) | (onset1 < melt_ini) | (onset2 == 999)])

            # mask the snow cover
            mask_snow = np.load('./result_05_01/other_product/snow_mask_360_2.npy')
            onset0 = np.ma.masked_array(onset0, mask=[mask_snow != 0])  # thawing
            onset1 = np.ma.masked_array(onset1, mask=[mask_snow != 0])  # melting
            onset2 = np.ma.masked_array(onset2, mask=[mask_snow != 0])
            # mask odd value
            min2 = np.abs(np.min(onset2))
            onset2 = np.ma.masked_array(onset2, mask=[np.abs(onset2) > 1000 * min2])
            onset2_max, onset2_min = max(np.max(onset2) * 1.1, np.max(onset2) * 0.9), \
                                     min(np.min(onset2) * 0.9, np.min(onset2) * 1.1)
            # temporaly save a level mask
            # new_level = onset2.data
            # np.save('result_08_01/melt_level_new_7.npy', onset2.data)
            if 'difference' in conv_name:
                print conv_name
                onset2_max, onset2_min = 50, -50
            # odd_onset2 check
            # odd_id = (np.array([10, 10, 19, 20, 20, 20, 20, 24, 24, 25, 27, 28, 33, 34, 39, 40, 40,
            #          41, 56, 57]), np.array([54, 58, 46, 48, 56, 57, 70, 44, 45, 48, 44, 46, 40, 45, 33, 39, 40,
            #         37, 63, 61]))
            # print onset2[odd_id]
            # plot a scattring plot
            thawing_est0 = np.ma.masked_array(onset0, mask=onset1.mask).data
            thawing_est = thawing_est0[~onset1.mask]
            # thawing_est = np.ma.masked_array(onset0, mask=np.ma.get_mask(onset1))
            melting_est = onset1.data[~onset1.mask]
            valid_point = ~onset1.mask
            plot_funcs.plot_quick_scatter(thawing_est, melting_est)


            # onset0 = np.ma.masked_array(onset0, mask=[onset0==0])
            # for ind in indicators:
            #     onset0 = np.load(fpath1+'smap_onset_0_2016_'+ind+'_AS'+'_w'+str(std)+'.npy')
            #     onset1 = np.load(fpath1+'smap_onset_1_2016_'+ind+'_AS'+'_w'+str(std)+'.npy')
            #     if mask is True:
            #         mask0 = np.load(anc_direct+'snow_mask_360s.npy')
            #         onset0 = np.ma.masked_array(onset0, mask=[mask0==0])
            #         onset1 = np.ma.masked_array(onset1, mask=[mask0==0])
            #     fpath1 = 'result_08_01/'
            key_name0 = th_name.split('/')[-1].split('.')[0]
            key_name1 = fr_name.split('/')[-1].split('.')[0]
            thawname = '%s_thaw_s%s' % (product, key_name0)  # name 1
            thawtitle = 's of the normal distribution: %d' % std
            thawtitle = ' '
            frname = '%s_melt_s%s' % (product, key_name1)  # name 2
            frtitle = 's of the normal distribution: %d' % std
            frtitle = ' '
            key_name2 = conv_name.split('/')[-1].split('.')[0]
            convfile = '%s_%s_s%d_doy' % (product, key_name2, std)  # name 3
            print key_name0, key_name1, key_name2
            # pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname=thawname, z_max=60, z_min=150, prj='aea',
            #                odd_points=odd_point[:, [0, 2, 3, 4]], title_str=thawtitle)  # fpath1
            # pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname=frname, z_max=250, z_min=340, prj='aea',
            #                odd_points=odd_point[:, [1, 2, 3, 4]], title_str=frtitle)

            pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname=thawname, z_max=30, z_min=150, prj='aea',
                           odd_points=np.array([odd_lon, odd_lat]), title_str=thawtitle, txt=points_index)  # fpath1
            pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname=frname, z_max=30, z_min=150, prj='aea',
                           odd_points=np.array([odd_lon, odd_lat]), title_str=frtitle, txt=points_index)
            onset2_max, onset2_min = 150, 30
            pass_zone_plot(lons_grid, lats_grid, onset2, fpath1, fname=convfile, z_max=onset2_max, z_min=onset2_min,
                           prj='aea', odd_points=np.array([odd_lon, odd_lat]), title_str=frtitle, txt=points_index)
    elif product == 'input_onset':
        for d_str in ['20151102']:
            if resolution == 125:
                lons_grid, lats_grid = np.load('./result_05_01/onset_result/lon_ease_grid.npy'), \
                                       np.load('./result_05_01/onset_result/lat_ease_grid.npy')
            elif resolution == 360:
                h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % d_str
                h0 = h5py.File(h5_name)
                lons_grid = h0['cell_lon'].value
                lats_grid = h0['cell_lat'].value
            lons_1d, lats_1d = lons_grid.ravel(), lats_grid.ravel()

            indicators = [product]
            # change onset vlaue
            th_name = custom[0]
            fr_name = custom[1]
            conv_name = custom[2]
            onset0 = input_onset[0]
            onset1 = input_onset[1]  # test_onset1.npy
            onset2 = input_onset[2]

            if 'level' in conv_name:
                onset2 = np.abs(onset2)
            out_bound = lons_grid > -141.0
            onset0[out_bound], onset1[out_bound], onset2[out_bound] = 0, 0, 0
            south_bound = (lats_grid < 61) & (onset0 > 120)
            onset0[south_bound] = 0

            if len(odd_point) == 4:
                odd_lon = odd_point[2]
                odd_lat = odd_point[3]
            else:
                print 'add all location with labels'
                odd_lon = np.array([-1, -1])
                odd_lat = np.array([-1, -1])
            if points_index.size > 3:
                odd_lon = lons_1d[points_index]
                odd_lat = lats_1d[points_index]

            tbv0 = h0[u'cell_tb_v_aft'].value
            tbv0[tbv0 < 0] = 0
            mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
            melt_ini = 0
            onset0 = np.ma.masked_array(onset0, mask=[(onset0 == 0) | (mask == 0) | (onset0 < 60) | (onset0 > 140)])
            onset1 = np.ma.masked_array(onset1, mask=[(onset1 < melt_ini) | (mask == 0) | (onset1 < 30)])
            # onset2 = np.ma.masked_array(onset2, mask=[(onset2<-20)|(onset1<30)])
            onset2 = np.ma.masked_array(onset2, mask=[
                ((onset2 == -999) | (onset2 == 0)) | (onset1 < melt_ini) | (onset2 == 999)])

            # mask the snow cover
            mask_snow = np.load('./result_05_01/other_product/snow_mask_360_2.npy')
            onset0 = np.ma.masked_array(onset0, mask=[mask_snow != 0])  # thawing
            onset1 = np.ma.masked_array(onset1, mask=[mask_snow != 0])  # melting
            onset2 = np.ma.masked_array(onset2, mask=[mask_snow != 0])
            # mask odd value
            min2 = np.abs(np.min(onset2))
            onset2 = np.ma.masked_array(onset2, mask=[np.abs(onset2) > 1000 * min2])
            onset2_max, onset2_min = max(np.max(onset2) * 1.1, np.max(onset2) * 0.9), \
                                     min(np.min(onset2) * 0.9, np.min(onset2) * 1.1)
            # temporaly save a level mask
            # new_level = onset2.data
            # np.save('result_08_01/melt_level_new_7.npy', onset2.data)
            if 'difference' in conv_name:
                print conv_name
                onset2_max, onset2_min = 50, -50
            # odd_onset2 check
            # odd_id = (np.array([10, 10, 19, 20, 20, 20, 20, 24, 24, 25, 27, 28, 33, 34, 39, 40, 40,
            #          41, 56, 57]), np.array([54, 58, 46, 48, 56, 57, 70, 44, 45, 48, 44, 46, 40, 45, 33, 39, 40,
            #         37, 63, 61]))
            # print onset2[odd_id]
            # plot a scattring plot
            thawing_est0 = np.ma.masked_array(onset0, mask=onset1.mask).data
            thawing_est = thawing_est0[~onset1.mask]
            # thawing_est = np.ma.masked_array(onset0, mask=np.ma.get_mask(onset1))
            melting_est = onset1.data[~onset1.mask]
            valid_point = ~onset1.mask
            plot_funcs.plot_quick_scatter(thawing_est, melting_est)


            # onset0 = np.ma.masked_array(onset0, mask=[onset0==0])
            # for ind in indicators:
            #     onset0 = np.load(fpath1+'smap_onset_0_2016_'+ind+'_AS'+'_w'+str(std)+'.npy')
            #     onset1 = np.load(fpath1+'smap_onset_1_2016_'+ind+'_AS'+'_w'+str(std)+'.npy')
            #     if mask is True:
            #         mask0 = np.load(anc_direct+'snow_mask_360s.npy')
            #         onset0 = np.ma.masked_array(onset0, mask=[mask0==0])
            #         onset1 = np.ma.masked_array(onset1, mask=[mask0==0])
            #     fpath1 = 'result_08_01/'
            key_name0 = th_name.split('/')[-1].split('.')[0]
            key_name1 = fr_name.split('/')[-1].split('.')[0]
            thawname = '%s_thaw_s%s' % (product, key_name0)  # name 1
            thawtitle = 's of the normal distribution: %d' % std
            thawtitle = ' '
            frname = '%s_melt_s%s' % (product, key_name1)  # name 2
            frtitle = 's of the normal distribution: %d' % std
            frtitle = ' '
            key_name2 = conv_name.split('/')[-1].split('.')[0]
            convfile = '%s_%s_s%d_doy' % (product, key_name2, std)  # name 3
            print key_name0, key_name1, key_name2
            # pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname=thawname, z_max=60, z_min=150, prj='aea',
            #                odd_points=odd_point[:, [0, 2, 3, 4]], title_str=thawtitle)  # fpath1
            # pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname=frname, z_max=250, z_min=340, prj='aea',
            #                odd_points=odd_point[:, [1, 2, 3, 4]], title_str=frtitle)

            pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname=thawname, z_max=30, z_min=150, prj='aea',
                           odd_points=np.array([odd_lon, odd_lat]), title_str=thawtitle, txt=points_index)  # fpath1
            pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname=frname, z_max=30, z_min=150, prj='aea',
                           odd_points=np.array([odd_lon, odd_lat]), title_str=frtitle, txt=points_index)
            onset2_max, onset2_min = 150, 30
            pass_zone_plot(lons_grid, lats_grid, onset2, fpath1, fname=convfile, z_max=onset2_max, z_min=onset2_min,
                           prj='aea', odd_points=np.array([odd_lon, odd_lat]), title_str=frtitle, txt=points_index)
    else:
        lons_grid = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
        lats_grid = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
        indicators = [product]
        for ind in indicators:
            onset0 = np.load(fpath1 + 'smap_onset_0_2016_' + ind + '_AS' + '_w' + str(std) + '.npy')
            onset1 = np.load(fpath1 + 'smap_onset_1_2016_' + ind + '_AS' + '_w' + str(std) + '.npy')
            if mask is True:
                mask_snow = np.load(anc_direct + 'snow_mask_360s.npy')
                onset0 = np.ma.masked_array(onset0, mask=[mask_snow == 0])
                onset1 = np.ma.masked_array(onset1, mask=[mask_snow == 0])
            fpath1 = 'result_08_01/'
            pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname='onset_0_smap_' + ind + '_w' + str(std),
                           z_max=180, z_min=50, odd_points=odd_point)  # fpath1
            pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname='onset_1_smap_' + ind + '_w' + str(std),
                           z_max=360, z_min=250, odd_points=odd_point)


def prepare_onset(custom_name):
    input_onset = []
    for c_name in custom_name:
        value = np.load(c_name)
        if len(value.shape) > 1:
            if value.shape[1] == 9000:
                onset_tup = bxy.time_getlocaltime(value[0], ref_time=[2000, 1, 1, 0])
                onset_doy = onset_tup[-2].reshape(90, 100)
            elif value.shape[1] == 90000:
                onset0 = onset0[0].reshape(300, 300)
        if 'melt' in c_name:
            onset_doy[onset_doy > 150] = 0
        input_onset.append(onset_doy)
    return input_onset


def check_shape(onset0):
    if len(onset0.shape) > 1:
        if onset0.shape[1] == 9000:
            onset_tup = bxy.time_getlocaltime(onset0[0], ref_time=[2000, 1, 1, 0])
            onset_doy = onset_tup[-2].reshape(90, 100)
        elif onset0.shape[1] == 90000:
            onset0 = onset0[0].reshape(300, 300)
    return onset_doy


def ascat_result_test(area, key='AS', mode='_norm_', odd_rc=([], []), ft='0', version='new', std=4):
    lons_grid, lats_grid = np.load('./result_05_01/onset_result/lon_ease_grid.npy'), \
                           np.load('./result_05_01/onset_result/lat_ease_grid.npy')
    arctic = lats_grid > 66.5
    # check the odd value
    if version == 'new':
        result_doc = 'all_year_observation'
    elif version == 'old':
        result_doc = 'all_year_observation_old'
    prefix = './result_05_01/onset_result/%s/ascat_onset_' % result_doc
    onset_path = prefix + ft + '_2016' + mode + key + '_w%s.npy' % str(std)
    onset_ak = np.load(onset_path)
    if area == 'area_0':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (onset_ak < 60) & (lons_grid < -155) & (lons_grid > -160)
    elif area == 'area_1':
        odd_thaw = (onset_ak > 0) & (lats_grid < 60) & (onset_ak < 80) & (lons_grid < -160)
    elif area == 'area_2':
        odd_thaw = (lats_grid > 55) & (lats_grid < 60) & (lons_grid < -155) & (lons_grid > -160) & (onset_ak < 75) & (
        onset_ak > 0)
    elif area == 'area_3':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid < -150) & (lons_grid > -155) & (onset_ak > 140) & (
        onset_ak > 0)
    elif area == 'area_4':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid > -150) & (lons_grid < -145) & (onset_ak < 100)
    elif area == 'area_5':
        odd_thaw = (lats_grid > 60) & (lats_grid < 63) & (lons_grid > -160) & (lons_grid < -155) & (onset_ak > 120)
    elif area == 'area_6':
        odd_thaw = (lats_grid > 65) & (lats_grid < 67) & (lons_grid > -160) & (lons_grid < -155) & (onset_ak > 130) & (
        onset_ak > 0)
    elif area == 'area_7':
        odd_thaw = (lats_grid > 66) & (lats_grid < 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_ak > 135) & (
        onset_ak > 0)
    elif area == 'area_8':
        odd_thaw = (lats_grid > 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_ak > 120) & (onset_ak > 0)
    elif area == 'area_88':
        odd_thaw = (lats_grid > 70) & (lons_grid > -155) & (lons_grid < -150) & (onset_ak > 140) & (onset_ak > 0)
    elif area == 'area_9':
        odd_thaw = (lats_grid > 67) & (lats_grid < 70) & (lons_grid > -150) & (lons_grid < -145) & (onset_ak > 150) & (
        onset_ak < 179)
    elif area == 'area_5f':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid > -145) & (lons_grid < -140) & (onset_ak > 345) & (
        onset_ak > 0)
    elif area == 'area_11':
        odd_thaw = (lats_grid > 68) & (lats_grid < 70) & (lons_grid > -145) & (lons_grid < -140) & (onset_ak < 140) & (
        onset_ak > 0)
    elif area == 'area_12':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -163) & (lons_grid < -160) & (onset_ak > 125) & (
        onset_ak < 150)
    id_thaw = np.where(odd_thaw)
    odd_row, odd_col = id_thaw[0], id_thaw[1]
    sigma_path = ('./result_05_01/onset_result/%s/ascat_all_2016' % result_doc) + mode + key + '.npy'
    sigma_3d = np.load(sigma_path)
    inc_path = ('./result_05_01/onset_result/%s/ascat_inc_2016_' % result_doc) + key + '.npy'
    inc_3d = np.load(inc_path)
    odd_sigma = sigma_3d[odd_thaw]
    odd_inc = inc_3d[odd_thaw]
    odd_lon, odd_lat = lons_grid[odd_thaw], lats_grid[odd_thaw]
    id_odd = np.arange(0, odd_lon.size)
    odd_value = onset_ak[odd_thaw]
    np.save('./result_05_01/onset_result/odd_thaw_' + area, odd_sigma)
    np.save('./result_05_01/onset_result/odd_thaw_inc' + area, odd_inc)
    # np.save()
    fpath = './result_08_01/onset_result/odd_thaw_%s.txt' % area
    np.savetxt(fpath,
               np.array([id_odd.T, odd_value.T, odd_lon.T, odd_lat.T, odd_row.T, odd_col.T]).T,
               fmt='%d, %d, %.8f, %.8f, %d, %d')

    # find the rows and cols number in smap_map
    dis_grid = bxy.cal_dis(odd_rc[1], odd_rc[0], lats_grid, lons_grid)
    min_dis = np.min(dis_grid)
    row_col = np.where(dis_grid == min_dis)
    p_id = 37
    # ascat_test_odd_point_plot('./result_05_01/onset_result/odd_thaw_'+area+'.npy',
    #                           './result_05_01/onset_result/odd_thaw_'+area+'.txt',
    #                           p_id, area=area, orb=key,
    #                           odd_point=[sigma_3d[row_col[0][0], row_col[1][0]], lons_grid[row_col], lats_grid[row_col], odd_rc[0], odd_rc[1], min_dis],
    #                           mode=mode, std=std, start_date='20151001')
    # test angular dependency
    # id_angular = odd_sigma[p_id] != 0
    # y_sig = odd_sigma[p_id][id_angular]
    # x_inc = odd_inc[p_id][id_angular]
    # ascat_test_odd_angular(x_inc, y_sig, area=area)
    return 0


def smap_result_test(area, orbit='AS', odd_rc=([], []), ft='0', mode='tb', std=4, version='old', ini_doy=1):
    lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy'), \
                           np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
    # check the odd value
    if version == 'new':
        result_doc = 'all_year_observation'
    elif version == 'old':
        result_doc = 'all_year_observation_old'
    prefix = './result_05_01/onset_result/%s/' % result_doc
    # file path:
    series_tbv_2016 = prefix + 'smap_all_2016_tbv_' + orbit + '.npy'
    series_tbh_2016 = prefix + 'smap_all_2016_tbh_' + orbit + '.npy'
    odd_value_path = prefix + 'odd_value_smap' + '.txt'
    onset_value = prefix + 'smap_onset_%s_2016_%s_%s_w%s.npy' % (
    ft, mode, orbit, str(std))  # smap_onset_0_2016_npr_AS_w4.npy
    odd_series = prefix + 'series/' + area + orbit
    fsize = [8, 5]
    # Test onset by specifying an area
    onset_0_file = np.load(onset_value)
    if area == 'area_0':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (onset_0_file < 80) & (lons_grid < -145) & (lons_grid > -150)
    elif area == 'area_1':
        odd_thaw = (onset_0_file > 0) & (lats_grid < 60) & (onset_0_file < 80) & (lons_grid < -160)
    elif area == 'area_2':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid < -145) & (lons_grid > -150) & (
        onset_0_file > 110) & (onset_0_file > 0)
    elif area == 'area_3':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid < -150) & (lons_grid > -155) & (
        onset_0_file > 140) & (onset_0_file > 0)
    elif area == 'area_4':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid > -150) & (lons_grid < -145) & (onset_0_file < 100)
    elif area == 'area_5':
        odd_thaw = (lats_grid > 70) & (lats_grid < 80) & (lons_grid > -155) & (lons_grid < -150) & (onset_0_file > 130)
    elif area == 'area_6':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -160) & (lons_grid < -155) & (
        onset_0_file > 300) & (onset_0_file > 0)
    elif area == 'area_7':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -160) & (lons_grid < -155) & (
        onset_0_file > 100) & (onset_0_file > 0)
    elif area == 'area_8':
        odd_thaw = (lats_grid > 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_0_file > 120) & (
        onset_0_file > 0)
    elif area == 'area_88':
        odd_thaw = (lats_grid > 70) & (lons_grid > -155) & (lons_grid < -150) & (onset_0_file > 140) & (
        onset_0_file > 0)
    elif area == 'area_9':
        odd_thaw = (lats_grid > 67) & (lats_grid < 70) & (lons_grid > -150) & (lons_grid < -145) & (
        onset_0_file > 150) & (onset_0_file < 179)
    elif area == 'area_100':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -145) & (lons_grid < -140) & (
        onset_0_file > 300) & (onset_0_file > 0)
    elif area == 'area_11':
        odd_thaw = (lats_grid > 68) & (lats_grid < 70) & (lons_grid > -145) & (lons_grid < -140) & (
        onset_0_file < 140) & (onset_0_file > 0)
    id_thaw = np.where(odd_thaw)

    # read data
    odd_row, odd_col = id_thaw[0], id_thaw[1]
    tbh_3d = np.load(series_tbh_2016)
    tbv_3d = np.load(series_tbv_2016)  ####
    if mode == 'tb':
        indicator_3d = tbv_3d
    else:
        indicator_3d = (tbv_3d - tbh_3d) / (tbv_3d + tbh_3d)
    odd_sigma = indicator_3d[odd_thaw]
    odd_lon, odd_lat = lons_grid[odd_thaw], lats_grid[odd_thaw]
    id_odd = np.arange(0, odd_lon.size)
    odd_value = onset_0_file[odd_thaw]
    np.savetxt(odd_value_path,
               np.array([id_odd.T, odd_value.T, odd_lon.T, odd_lat.T, odd_row.T, odd_col.T]).T,
               fmt='%d, %d, %.8f, %.8f, %d, %d')

    # find the rows and cols number in smap_map
    dis_grid = bxy.cal_dis(odd_rc[1], odd_rc[0], lats_grid, lons_grid)
    min_dis = np.min(dis_grid)
    row_col = np.where(dis_grid == min_dis)  # row_col[0][0] is row, row_col[1][0] is col
    # odd_point=[indicator_3d[odd_target], lons_grid[odd_target], lats_grid[odd_target], odd_target[0], odd_target[1]]
    # odd_info = np.loadtxt(area_file, delimiter=',')
    # sig = np.load(sig_file)
    # odd_onset = odd_info[:, 1]
    # id of testing pixel
    # p_id = np.argsort(odd_onset)[odd_onset.size//2]
    # p_id = 37

    v, loni, lati, rowi, coli, dis_i = indicator_3d[row_col[0][0], row_col[1][0]], \
                                       odd_rc[0], odd_rc[1], \
                                       row_col[0][0], row_col[1][0], min_dis
    print 'the odd pixel is (%.5f, %.5f)' % (loni, lati), '\n'
    if mode == 'tb':
        for pol in ['V']:
            if pol == 'H':
                v = tbh_3d[odd_rc]
            iv_valid = (v > 100)
            v_valid = v[iv_valid]
            t = np.arange(ini_doy, v.size) - 1
            t_valid = t[iv_valid]
            g_size = 3 * std
            g_sig, ig2 = gauss_conv(v_valid, sig=std, size=2 * g_size + 1)
            g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size])) \
                          / (np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
            max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, t_valid[g_size: -g_size])
            onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=mode)
            # test the temporal variation
            # print 'the temporal change of tb%s is: ' % pol, v_valid[g_size: -g_size]
            fig = pylt.figure(figsize=fsize)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d, Days: %d, distance: %.1f'
                         % (loni, lati, rowi, coli, onset[0], onset[1], sum(v < 0), min_dis))
            plot_funcs.pltyy(t_valid, v_valid, 'test', '$T_{B%s}$ (K)' % pol,
                             t2=t_valid[ig2][g_size: -g_size], s2=g_sig[g_size: -g_size],
                             ylim=[230, 280], symbol=['bo', 'g-'], handle=[fig, ax])
            ax.set_xlim([0, 365])
            pylt.savefig(odd_series + '_' + pol + '_w' + str(std), dpi=300)
    elif mode == 'npr':
        iv_valid = (v != 0) & (v != 1) & (v != -1) & (v != -0) & (v != -9999) & (~np.isnan(v))
        # iv_valid = ((v != 0) & (v != -9999))
        # iv_valid = (v != 0) & (v > -999)
        v_valid = v[iv_valid]
        t = np.arange(ini_doy, v.size) - 1
        t_valid = t[iv_valid]
        g_size = 3 * std
        g_sig, ig2 = gauss_conv(v_valid, sig=std, size=2 * g_size + 1)  # non-smoothed
        g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size])) \
                      / (np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
        max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig[g_size: -g_size], 1e-4, t_valid[g_size: -g_size])
        onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=mode)
        print 'onset in odd_point test:', onset
        # the temporal change of npr/tb
        id_test = int(max_gsig_s[max_gsig_s[:, 1] == onset[0]][0, 0])
        # test the temporal variation
        print 'the temporal change of npr is: ', v_valid[g_size: -g_size][id_test - 1: id_test + 7]
        fig = pylt.figure(figsize=fsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d distance: %.1f'
                     % (loni, lati, rowi, coli, onset[0], onset[1], min_dis))
        plot_funcs.pltyy(t_valid, v_valid, odd_series + '_' + mode + '_w' + str(std), '$NPR$ (%)',
                         t2=t_valid[ig2][g_size: -g_size], s2=g_sig[g_size: -g_size],
                         ylim=[0, 1.2 * np.max(v_valid)], symbol=['bo', 'g-'], handle=[fig, ax])
        ax.set_xlim([0, 365])
        pylt.savefig(odd_series + '_' + mode + '_w' + str(std), dpi=300)


def ascat_test_odd_point_plot(sig_file, area_file, p_id, area='odd_arctic1', odd_point=[], orb='AS', mode=[],
                              ptype='sig', std=4, start_date='20160101'):
    prefix = './result_05_01/onset_result/odd_series_smap/'
    prefix_new = './result_05_01/onset_result/all_year_observation/'
    odd_info = np.loadtxt(area_file, delimiter=',')
    sig = np.load(sig_file)
    # odd_onset = odd_info[:, 1]
    # id of testing pixel
    # p_id = np.argsort(odd_onset)[odd_onset.size//2]
    # p_id = 37
    if len(odd_point) > 0:
        v, loni, lati, rowi, coli, dis = odd_point[0], odd_point[1], odd_point[2], odd_point[3], odd_point[4], \
                                         odd_point[5]
    else:
        v, loni, lati, rowi, coli = sig[p_id], odd_info[:, 2][p_id], odd_info[:, 3][p_id], \
                                    odd_info[:, 4][p_id], odd_info[:, 5][p_id]
    print 'the odd pixel is (%.5f, %.5f)' % (loni, lati), '\n'
    iv_valid = v != 0
    v_valid = v[iv_valid]
    t = np.arange(0, v.size) + get_doy(start_date) - 365 - 1
    t_valid = t[iv_valid]
    g_size = 3 * std
    g_sig, ig2 = gauss_conv(v_valid, sig=std, size=2 * g_size + 1)
    # g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
    #             /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))

    g_sig_valid = g_sig[g_size: -g_size]
    max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, t_valid[g_size: -g_size])
    onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=ptype)
    fig = pylt.figure(figsize=[8, 5])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d, Days: %d, distance: %.1f' % (
    loni, lati, rowi, coli, onset[0], onset[1], sum(v < 0), dis))
    print prefix + area + orb
    plot_funcs.pltyy(t_valid, v_valid, prefix_new + 'series/' + area + orb + '_w' + str(std), '$\sigma_0$ (dB)',
                     t2=t_valid[ig2][g_size: -g_size], s2=g_sig_valid,
                     ylim=[1.2 * np.min(v_valid), 0.8 * np.max(v_valid)], symbol=['bo', 'g-'], handle=[fig, ax])
    ax.set_xlim([0, 365])
    pylt.savefig(prefix_new + 'series/' + area + orb + '_w' + str(std), dpi=300)

    return 0


def ascat_test_odd_angular(inc, sig, t=None, period=[], area='test'):
    if len(period) > 0:
        id = (t > period[0]) & (t < period[1])
        sig, inc = inc[id], sig[id]
    a, b = np.polyfit(inc, sig, 1)
    f = np.poly1d([a, b])
    fig = pylt.figure(figsize=[4, 3])
    ax = fig.add_subplot(111)
    sig_mean = np.sum(sig) / sig.size
    sstot = np.sum((sig - sig_mean) ** 2)
    ssres = np.sum((sig - f(inc)) ** 2)
    r2 = 1 - ssres / sstot
    fig.text(0.15, 0.15, '$y = %.2f x + %.f$\n $r^2 = %.4f$' % (a, b, r2))
    ax.plot(inc, sig, 'k.')
    plot_funcs.plt_more(ax, inc, f(inc), symbol='r-',
                        fname='./result_05_01/onset_result/odd_series/odd_angular_' + area)
    fig.clear()
    pylt.close()


def ascat_gap(ascat_series, id):
    d0 = np.unique(ascat_series[:, 0])
    d1 = np.arange(ascat_series[:, 0][0], d0.size + ascat_series[:, 0][0])
    fig = pylt.figure(figsize=[8, 2])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(d0, d0 - d1, '.')
    pylt.savefig('./result_05_01/point_result/ascat/data_gap' + id + '.png')


def ascat_check_single_pixel():
    return 0


def ascat_map_range():
    lons_grid, lats_grid = np.load('./result_05_01/onset_result/lon_ease_grid.npy'), \
                           np.load('./result_05_01/onset_result/lat_ease_grid.npy')
    print lons_grid[0], lats_grid[0]
    onset = np.load('./result_05_01/onset_result/ascat_onset_0_2016_norm_AS.npy')
    # np.savetxt('./result_05_01/onset_result/maps/onset_0_norm_AS.txt', np.array([lons_grid.T, lats_grid.T, onset.T]).T, delimiter=',', fmt='%.5f, %.5f, %d')


def tiff_read():
    ds = gdal.Open('NIC.tif', gdal.GA_ReadOnly)
    arys = []
    for i in xrange(1, ds.RasterCount + 1):
        arys.append(ds.GetRasterBand(i).ReadAsArray())
    arys = np.concatenate(arys)
    return 0


def bbox_find(lat_array, lon_array, lats, lons):
    row, col = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    array_shape = lat_array.shape
    lat_array1d, lon_array1d = lat_array.ravel(), lon_array.ravel()
    for i in [0, 1]:
        lat0, lon0 = lats[i], lons[i]  # up left and down right
        dis0 = bxy.cal_dis(lat0, lon0, lat_array1d, lon_array1d)
        id0 = np.argmin(dis0)
        row[i] = id0 / array_shape[1]
        col[i] = id0 - id0 / array_shape[1] * array_shape[1]
    print row, col
    return row, col


def plot_tbtxt(site_no, orb, txt_name, att_name, prefix='./result_07_01/'):
    att = site_infos.get_attribute(sublayer='smap_tb')
    tbs = np.loadtxt(txt_name)
    fig0 = pylt.figure(figsize=[10, 8])
    i = 0
    axs = []
    for name0 in att_name:
        i += 1
        att_id = att[1].index(name0) + 1
        x, y = tbs[:, 0], tbs[:, att_id]
        ax0 = fig0.add_subplot(3, 2, i)
        if name0 != 'cell_tb_time_seconds_aft':
            ax0.plot(x, y, 'ko')
        else:
            sec_2016 = datetime.datetime.strptime('20160101 00:00:00', '%Y%m%d %X') \
                       - datetime.datetime.strptime('20000101 11:58:56', '%Y%m%d %X')
            sec_now = y - sec_2016.total_seconds()
            pass_hr = np.modf(sec_now / (24 * 3600.0))[0] * 24
            n, bins, patches = ax0.hist(pass_hr, 50, normed=1, facecolor='green', alpha=0.75)
        ax0.set_ylabel(name0)
        axs.append(ax0)
    pylt.savefig(prefix + 'ancplot/' + site_no + orb + 'anc.png')


def cal_obd(ass, des, as_time, des_time):
    """

    :param ass:
    :param des:
    :param as_time:
    :param des_time:
    :return: orbit difference, date of difference, pass hours of ascending and descending (UTC time)
    """
    # rmv unvalid
    ass[1][ass[1] < -9000], des[1][des[1] < -9000] = np.nan, np.nan
    # find the same date
    id_as = np.in1d(ass[0], des[0])
    date_as = ass[0][id_as]
    id_des = np.in1d(des[0], date_as)
    date_des = des[0][id_des]
    if all(date_as == date_des) is not True:
        print 'the number of ascending passes dont equal the descending'
        return -1, -1, -1
    # cal the difference
    obd = des[1][id_des] - ass[1][id_as]
    # keep the pass time
    time_as, time_des = as_time[id_as], des_time[id_des]
    return obd, date_as, [time_as, time_des]


def plot_obd(site_no, p='v', passvalid=False, isplot=True):
    prefix = './result_07_01/'
    # pm/as, am/des
    as_fname, des_fname = prefix + 'txtfiles/site_tb/tb_' + site_no + '_A_2016.txt', prefix + 'txtfiles/site_tb/tb_' + site_no + '_D_2016.txt'
    site_as = np.loadtxt(as_fname)
    site_des = np.loadtxt(des_fname)
    with open(as_fname, 'rb') as as0:
        reader = csv.reader(as0)
        for row in reader:
            if '#' in row[0]:
                n_time = row.index('cell_tb_time_seconds_aft')
                n_tbv = row.index('cell_tb_v_aft')
                n_tbh = row.index('cell_tb_h_aft')
                break
    as0.closed
    time_as, time_des = site_as[:, n_time], site_des[:, n_time]
    # test strange time
    # hr_tp = np.modf(time_as/24/3600)[0]

    # transform utc to local pass time
    sec_as, sec_des = bxy.timetransform(time_as, '20000101 11:58:56', '%Y%m%d %X', tzone=True), \
                      bxy.timetransform(time_des, '20000101 11:58:56', '%Y%m%d %X',
                                        tzone=True)  # seconds from 20150101 (utc or local)
    pass_as, pass_des = np.modf(sec_as / (24 * 3600.0))[0] * 24, np.modf(sec_des / (24 * 3600.0))[0] * 24
    as_date, des_date = np.modf(sec_as / (24 * 3600.0))[1] + 1, np.modf(sec_des / (24 * 3600.0))[1] + 1
    tbv_a, tbv_d = [as_date, site_as[:, n_tbv]], [des_date, site_des[:, n_tbv]]
    tbh_a, tbh_d = [as_date, site_as[:, n_tbh]], [des_date, site_des[:, n_tbh]]
    # check the passtime distributionn
    # fig = pylt.figure(figsize=[5, 5])
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot()
    if p == 'v':
        obd, tb_date, pass_hr = cal_obd(tbv_a, tbv_d, pass_as, pass_des)
    elif p == 'h':
        obd, tb_date, pass_hr = cal_obd(tbh_a, tbh_d, pass_as, pass_des)
    elif p == 'npr':
        npr_a, npr_d = (tbv_a[1] - tbh_a[1]) / (tbv_a[1] + tbh_a[1]), (tbv_d[1] - tbh_d[1]) / (tbv_d[1] + tbh_d[1])
        obd, tb_date, pass_hr = cal_obd(np.array([tbh_a[0], npr_a]), np.array([tbh_d[0], npr_d]), pass_as, pass_des)
    elif p == 'vh':
        obd_v, tb_date, pass_hr = cal_obd(tbv_a, tbv_d, pass_as, pass_des)
        obd_h, tb_date, pass_hr = cal_obd(tbh_a, tbh_d, pass_as, pass_des)
    elif p == 'vh0':
        obd_v, tb_date, pass_hr = cal_obd(tbv_a, tbv_d, pass_as, pass_des)
        obd_h, tb_date, pass_hr = cal_obd(tbh_a, tbh_d, pass_as, pass_des)
    elif p == 'sig':
        sigconv, sm, tsoil, swe, sigseries, ons_new, pass_as_ascat = \
            ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g=5, pp=False)  # 0 for ascending
        sigconv_d, sm_d, tsoil_d, swe_d, sigseries_d, ons_new_d, pass_des_ascat = \
            ascat_plot_series(site_no, orb_no=1, inc_plot=False, sigma_g=5, pp=False)
        obd_sig, obd_date, pass_hr = cal_obd(sigseries, sigseries_d, pass_as_ascat, pass_des_ascat)
        tb_date = obd_date
    # plotting
    # read site data
    stats_sm, sm5 = read_site.read_sno('', "snow", site_no, field_no=-1)
    if passvalid:
        passtime = np.round(pass_hr)
    else:
        passtime = [0, 0]
    site_date = np.arange(366, 366 + 365)
    sm_as, sm_date_as = cal_emi(sm5, ' ', site_date, hrs=passtime[0])
    sm_des, sm_date_des = cal_emi(sm5, ' ', site_date, hrs=passtime[1])
    mm_daily, mm_doy = read_site.read_measurements(site_no, "snow", site_date)
    sm_as[sm_as < -90], sm_des[sm_des < -90] = np.nan, np.nan

    # plotting parameters
    # tb_date, obd_v, obd_h; p1

    sm_date365 = sm_date_as.astype(int) - 365  # p3
    sm_change = np.diff(sm_as)
    idd = np.in1d(sm_date365[1:], tb_date)
    tb_date -= 365
    # comments on 07/26
    # if isplot:
    #     fig = pylt.figure(figsize=[8, 8])
    #     ax0 = fig.add_subplot(3, 1, 3)
    #     if p == 'vh':
    #         p0, = ax0.plot(tb_date, obd_v, 'o', markersize=3)
    #         p0_1 = plot_funcs.plt_more(ax0, tb_date, obd_h, line_list=[p0], symbol='ro')
    #         ax0.legend([p0_1[0], p0_1[1]], ['V pol.', 'H pol.'], loc=0, prop={'size': 8})
    #         ax0.set_ylim([-15, 15])
    #     elif p == 'vh0':
    #         ax0.plot(tb_date, obd_h-obd_v, 'o', markersize=3)
    #         ax0.set_ylim([-2, 2])
    #     elif p == 'sig':
    #         tb_date+=365
    #         ax0.plot(obd_date, obd_sig, 'o', markersize=3)
    #     if p=='npr':
    #         ax0.set_ylim([-0.01, 0.01])
    #     ax0.axhline(ls=':', lw=1.5)
    #
    #     ax1 = fig.add_subplot(3, 1, 2)
    #     p1, = ax1.plot(tb_date, pass_hr[0], 'o')
    #     p1_2 = plot_funcs.plt_more(ax1, tb_date, pass_hr[1], line_list=[p1], symbol='r+')
    #     ax1.legend([p1_2[0], p1_2[1]], ['as_hour', 'des_hour'], loc=0, prop={'size': 8})
    #
    #     ax2 = fig.add_subplot(3, 1, 1)
    #     t = sm_date_as.astype(int)-365
    #     # p1, = ax2.plot(t, sm_as)
    #     # t = sm_date_des.astype(int)-365
    #     # p1_2 = plot_funcs.plt_more(ax2, t, sm_des, line_list=[p1], symbol='r-')
    #     # ax2.legend([p1_2[0], p1_2[1]], ['as_VWC', 'des_VWC'], loc=0, prop={'size': 8})
    #
    #     # d, t, hr = cal_obd([np.fix(sm_date_as), sm_as], [np.fix(sm_date_des), sm_des], sm_as, sm_des)
    #     # t-=365
    #     # ax2.plot(t, d, 'o')
    #     # ax2.axhline(ls=':', lw=1.5)
    #
    #     # else:
    #     #     d, t, hr = cal_obd([np.fix(sm_date_as), sm_as], [np.fix(sm_date_des), sm_des], sm_as, sm_des)
    #     #     t-=365
    #     #     ax2.plot(t, d, 'o')
    #     #     ax2.axhline(ls=':', lw=1.5)
    #     ax2.set_xlim([0, 400])
    #     save_prefix = './result_08_01/'
    #     fig.savefig(save_prefix+site_no+'_'+p+'_obd.png')
    #     pylt.close()
    # else:
    #     print 'no plotting'
    heads = 'date, orbit_difference_v, orbit_difference_h'
    # if p=='vh':
    #     np.savetxt('./result_07_01/for_comparison/tp/obd_tb'+site_no+'.txt', np.array([tb_date, obd_v, obd_h]).T,
    #               fmt='%d, %.2f, %.2f', delimiter=',', header=heads)

    # return: tbv, tbh, obd_v, obd_h, swe, dswe

    return [tb_date, obd_v], [tb_date, obd_h], [sm_date365, sm_des], [sm_date365[1:], sm_change]


def save_transition(site_no, k_width, peakdate, gau, sm, tsoil, swe, layers='/thaw/npr', period=[0, 150]):
    """

    :param site_no:
    :param k_width:
    :param peakdate: 0: max, 1: min
    :param gau:
    :param sm:
    :param tsoil:
    :param swe:
    :param layers:
    :return:
    """
    if layers.find('freeze') > -1:  # for freezing method
        period = [0, 350]
        if layers.find('tb') > 0:
            date_npr_Emax = peakdate[0][:, 1]
            norm_v = peakdate[0][:, 2]
        else:
            date_npr_Emax = peakdate[1][:, 1]
            norm_v = peakdate[1][:, 2]
    else:
        if layers.find('tb') > 0:
            date_npr_Emax = peakdate[1][:, 1]  # minimum for tb
            norm_v = peakdate[1][:, 2]
        else:
            date_npr_Emax = peakdate[0][:, 1]  # the date when Conv of NPR reach maximum
            norm_v = peakdate[0][:, 2]
    window0 = (date_npr_Emax > period[0]) & (date_npr_Emax < period[1])  # window for thawing: 0~150, freezing: 250 ~350
    max_date_npr = date_npr_Emax[window0]
    i0_npr_Et = np.in1d(gau[0], max_date_npr)
    max_trans = [gau[0][i0_npr_Et], gau[1][i0_npr_Et],
                 gau[2][i0_npr_Et]]  # maximum during transition (minimum for tb): date, norm vlaue, non_norm value
    i_Et_05 = max_trans[1] >= -1
    # the extremum date, normaled extremum, non-normaled ex
    max_date_npr, Emax, Emax_non = max_trans[0][i_Et_05], max_trans[1][i_Et_05], max_trans[2][i_Et_05]
    max_dsm_npr, max_soil_npr, max_swe_npr = \
        np.zeros(max_date_npr.size), np.zeros(max_date_npr.size), np.zeros(max_date_npr.size)
    # some in situ data
    i_dsm = 0
    for d0 in max_date_npr:
        i_window_swe = [np.argmin(np.abs(np.fix(swe[0] - d0 + np.fix(3 * k_width)))),
                        np.argmin(np.abs(np.fix(swe[0] - d0 - np.fix(3 * k_width))))]
        i_window_sm = [np.argmin(np.abs(np.fix(sm[0] - d0 + np.fix(3 * k_width)))),
                       np.argmin(np.abs(np.fix(sm[0] - d0 - np.fix(3 * k_width))))]
        i_window_soil = [np.argmin(np.abs(np.fix(tsoil[0] - d0 + np.fix(3 * k_width)))),
                         np.argmin(np.abs(np.fix(tsoil[0] - d0 - np.fix(3 * k_width))))]
        dsm = (sm[1][i_window_sm[1]] - sm[1][i_window_sm[0]])
        dsoil = np.nanmean(tsoil[1][i_window_soil[0]: i_window_soil[1] + 1])
        dswe = (swe[1][i_window_swe[1]] - swe[1][i_window_swe[0]])

        max_dsm_npr[i_dsm], max_soil_npr[i_dsm], max_swe_npr[i_dsm] = dsm, dsoil, dswe
        i_dsm += 1
    # save results during transition
    filedoc = 'result_07_01/methods/'
    h5name1 = 'transition_' + site_no + '_v00.h5'
    narr1 = np.array([max_date_npr, Emax, Emax_non, max_dsm_npr, max_soil_npr,
                      max_swe_npr])  # date, value, change of sm, mean t and change of swe
    print [layers + "/width_" + str(k_width)]
    bxy.h5_writein(filedoc + h5name1, layers + "/width_" + str(k_width), narr1)
    return 0


def test_winter_trans(site_no, indic='thaw/tb', w=[1, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], trans_date=[80, 80]):
    """
    Test the extremum of Gaussian Conv in winter, which is considered as noise-inducing edge
    :param h5name:
    :param ind0:
    :param w:
    :return:
    """
    h5name = site_infos.get_site_h5(site_no)
    trans2winter = np.zeros(len(w))
    thaw_stds = np.zeros(len(w))
    win_stds = np.zeros(len(w))  # above are the snr, std of convolution during thawing, and that during winter
    i0 = -1
    # the indicator: thaw or freeze
    ft = indic.split('/')
    if ft[0] == 'thaw':
        trans_date = [70, 80, 150]
    else:
        trans_date = [70, 260, 350]
    # thaw
    # if indic!='tb':
    for w0 in w:
        i0 += 1

        # if w0 == 4:
        #     pause = 0
        #
        # h1 = h0[indic+'/width_'+str(w0)]
        # conv_attname = 'conv/%s/width_%s' % (ft[1], str(w0))
        # # conv_attname = 'all_2016/%s' % ft[1]
        # h_conv = h0[conv_attname]
        # if (site_no == '947') & (w0 == 9):
        #     pause = 0
        # # get the extrema in winter
        # date_extremum = h1[0]
        # extremum = h1[2]
        # # get the conv_series in winter:
        # conv_date = h_conv[0]
        # convs = h_conv[-1]
        # conv_win = convs[(conv_date>30) & (conv_date<trans_date[0])]
        # conv_std_win = np.nanstd(conv_win)

        # new added
        if ft[1] == 'ascat':
            conv0, series, ons_new, gg, sig_pass, peakdate = \
                ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g=w0, pp=False, order=1)
            conv0[0] -= 365
        elif ft[1] == 'tb':
            series, tbh1, npr1, conv0, ons1, sitetime, peakdate = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                                tbob='_A_', sig0=w0, order=1,
                                                                                seriestype='tb')  # result npr
            conv0[-1] *= -1
            tp = [peakdate[1], peakdate[0]]
            peakdate = tp
        else:
            tbv1, tbh1, series, conv0, ons1, sitetime, peakdate = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                                tbob='_A_', sig0=w0,
                                                                                order=1)  # result npr
        if ft[0] == 'freeze':
            conv0[-1] *= -1
            peaksz = peakdate[1].T
        elif ft[0] == 'thaw':
            peaksz = peakdate[0].T
        conv_date = conv0[0]
        convs = conv0[-1]
        conv_win = convs[(conv_date > 10) & (conv_date < trans_date[0])]
        conv_std_win = np.nanstd(conv_win)

        extremun_date = peaksz[1]
        extremum = peaksz[-1]
        window_trans = (extremun_date > trans_date[1]) & (extremun_date < trans_date[2])
        np.set_printoptions(precision=5, suppress=True)
        conv_trans = convs[((conv_date > 75) & (conv_date < trans_date[2])) | (
        (conv_date > 260) & (conv_date < 350))]  # thawing transition
        conv_trans_std = np.nanstd(conv_trans)
        conv_thaw = convs[(conv_date > 75) & (conv_date < trans_date[2])]
        conv_thaw_std = np.nanstd(conv_thaw)
        print 'Filter width: ', w0
        # print 'winter: ',extremum[window_winter], '\n', 'mean: ', mean_winter, 'std: ', np.std(extremum[window_winter]), \
        #     '\n' 'transition: ', extremum[window_trans][ind_trans_valid], \
        #     'mean: ', np.nanmean(extremum[window_trans][ind_trans_valid]), '\n', 'transition date are: ', date_extremum[window_trans][ind_trans_valid], '\n'
        # trans2winter[i0] = np.abs(np.nanmean(extremum[window_trans][ind_trans_valid])/mean_winter)

        # trans2winter[i0] = np.abs(np.nanmean(extremum[window_trans])/conv_std_win)
        trans2winter[i0] = np.abs(conv_trans_std / conv_std_win)
        thaw_stds[i0] = conv_thaw_std
        win_stds[i0] = conv_std_win
    return trans2winter, [win_stds, thaw_stds]


def interp_radius_v2(filename, site_no, prj='np', dat='smap_tb', disref=0.5):
    s_info = site_infos.change_site(site_no)
    attributes = 'North_Polar_Projection'
    atts = site_infos.get_attribute('np', sublayer=dat)
    hf_l0 = h5py.File(filename, 'r')
    # the lat and lon of all available tb pixel
    # print filename[48: ]
    if attributes not in hf_l0:
        hf_l0.close()
        stat = -1
        print '%s was not included in \n %s' % (attributes, filename[48:])
        return [-9999 + i * 0 for i in range(len(atts[1]))], -9999
    hf = hf_l0[attributes]  # open second layer
    lat = hf['cell_lat'].value
    lon = hf['cell_lon'].value
    # passing time
    time_str = hf['cell_tb_time_utc_aft'].value
    pass_hour = np.zeors(time_str.size) - 1
    p0 = 0
    for str0 in time_str:
        date_obj = datetime.strptime(str0, '%Y-%m-%dT%H:%M:%S.%fZ')
        pass_hour[p0] = date_obj.timetuple().tm_hour
    uni_hour = np.unique(pass_hour)  # the unique pass hour for a given station
    dis = bxy.cal_dis(s_info[1], s_info[2], lat, lon)
    inner = np.where(dis < disref)
    dis_inner = dis[inner[0]]
    interpolated_value = []
    if dis_inner.size > 0:  # not empty
        inner_id = inner[0]
        if dis_inner.size > 1:
            print '%d pixels:' % dis_inner.size, dis_inner, ' within the r=%d km' % disref
            inner_id = inner[0][np.argmin(dis_inner)]
            if site_no == '1177':
                inner_id = inner[0][0]
                print lat[inner_id], lon[inner_id]
        for atti in atts[1]:
            var = hf[atti].value
            var_inner = var[inner_id]
            v_interp = bxy.dis_inter(dis_inner, var_inner)
            interpolated_value.append(v_interp)
        return interpolated_value, dis_inner
    else:
        return [-9999 + i * 0 for i in range(len(atts[1]))], -9999


def read_h5_latlon(h5name, latlon, att, orb=1):  # orb 0 is the AM pass
    """
    :param h5name:
    :param latlon: 0: lon, 1: lat, 2: dis; use distance as unique id
    :param att:
    :return: att_dict_list: a list that contains the <attribute dict> read for target pixels
    """
    # initial
    att_value = np.zeros([latlon.shape[0], len(att) + 1]) - 1
    att_dict_list = []
    distance_id = []
    h0 = h5py.File(h5name)
    lats, lons = h0['Freeze_Thaw_Retrieval_Data/latitude'].value[orb].ravel(), \
                 h0['Freeze_Thaw_Retrieval_Data/longitude'].value[orb].ravel()
    for i, coord in enumerate(latlon.reshape(-1, 4)):
        index0 = (np.abs(lons - coord[1]) < 0.01) & (np.abs(lats - coord[2]) < 0.01)
        check_index0 = np.where(index0)
        jndex1 = np.where((np.abs(h0['Freeze_Thaw_Retrieval_Data/latitude'].value[orb] - coord[2]) < 0.01)
                          & (np.abs(h0['Freeze_Thaw_Retrieval_Data/longitude'].value[orb] - coord[1]) < 0.01))
        att_dict = {}
        for j, att0 in enumerate(att):
            value = h0[att0].value[orb].ravel()[index0]
            if value.size > 0.5:
                att_value[i, j] = h0[att0].value[orb].ravel()[index0]
            else:
                continue
        att_value[i, -1] = coord[3]
    return att_value, 1, 1


def check_amsr2():
    """
    keys:
    Area Mean Height
    Attitude Data
    Brightness Temperature (original,89GHz-A,H)
    Brightness Temperature (original,89GHz-A,V)
    Brightness Temperature (original,89GHz-B,H)
    Brightness Temperature (original,89GHz-B,V)
    Brightness Temperature (res06,10.7GHz,H)
    Brightness Temperature (res06,10.7GHz,V)
    Brightness Temperature (res06,18.7GHz,H)
    Brightness Temperature (res06,18.7GHz,V)
    Brightness Temperature (res06,23.8GHz,H)
    Brightness Temperature (res06,23.8GHz,V)
    Brightness Temperature (res06,36.5GHz,H)
    Brightness Temperature (res06,36.5GHz,V)
    Brightness Temperature (res06,6.9GHz,H)
    Brightness Temperature (res06,6.9GHz,V)
    Brightness Temperature (res06,7.3GHz,H)
    Brightness Temperature (res06,7.3GHz,V)
    Brightness Temperature (res06,89.0GHz,H)
    Brightness Temperature (res06,89.0GHz,V)
    Brightness Temperature (res10,10.7GHz,H)
    Brightness Temperature (res10,10.7GHz,V)
    Brightness Temperature (res10,18.7GHz,H)
    Brightness Temperature (res10,18.7GHz,V)
    Brightness Temperature (res10,23.8GHz,H)
    Brightness Temperature (res10,23.8GHz,V)
    Brightness Temperature (res10,36.5GHz,H)
    Brightness Temperature (res10,36.5GHz,V)
    Brightness Temperature (res10,89.0GHz,H)
    Brightness Temperature (res10,89.0GHz,V)
    Brightness Temperature (res23,18.7GHz,H)
    Brightness Temperature (res23,18.7GHz,V)
    Brightness Temperature (res23,23.8GHz,H)
    Brightness Temperature (res23,23.8GHz,V)
    Brightness Temperature (res23,36.5GHz,H)
    Brightness Temperature (res23,36.5GHz,V)
    Brightness Temperature (res23,89.0GHz,H)
    Brightness Temperature (res23,89.0GHz,V)
    Brightness Temperature (res36,36.5GHz,H)
    Brightness Temperature (res36,36.5GHz,V)
    Brightness Temperature (res36,89.0GHz,H)
    Brightness Temperature (res36,89.0GHz,V)
    Earth Azimuth
    Earth Incidence
    Land_Ocean Flag 6 to 36
    Land_Ocean Flag 89
    Latitude of Observation Point for 89A
    Latitude of Observation Point for 89B
    Longitude of Observation Point for 89A
    Longitude of Observation Point for 89B
    Navigation Data
    Pixel Data Quality 6 to 36
    Pixel Data Quality 89
    Position in Orbit
    Scan Data Quality
    Scan Time
    Sun Azimuth
    Sun Elevation
    :return:
    """
    h0 = h5py.File('test_amsr2.h5')
    for k0 in h0.keys():
        print k0
    h01 = h0['Latitude of Observation Point for 89A']
    h02 = h0['Attitude Data']
    print 'position of 89A has shape:', h01.value.shape
    return 0


def check_amsr2_result0(datestr, sn, orb):
    """
    Brightness Temperature (res36,89.0GHz,H)
    'Latitude of Observation Point for 89A'
    'Longitude of Observation Point for 89A'
    :return:
    """
    h5_newfile = 'AMSR2_l2r_%s_%s_%s.h5' % (datestr, sn, orb)
    hf0 = h5py.File('./tp/' + h5_newfile)
    for k0 in hf0.keys():
        print k0
    lat_test, lon_test = hf0['latitude_36GHz'], hf0['longitude_36GHz']
    print 'The nearby pixels:\n', lat_test.size, ' by ', \
        lon_test.size
    print 'latitude: ', np.mean(lat_test), np.min(lat_test), np.max(lat_test)
    print 'longitude: ', np.mean(lon_test), np.min(lon_test), np.max(lon_test)
    print 'The tb values are:\n', hf0['Brightness Temperature (res06,10.7GHz,H)'].size
    print 'The size of azimuth are:\n', hf0['Earth Azimuth'].size


def check_station(site_no, d0):
    date = [85, 85.875, 95, 95.875, 115.875, 277.875, 279.875, 947.000]
    date = [a for a in range(95, 105)]
    date = [98 + a * 0.125 for a in range(0, 24)]
    date = [d0 + a for a in range(0, 10)]
    for d0 in date[0: -1]:
        opt = read_site.search_snotel(site_no, d0)


def get_h5_atts(h5_name, uplayer):
    h0 = h5py.File(h5_name)
    atts = h0[uplayer].keys()
    h0.close()
    return atts


def smap_ft_save(data_array, groups, h5_name='test_smap_ft_save.h5'):
    h0 = h5py.File(h5_name, 'a')
    grp0 = h0.create_group(groups)
    for ft_value in data_array:
        if any(ft_value[:, 0] > -1):
            pixel_id = ft_value[
                ft_value[:, 0] > -1, -1]  # get the pixel id based on the distance from station to pixel center
            break
    for i, id in enumerate(pixel_id):
        gtp0 = data_array
        gtp0[str(id)] = data_array[:, i, :]
    return 0


def smap_ft_extract(ft10):
    ft10_dict_value = {}
    ft10_dict_time = {}
    ft10_dict_fr_ref, ft10_dict_th_ref, ft10_dict_npr = {}, {}, {}
    # extract value from npy array
    for ft_value in ft10:
        if any(ft_value[:, 0] > -1):
            pixel_id = ft_value[
                ft_value[:, 0] > -1, -1]  # get the pixel id based on the distance from station to pixel center
            break
    for i, id in enumerate(pixel_id):
        ft10_dict_value[str(id)] = ft10[:, i, 0]
        ft10_dict_time[str(id)] = ft10[:, i, 1]
        # if site_no in ['947', '968']:
        ft10_dict_fr_ref[str(id)] = ft10[22, i, 2]
        ft10_dict_th_ref[str(id)] = ft10[22, i, 8]
        ft10_dict_npr[str(id)] = ft10[:, i, 6]
    return ft10_dict_value, ft10_dict_time, ft10_dict_fr_ref, ft10_dict_th_ref, ft10_dict_npr


def ascat_corner_rotate(pc, theta0):
    """
    calculate the corner and rotate
    :param pc:
    theta: in rads
    :return:
    """
    xc, yc = pc[0], pc[1]
    # x0 = np.sqrt(xc**2-)
    # bxy.cal_dis(xc, yc, a, b)
    dis_ref = bxy.cal_dis(yc, xc, yc, xc + 0.01)
    step_lon = 6.0 / dis_ref
    x0 = xc - step_lon * 0.01
    x1 = xc + step_lon * 0.01
    dis_ref = bxy.cal_dis(yc, xc, yc + 0.01, xc)
    step_lat = 6.0 / dis_ref
    y0 = yc + step_lat * 0.01
    y1 = yc - step_lat * 0.01
    locs = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) - np.array([xc, yc])
    theta = theta0 / 180 * np.pi
    rot_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    new_loc = np.matmul(locs, rot_matrix)

    xy = np.array([[1, 1], [1, 2], [2, 2], [2, 1]]) - np.array([1.5, 1.5])
    xy_r1 = np.matmul(xy, rot_matrix)
    xy_new = xy_r1 + np.array([1.5, 1.5])

    theta3 = 37.0 / 180 * np.pi
    rot_matrix2 = np.array([[np.cos(theta3), np.sin(theta3)], [-np.sin(theta3), np.cos(theta3)]])
    xy_new2 = np.matmul(xy_r1, rot_matrix2) + np.array([1.5, 1.5])
    fig0 = pylt.figure()
    ax = fig0.add_subplot(1, 1, 1)
    ax.plot(xy_new[:, 0], xy_new[:, 1], 'r-')
    ax.plot(xy_new2[:, 0], xy_new2[:, 1], 'b--')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    pylt.savefig('test_ascat_corner_roate')
    return new_loc + np.array([xc, yc])


def ascat_corner_rotate_new(pc, theta0):
    half_diag = 12.5 / 2.0 * np.sqrt(2.0)
    if theta0 < 45:
        theta2 = 45 - theta0
    xc, yc = pc[0], pc[1]
    # x0 = np.sqrt(xc**2-)
    # bxy.cal_dis(xc, yc, a, b)
    dis_lon = bxy.cal_dis(yc, xc, yc, xc + 0.01)
    # step_lon = 6.0/dis_ref
    dis_lat = bxy.cal_dis(yc, xc, yc + 0.01, xc)
    # step_lat = 6.0/dis_ref

    delta_x1 = half_diag * np.sin(theta2 / 180.0 * np.pi) * 1e-2 / dis_lon
    delta_x2 = half_diag * np.cos(theta2 / 180.0 * np.pi) * 1e-2 / dis_lon
    delta_y1 = half_diag * np.cos(theta2 / 180.0 * np.pi) * 1e-2 / dis_lat
    delta_y2 = half_diag * np.sin(theta2 / 180.0 * np.pi) * 1e-2 / dis_lat
    coe_matrix = np.zeros([8, 4])
    coe_matrix[[0, 1, 5, 6], [0, 1, 3, 2]] = -1
    coe_matrix[[2, 3, 4, 7], [0, 1, 2, 3]] = 1
    deltas = np.array([delta_x1, delta_x2, delta_y1, delta_y2]).reshape(4, -1)
    rotates = np.matmul(coe_matrix, deltas).reshape(4, -1)
    return np.array(pc) + rotates


def angular_correct(sigma0, inc0, sec, inc_c=45, inc_01=[20, 60], coef=False):
    """
    the time in unit of sec are used to extract measurement during winter, while the back-scatter is affected mainly by
    angles. The extracted data are used in linear regression
    regression
    :param sigma0:
    :param inc0:
    :param sec:
    :param inc_c:
    :param inc_01:
    :return:
    """
    value = sigma0.ravel()
    angle = inc0.ravel()
    t = sec.ravel()
    # print 'angular correct, times on x axis are', t
    t_tuple = bxy.time_getlocaltime_v2(t, ref_time=[2000, 1, 1, 0], t_source='US/Alaska')
    i_valid0 = (value > -180) & (value != 0) & \
               (angle > inc_01[0]) & (angle < inc_01[1])
    i_valid1 = (t_tuple[-2] > 20) & (t_tuple[-2] < 70)  # winter time, used for regression
    i_valid = i_valid0 & i_valid1
    if sum(i_valid) < 10:
        a = -0.12001
        b = -9.99
    else:
        a, b = np.polyfit(angle[i_valid], value[i_valid], 1)
        # print 'the (%d, %.2f dB) linear regression coefficients, a: %.3f, b: %.3f' \
        #       % (value[i_valid].size, np.std(value[i_valid]), a, b)
    value[i_valid0] = value[i_valid0] - (angle[i_valid0] - inc_c) * a
    # print 'after correction, the std is %.2f dB' % (np.std(value[i_valid0]))
    if coef:
        return value, a, b
    else:
        return value


def angular_correct_v2(sigma0, inc0, inc_c=45):
    return 0


# updated 0515/2018
def ascat_plot_series_copy(site_no, orb_no=0, inc_plot=False, sigma_g=5, pp=False, norm=False,
                           txt_path='./result_05_01/ascat_point/', is_sub=False,
                           order=1):  # orbit default is 0, ascending, night pass
    # read series sigma
    site = site_no
    site_nos = site_infos.get_id(site, mode='single')
    for si0 in site_nos:
        if is_sub is False:
            txtname = glob.glob(txt_path + 'ascat_s' + si0 + '*')[0]
        # txtname = './result_05_01/ascat_point/ascat_s'+si0+'_2016.npy'
        ascat_all = np.load(txtname)
        if ascat_all.size < 1:
            return [-1, -1, -1], [-1, -1, -1], [-1, -1], -1, [-1, -1], [-1, -1]
        id_orb = ascat_all[:, -1] == orb_no
        ascat_ob = ascat_all[id_orb]
        # transform utc time to local time
        sec_ascat = bxy.timetransform(ascat_ob[:, 1], '20000101 00:00:00', '%Y%m%d %X', tzone=True)
        doy_tp = np.modf((sec_ascat) / 3600.0 / 24.0)
        doy = doy_tp[1] + 1
        passhr = np.round(doy_tp[0] * 24.0)
        times_ascat = bxy.time_getlocaltime(ascat_ob[:, 1], ref_time=[2000, 1, 1, 0])
        doy_tp2 = times_ascat[-2]
        passhr2 = times_ascat[-1]

        # angular normalization
        tx = doy_tp[1] + 1 - 365
        p = (tx > 0) & (tx < 60)
        x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
        a, b = np.polyfit(x, y, 1)
        print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)
        f = np.poly1d([a, b])

        # angular dependency for each ob mode
        p_win, p_su = (tx > 0) & (tx < 60), (tx > 150) & (tx < 260)
        x_trip_win, x_trip_su = ascat_ob[p_win, 5: 8], ascat_ob[p_su, 5: 8]
        y_trip_win, y_trip_su = ascat_ob[p_win, 2: 5], ascat_ob[p_su, 2: 5]
        a_array, b_array, d_array = np.zeros(6) - 1, np.zeros(6) - 1, np.zeros(3) - 99
        for i1 in range(0, 3):
            a_w, b_w = np.polyfit(x_trip_win[:, i1], y_trip_win[:, i1], 1)
            a_s, b_s = np.polyfit(x_trip_su[:, i1], y_trip_su[:, i1], 1)
            a_array[i1], b_array[i1] = a_w, b_w
            a_array[i1 + 3], b_array[i1 + 3] = a_s, b_s
            d_array[i1] = np.mean(y_trip_su[:, i1]) - np.mean(y_trip_win[:, i1])
        a_coef_name = 'ascat_linear_a.txt'
        b_coef_name = 'ascat_linear_b.txt'
        with open(a_coef_name, 'a') as t_file:
            t_file.write(
                '%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, a_array[0], a_array[1], a_array[2],
                                                                       a_array[3], a_array[4], a_array[5], d_array[0],
                                                                       d_array[1], d_array[2]))
        with open(b_coef_name, 'a') as t_file:
            t_file.write(
                '%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, b_array[0], b_array[1], b_array[2],
                                                                       b_array[3], b_array[4], b_array[5], d_array[0],
                                                                       d_array[1], d_array[2]))

        sig_m = ascat_ob[:, 3]
        inc_m = ascat_ob[:, 6]
        # sig_m = ascat_ob[:, 2]  # changed using the forwards mode to increase the winter summer difference
        # inc_m = ascat_ob[:, 5]
        sig_mn = sig_m - (ascat_ob[:, 6] - 45) * a
        # daily average:
        tdoy = tx
        u_doy = np.unique(tdoy)
        sig_d, i0 = np.zeros([u_doy.size, 2]), 0
        pass_hr_d = np.zeros(u_doy.size)
        inc_d = np.zeros(u_doy.size)
        for td in u_doy:
            sig_d[i0][0] = td
            sig_d[i0][1] = np.mean(sig_mn[tdoy == td])
            inc_d[i0] = np.mean(inc_m[tdoy == td])  # daily incidence angle
            pass_hr_d[i0] = np.mean(passhr[tdoy == td])
            i0 += 1
        tx = sig_d[:, 0]
        sig_mn = sig_d[:, 1]
        pass_hr_d = np.round(pass_hr_d)
        # one more constraints, based on incidence angle
        # id_inc = bxy.gt_le(inc_d, 30, 35)
        # tx, sig_mn = tx[id_inc], sig_mn[id_inc]

        # edge detect
        sig_g = sigma_g  # gaussian stds
        g_size = 6 * sig_g / 2
        if order == 1:
            g_sig, ig2 = gauss_conv(sig_mn, sig=sig_g, size=2 * g_size + 1)  # non-smoothed
        elif order == 2:
            g_sig, ig2 = gauss2nd_conv(sig_mn, sig=sig_g)
        g_sig_valid = 2 * (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size])) \
                      / (np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size])) - 1
        g_sig_valid_non = g_sig[g_size: -g_size]  # NON-normalized
        if norm == True:
            g_sig_valid = g_sig[g_size: -g_size]  # NON-normalized
        max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid_non, 1e-1, tx[g_size: -g_size])
        onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual')

        # new updated 20161130
        # onset, g_npr, i_gaussian, g_sig_valid_non, max_gsig_s, min_gsig_s, sig \
        #     = gauss_cov_snr(sig_mn, 1e-1, tx+365)
        # tp = [x for x in onset]
        # onset = tp
        # g_size = 6*sig/2
        # print 'site no is %s' % si0

        print 'station ID is %s' % si0
        if inc_plot is True:
            # tx = doy_tp[1]+1 -365
            # p = (tx > 20) & (tx < 90)
            # x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
            plot_funcs.inc_plot_ascat(ascat_ob, site_no)
            # ons_site = sm_onset(sm5_date-365, sm5_daily, t5_daily)
        # onset based on ascat

        # actual pass time
        # sec_ascat = bxy.timetransform(ascat_ob[:, 1], '20000101 00:00:00', '%Y%m%d %X', tzone=True)
        # doy, passhr = np.modf((sec_ascat)/3600.0/24.0)[1] + 1, np.round(np.modf((sec_ascat)/3600.0/24.0)[0]*24)
        return [tx[ig2][g_size: -g_size] + 365, g_sig_valid, g_sig_valid_non], \
               [tx, sig_mn, inc_d], \
               onset, \
               pass_hr_d, \
               [u_doy + 365, pass_hr_d], \
               [max_gsig_s, min_gsig_s]  # g_sig[g_size: -g_size] g_sig_valid
    # read site data
    # plot
    return 0


def smap_ascat_position():
    ascat_grid_lat, ascat_grid_lon = np.load('lat_ease_grid.npy'), np.load('lon_ease_grid.npy')
    smap_h5 = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20160105.h5'
    h0 = h5py.File(smap_h5)
    smap_grid_lat, smap_grid_lon = h0[u'cell_lat'].value, h0[u'cell_lon'].value
    ascat_lat, ascat_lon, smap_lat, smap_lon = \
        ascat_grid_lat.ravel(), ascat_grid_lon.ravel(), smap_grid_lat.ravel(), smap_grid_lon.ravel()
    ascat_table_row, ascat_table_col = np.zeros([smap_lat.size, 9]) - 99, np.zeros([smap_lat.size, 9]) - 99
    for id0 in range(0, smap_lat.size):
        # print id0
        smap_ascat_table0 = np.zeros([2, 9]) - 99
        smap0 = [smap_lat[id0], smap_lon[id0]]
        dis = bxy.cal_dis(smap_lat[id0], smap_lon[id0], ascat_lat, ascat_lon)
        sub9 = np.argsort(dis)[0: 9]
        for i0, subi in enumerate(sub9):
            tp_rc = bxy.trans_in2d(subi, [300, 300])
            smap_ascat_table0[0, i0] = tp_rc[0]
            smap_ascat_table0[1, i0] = tp_rc[1]
        ascat_table_row[id0], ascat_table_col[id0] = smap_ascat_table0[0], smap_ascat_table0[1]
    np.savetxt('ascat_row_table.txt', ascat_table_row, fmt='%d', delimiter=',')
    np.savetxt('ascat_col_table.txt', ascat_table_col, fmt='%d', delimiter=',')


def latlon2rc(target):
    lon_t, lat_t = target[0], target[1]
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20160108'
    h0 = h5py.File(h5_name)
    lons_grid = h0['cell_lon'].value
    lats_grid = h0['cell_lat'].value
    dis_array = bxy.cal_dis(lat_t, lon_t, lats_grid.ravel(), lons_grid.ravel())
    idx_1d = np.argmin(dis_array)
    rc = bxy.trans_in2d(idx_1d, lons_grid.shape)
    # print rc
    # print lats_grid[rc[0], rc[1]], lons_grid[rc[0], rc[1]]
    return rc, idx_1d


def ascat_alaska_series():
    return 0


def turning_point(t0, series0, delta0):
    max_tab, min_tab = peakdetect.peakdet(series0, delta0, t0)
    # get the minimum value
    th_index = (min_tab[:, 1] > 60) & (min_tab[:, 1] < 150)
    min_date = min_tab[th_index][np.argmin(min_tab[:, 2][th_index])]
    return min_date


def get_onset(x_time, y_var, k=3, thaw_window=[], freeze_window=[], mode=1, type='npr', window2=False, year0=2016):
    """
    return the thawing onset, indicated by a rlocal maximum
    :param x_time:
    :param y_var:
    :param k:
    :param thaw_window:
    :param freeze_window:
    :return:
    """
    sec0 = bxy.get_total_sec('%d0101' % year0, reftime=[2000, 1, 1, 12])
    sec1 = bxy.get_total_sec('%d0301' % year0, reftime=[2000, 1, 1, 12])
    if len(thaw_window) < 1:
        ini_secs = bxy.get_total_sec('%d0101' % year0, reftime=[2000, 1, 1, 12])
        thaw_window = [ini_secs + doy0 * 3600 * 24 for doy0 in [60, 150]]
        freeze_window = [ini_secs + doy0 * 3600 * 24 for doy0 in [240, 340]]
    # obtain edges, organize it as functions
    if mode == 1:
        max_value, min_value, conv = test_def.edge_detect(x_time, y_var, k, seriestype=type)  # get edges (L. maxima)
    elif mode == 2:
        mean0 = np.abs(np.nanmean(y_var))
        peaks_iter = np.round(np.log10(mean0)) * 0.1
        max_value, min_value = peakdetect.peakdet(y_var, peaks_iter, x_time)
        max_value, min_value, conv = test_def.edge_detect(x_time, y_var, k, seriestype=type)
    # print 'the edge detection detect possible edges: max and min ', max_value.shape, min_value.shape
    if type == 'npr' or type == 'sigma' or type == 'sig':
        max_value_thaw = max_value[(max_value[:, 1] > thaw_window[0]) & (max_value[:, 1] < thaw_window[1])]
    elif type == 'tb':
        max_value_thaw = min_value[(min_value[:, 1] > thaw_window[0]) & (min_value[:, 1] < thaw_window[1])]
        max_value_thaw[:, -1] *= -1
    # min_value_freeze = min_value[(min_value[:, 1] > freeze_window[0]) & (min_value[:, 1] < freeze_window[1])]
    # check positions where onsets doesn't exist.
    if max_value_thaw.size == 0:
        # with open('onset_map0.txt', 'a-') as writer0:
        #     writer0.writelines('no thaw onset was find at: %d' % i0)
        thaw_onset0 = sec0
        thaw_onset_correct = sec1
        print 'warning: no thaw onset was found'
    else:
        thaw_onset0 = max_value_thaw[:, 1][max_value_thaw[:, -1].argmax()]  # thaw onset from npr
        local_max_value = max_value_thaw[:, -1]
        thaw_percent = max_value_thaw[:, -1]/max_value_thaw[:, -1].max()
        correct_onsets = max_value_thaw[:, 1][(thaw_percent > 0.5) & (max_value_thaw[:, -1] > 0.01)]
        if correct_onsets.size < 2:
            thaw_onset_correct = max_value_thaw[:, 1][thaw_percent > 0.9][0]
        else:
            thaw_onset_correct = correct_onsets[0]
        if max_value_thaw[:, -1][max_value_thaw[:, 1] == thaw_onset_correct] < 5e-3:
            thaw_onset_correct = sec1
    return conv, thaw_onset_correct, max_value, min_value


def get_onset_new(x_time, y_var, k=3, thaw_window=[], freeze_window=[],
                  mode=2, type='npr', melt_window=False, year0=2016):
    """
    return the thawing onset, indicated by a rising edge
    :param x_time:
    :param y_var:
    :param k:
    :param thaw_window:
    :param freeze_window:
    :return: conv_2, thaw_onset0, np.array([melt_onset0, s_level, conv_edge_onset, winter_noise_std, winter_noise])
    """
    s_level = -1
    ini_secs = bxy.get_total_sec('%d0101' % year0, reftime=[2000, 1, 1, 12])
    winter_window = [ini_secs + doy0 * 3600 * 24 for doy0 in [2, 60]]
    level, conv_edge_onset, conv_winter_mean, winter_noise, winter_noise_std = -1, -1, -1, -1, -1
    if len(thaw_window) < 1:
        ini_secs = bxy.get_total_sec('%d0101' % year0, reftime=[2000, 1, 1, 12])
        thaw_window = [ini_secs + doy0 * 3600 * 24 for doy0 in [60, 150]]
        freeze_window = [ini_secs + doy0 * 3600 * 24 for doy0 in [240, 340]]

    # obtain edges
    if mode == 1:
        max_value, min_value, conv = test_def.edge_detect(x_time, y_var, k, seriestype=type)
    elif mode == 2:
        mean0 = np.abs(np.nanmean(y_var))
        peaks_iter = np.round(np.log10(mean0)) * 0.1
        max_value, min_value, conv = test_def.edge_detect(x_time, y_var, k, seriestype=type)
        max_value_2, min_value, conv_2 = test_def.edge_detect(x_time, y_var, k, seriestype=type, w=1)
    print 'the edge detection detect possible edges: max and min ', max_value.shape, min_value.shape
    max_value_thaw = max_value[(max_value[:, 1] > thaw_window[0]) & (max_value[:, 1] < thaw_window[1])]

    # check positions where onsets doesn't exist.
    if max_value_thaw.size == 0:
        # with open('onset_map0.txt', 'a-') as writer0:
        #     writer0.writelines('no thaw onset was find at: %d' % i0)
        thaw_onset0 = ini_secs
        print 'warning: no thaw onset was found'
    else:
        thaw_onset0 = max_value_thaw[:, 1][max_value_thaw[:, -1].argmax()]  # thaw onset from npr
    # set the melt zone end
    melt_end = thaw_onset0
    melt_end = thaw_window[1]
    # detect the melt using down edge
    if melt_window is not False:
        # min_value edge value should < 0
        min_value = min_value[min_value[:, -1] < 0]
        if melt_window < melt_end:
            melt_zone = np.array([melt_window, melt_end])  # melt window: edges from npr_up to sig_up
            min_value_melt = min_value[(min_value[:, 1] > melt_zone[0]) & (min_value[:, 1] < melt_zone[1])]
            if min_value_melt.size == 0:
                melt_onset0 = ini_secs
                print 'warning: no thaw onset was found'
            else:
                conv = conv_2
                winter_index = (conv[0] > winter_window[0]) & (conv[0] < winter_window[1])
                # winter noise
                conv_winter_mean = np.nanmean(conv[1][winter_index])
                winter_noise = np.nanstd(conv[1][winter_index])
                # calculate the noisy edge
                noise0 = min_value[:, -1][(min_value[:, 1] > winter_window[0]) & (min_value[:, 1] < winter_window[1])]
                winter_noise = np.mean(noise0[noise0 < 0])
                winter_noise = np.min(noise0[noise0 < 0])
                winter_noise_std = np.std(noise0[noise0 < 0])
                if winter_noise_std == 0:
                    winter_noise_std = 1e-2
                # calculate the signal edge
                #  1: use the min value; 2:  use the first valid one
                level_array = (min_value_melt[:, -1] - conv_winter_mean) / winter_noise
                level_array = min_value_melt[:, -1] / winter_noise
                level_array = (np.abs(min_value_melt[:, -1]) - np.abs(winter_noise)) / winter_noise_std
                level_array = min_value_melt[:, -1] < (winter_noise - 3 * winter_noise_std)
                # first_ind = np.where(np.abs(level_array) > 0)  # the level threshold
                first_ind = np.where(level_array)
                if first_ind[0].size < 1:
                    print 'no significant edge in this case'
                    min_ind = min_value_melt[:, -1].argmin()
                else:
                    min_ind = min_value_melt[:, -1].argmin()
                    min_ind = first_ind[0][0]
                conv_edge_onset = min_value_melt[:, -1][min_ind]
                melt_onset0 = min_value_melt[:, 1][min_ind]
                # level = (conv_edge_onset - conv_winter_mean)/winter_noise
                level = conv_edge_onset / winter_noise
                level = (np.abs(conv_edge_onset) - np.abs(winter_noise)) / winter_noise_std
                level = level_array[min_ind]
                if level:
                    s_level = 1
                else:
                    s_level = 0
                    # level = np.max(noise0[noise0<0])
        else:
            melt_onset0 = ini_secs
    # , conv_winter_mean
    return conv_2, thaw_onset0, np.array([melt_onset0, s_level, conv_edge_onset, winter_noise_std, winter_noise])


def get_onset_zero_x(conv, onset, zero_x=0.5):
    if zero_x > 0:
        zero_cross = conv[0][(conv[0] > onset) & (conv[1] < zero_x)]
    else:
        zero_cross = conv[0][(conv[0] > onset) & (conv[1] > zero_x)]
    if zero_cross.size > 1:
        return zero_cross[0]
    else:
        p = 0
        # print 'no positive zero cross was found, time series may be too short'
        # print conv
        return conv[0][0]


def cal_ascat_ad(i_asc, i_des, dict_pixel, ascat_atts, i1):
    i_mean_valid = ~np.isnan(dict_pixel[ascat_atts[0] + '_9'])
    asc_tp, des_tp = bxy.time_getlocaltime(dict_pixel[ascat_atts[2] + '_9'][i_asc], ref_time=[2000, 1, 1, 0]), \
                     bxy.time_getlocaltime(dict_pixel[ascat_atts[2] + '_9'][i_des], ref_time=[2000, 1, 1, 0])
    ia_2016, id_2016 = asc_tp[0] == 2016, des_tp[0] == 2016

    all_tp = bxy.time_getlocaltime(dict_pixel[ascat_atts[2] + '_9'][i_mean_valid], ref_time=[2000, 1, 1, 0])
    # np.save('test_pixel_mean', np.array([ascat_dict['sate_type'][i_mean_valid], all_tp]))
    # np.save('test_asc', np.array([asc_tp[-2][ia_2016], asc_tp[-1][ia_2016]]))
    # np.save('test_des', np.array([des_tp[-2][id_2016], des_tp[-1][id_2016]]))
    # print 'file is saved'
    same_doy, i_a, i_d = np.intersect1d(asc_tp[-2][ia_2016], des_tp[-2][id_2016], return_indices=True)
    sane_d2, i_a2, i_d2 = np.intersect1d(np.flip(asc_tp[-2][ia_2016]),
                                         np.flip((des_tp[-2][id_2016])), return_indices=True)
    i_a0, i_d0 = -i_a2 + asc_tp[-2][ia_2016].size, -i_d2 + des_tp[-2][id_2016].size

    # calculate A/D backscatter, the index: [i_asc][ia_2016][i_a]
    n = 2 + (len(ascat_atts) - 1) * 20
    dict_a, dict_d = {}, {}
    for main_att in ascat_atts:
        for id in range(0, 10):
            daily_sigma = np.zeros([n, same_doy.size]) - 999.  # secs (ascending), asc_sig, des_sig
    # first occurrence in ascending: ia, lat occurrence in des: id0-1

    sec_asc = dict_pixel[ascat_atts[2] + '_9'][i_asc][ia_2016][i_a]  # ascending
    dict_a[ascat_atts[2] + '_9'] = sec_asc
    sec_des = dict_pixel[ascat_atts[2] + '_9'][i_des][id_2016][i_d0 - 1]
    dict_d[ascat_atts[2] + '_9'] = sec_des
    for main_att in ascat_atts:
        for id in range(0, 10):
            key_x = '%s_%d' % (main_att, id)
            dict_a[key_x] = dict_pixel[key_x][i_asc][ia_2016][i_a]  # ascending
            dict_d[key_x] = dict_pixel[key_x][i_des][id_2016][i_d0 - 1]
    np.savez('ascat_asc_%d.npz' % i1, **dict_a)
    np.savez('ascat_des_%d.npz' % i1, **dict_d)
    # sec_asc = dict_pixel[ascat_atts[2]+'_9'][i_asc][ia_2016][indice_a[0]]
    # inc_asc = dict_pixel[ascat_atts[1]+'_9'][i_asc][ia_2016][indice_a[0]]
    # asc_sig = dict_pixel[ascat_atts[0]+'_9'][i_asc][ia_2016][indice_a[0]]
    #
    # sec_des = dict_pixel[ascat_atts[2]+'_9'][i_des][id_2016][indice_d[-1]]
    # des_sig = dict_pixel[ascat_atts[0]+'_9'][i_des][id_2016][indice_d[-1]]
    # inc_des = dict_pixel[ascat_atts[1]+'_9'][i_des][id_2016][indice_d[-1]]
    # daily_sigma[:, i0] = np.array([sec_asc, asc_sig, inc_asc, sec_des, des_sig, inc_des])
    np.save('test_ascat_obd_36_%d' % (i1), daily_sigma)
    return 0


def get_base(h5_name, lon_key, lat_ley):
    h0 = h5py.File(h5_name)
    lon_grid = h0[lon_key].value.copy()
    lat_grid = h0[lat_ley].value.copy()
    h0.close()
    return lon_grid, lat_grid


def smap_alaska_grid(date_str, att_list, orbit, grid_size=9000):
    h5_path = 'result_08_01/area/smap_area_result'
    h5_list = []
    for d0 in date_str:
        match_name = '%s/SMAP_alaska_%s_GRID_%s.h5' % (h5_path, orbit, d0)
        f0 = glob.glob(match_name)  # all single orbit
        h5_list.append(f0)
    h5_list = filter(None, h5_list)
    if len(h5_list) < 1:
        print 'no Gridded h5 data was found'
        return 0
    smap_dict = {}
    for att0 in att_list:
        smap_dict[att0] = np.zeros([grid_size, len(h5_list)]) - 999
    if len(att_list) > 0:
        for i_date, resample0_path in enumerate(h5_list):
            h0 = h5py.File(resample0_path[0])
            for att0 in att_list:
                smap_dict[att0][:, i_date] = h0[att0].value.ravel()
            h0.close()
    return smap_dict


def smap_extract_ad(year, smap_input_a, smap_input_d):
    t_a_secs, t_d_secs = smap_input_a['cell_tb_time_seconds_aft'], \
                         smap_input_d['cell_tb_time_seconds_aft']
    t_tup_a, t_tup_d = bxy.time_getlocaltime(t_a_secs), bxy.time_getlocaltime(t_d_secs)
    i_year_a, i_year_d = t_tup_a[0] == year, t_tup_d[0] == year
    same_doy, i_same_a, i_same_d = np.intersect1d(t_tup_a[-2][i_year_a], t_tup_d[-2][i_year_d],
                                                  assume_unique=True, return_indices=True)
    # the input: 0: v, 1: h, 2: time, didn't use the fore-ward mode
    smap_0_0, smap_1_0 = [], []
    for key0 in ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft']:
        smap_1_0.append(smap_input_a[key0][i_year_a][i_same_a])
        smap_0_0.append(smap_input_d[key0][i_year_d][i_same_d])
    smap_1 = np.array(smap_1_0)  # ascending
    smap_0 = np.array(smap_0_0)  # descending
    smap_masked = []
    for smap0 in [smap_0[0], smap_0[1], smap_1[0], smap_1[1]]:  # reject the -9999
        smap0_new = bxy.nan_set(smap0, -9999.0)
        smap_masked.append(smap0_new)
    diff_tbv, diff_tbh = smap_masked[0] - smap_masked[2], smap_masked[1] - smap_masked[3]
    x_time = smap_0[2]
    return x_time, diff_tbv, diff_tbh


def ascat_alaska_grid(ascat_atts, path_ascat):
    '''
    :param ascat_atts:
    :param path_ascat:
    :return: ascat_dict, with keys of ascat_atts, and satellite type
    '''
    sate_type = ['metopA_A.h5', 'metopB_A.h5', 'metopA_D.h5', 'metopB_D.h5']
    ascat_dict = {}
    start0 = bxy.get_time_now()
    for att0 in ascat_atts:
        ascat_dict[att0] = np.zeros([300, 300, len(path_ascat)])
    ascat_dict['sate_type'] = np.zeros(len(path_ascat)) - 1
    secs_all = np.zeros([2, len(path_ascat)])  # sort the pass secs obtained from file name
    for i2, path0 in enumerate(path_ascat):
        fname_elems = path0.split('/')[-1].split('_')
        if '20160130' in fname_elems:
            pause = 0
        secs_all[0, i2] = bxy.get_total_sec(fname_elems[2], fmt='%Y%m%d') + float(fname_elems[3]) * 3600  # local secs
        secs_all[1, i2] = sate_type.index(fname_elems[1] + '_' + fname_elems[-1])
    time_sort_ind = np.argsort(secs_all[0])
    for i0, f0 in enumerate(np.array(path_ascat)[time_sort_ind]):
        # form 3d-array to save time-sorted ascat sigma, incidence, pass secs
        h0 = h5py.File(f0)
        ascat_dict['sate_type'][i0] = h0['sate_type'].value.copy()
        for att0 in ascat_atts:
            if att0 not in h0.keys():
                ascat_dict[att0][:, :, i0] = -999
            else:
                ascat_dict[att0][:, :, i0] = h0[att0].value.copy()
        h0.close()
    return ascat_dict


def ascat_alaska_grid_v2(ascat_atts, path_ascat, pid=np.array([3770])):
    '''
    get the ascat measurements of a specific pixel, ['inc_angle_trip_mid', 'utc_line_nodes', 'sigma0_trip_mid']
    :param ascat_atts:
    :param path_ascat:
    :return: ascat_dict, with keys of ascat_atts, and satellite type
    '''
    sate_type = ['metopA_A.h5', 'metopB_A.h5', 'metopA_D.h5', 'metopB_D.h5']
    row_table = np.loadtxt('ascat_row_table.txt', delimiter=',')  # row table of 12.5 grid to 36 grid
    col_table = np.loadtxt('ascat_col_table.txt', delimiter=',')
    ascat_pixel = {}
    key_inc, key_sigma = ascat_atts[0], ascat_atts[1]
    # get ASCAT pixel ids which are coincided with SMAP pixel, calculate the distance between them
    num_pixels = pid.size
    row_no, col_no = row_table[pid, :].astype(int).ravel(), col_table[pid, :].astype(int).ravel()
    lons_125, lats_125 = get_base('result_08_01/ascat_resample_all/ascat_metopB_20160101_11_A.h5',
                                  'longitude', 'latitude')
    lon_samp, lat_smap = get_base('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151102.h5',
                                  'cell_lon', 'cell_lat')  # lat/lon grid of alaska
    lon0, lat0 = lon_samp.ravel()[pid], lat_smap.ravel()[pid]
    pixels_lon, pixels_lat = lons_125[row_no, col_no], lats_125[row_no, col_no]
    pixels_lon.shape = num_pixels, 9
    pixels_lat.shape = num_pixels, 9

    ascat_dict = {}
    dis_array = np.zeros(pixels_lon.shape)  # first calculate the distance between ascat and smap pixels
    dis_array = bxy.cal_dis_v2(lat0, lon0, pixels_lat, pixels_lon)
    ascat_dict['distance'] = dis_array

    # generate dictionary that saves the data of interested pixels
    for att0 in ascat_atts:
        n = site_infos.empty_array_sz(att0)
        ascat_dict[att0] = np.zeros([pid.size * n, len(path_ascat)]) - 9999  # pixels * 9, time
    ascat_dict['sate_type'] = np.zeros(len(path_ascat)) - 1
    secs_all = np.zeros([2, len(path_ascat)])
    # the ascat files are re-ordered based on time
    for i2, path0 in enumerate(path_ascat):
        fname_elems = path0.split('/')[-1].split('_')
        # fname_elems: ['ascat', 'metopA', '20151231', '12', 'A.h5']
        secs_all[0, i2] = bxy.get_total_sec(fname_elems[2], fmt='%Y%m%d') + float(fname_elems[3]) * 3600  # local secs
        secs_all[1, i2] = sate_type.index(fname_elems[1] + '_' + fname_elems[-1])
    time_sort_ind = np.argsort(secs_all[0])
    ascat_dict['sate_type'] = secs_all[1][time_sort_ind]
    path_sorted = np.array(path_ascat)[time_sort_ind]
    # read data in a time series
    for i0, f0 in enumerate(np.array(path_ascat)[time_sort_ind]):
        if i0 == 7:
            pause = 0
        h0 = h5py.File(f0)
        # print i0, ':', f0
        # sigma_r = [angular_correct(sigma_i, inc_i, sec_i, inc_c=40)
        # for inc_i, sigma_i, sec_i in zip(inc0, sigma0, secs0)]
        for att0 in ascat_atts:
            if att0 not in h0.keys():
                ascat_dict[att0][:, i0] = -999
            else:
                ascat_dict[att0][:, i0] = h0[att0].value.copy()[row_no, col_no]
        h0.close()
    return ascat_dict


def ascat_alaska_grid_v3(ascat_atts, path_ascat, pid=np.array([3770])):
    '''
    get the ascat measurements of all not-oncean pixels, ['inc_angle_trip_mid', 'utc_line_nodes', 'sigma0_trip_mid']
    :param ascat_atts:
    :param path_ascat:
    :param pid: 2-d ascat fine grid (resolution = 12.5 km) coordinate
    :return: ascat_dict, with keys of ascat_atts, and satellite type: 'sate_type'
    '''
    sate_type = ['metopA_A.h5', 'metopB_A.h5', 'metopA_D.h5', 'metopB_D.h5']
    ascat_pixel = {}
    key_inc, key_sigma = ascat_atts[0], ascat_atts[1]
    # get ASCAT pixel ids which are coincided with SMAP pixel, calculate the distance between them
    num_pixels = len(pid[0])
    ascat_dict = {}
    # generate dictionary that saves the data of interested pixels
    for att0 in ascat_atts:
        n = site_infos.empty_array_sz(att0)
        ascat_dict[att0] = np.zeros([num_pixels, len(path_ascat)]) - 9999  # pixels * 9, time
    ascat_dict['sate_type'] = np.zeros(len(path_ascat)) - 1
    secs_all = np.zeros([2, len(path_ascat)])  #
    # the ascat files are re-ordered based on time
    for i2, path0 in enumerate(path_ascat):
        fname_elems = path0.split('/')[-1].split('_')
        # fname_elems: ['ascat', 'metopA', '20151231', '12', 'A.h5'] # secs_all: sate type
        secs_all[0, i2] = bxy.get_total_sec(fname_elems[2], fmt='%Y%m%d') + float(fname_elems[3]) * 3600  # local secs
        secs_all[1, i2] = sate_type.index(fname_elems[1] + '_' + fname_elems[-1])
    time_sort_ind = np.argsort(secs_all[0])
    ascat_dict['sate_type'] = secs_all[1][time_sort_ind]
    path_sorted = np.array(path_ascat)[time_sort_ind]
    # get the not-ocean pxiels
    # read data in a time series
    for i0, f0 in enumerate(path_sorted):
        h0 = h5py.File(f0)
        # print i0, ':', f0
        # sigma_r = [angular_correct(sigma_i, inc_i, sec_i, inc_c=40)
        # for inc_i, sigma_i, sec_i in zip(inc0, sigma0, secs0)]
        for att0 in ascat_atts:
            if att0 not in h0.keys():
                ascat_dict[att0][:, i0] = -999
            else:
                ascat_dict[att0][:, i0] = h0[att0].value.copy()[pid]
        h0.close()
    # temp check

    # new_file_list = path_sorted[np.where(ascat_dict['utc_line_nodes'][2950]>0)]
    # for i1, f1 in enumerate(new_file_list):
    #     h0 = h5py.File(f1)
    #     # print i0, ':', f0
    #     # sigma_r = [angular_correct(sigma_i, inc_i, sec_i, inc_c=40)
    #     # for inc_i, sigma_i, sec_i in zip(inc0, sigma0, secs0)]
    #     t_current = h0['utc_line_nodes'].value[pid[0][2950], pid[1][2950]]
    #     t_check = h0['utc_line_nodes'].value
    #     i_valid = t_check>0
    #     t_valid = t_check[i_valid]
    #     for att0 in ascat_atts:
    #         if att0 not in h0.keys():
    #             ascat_dict[att0][:, i1] = -999
    #         else:
    #             ascat_dict[att0][:, i1] = h0[att0].value.copy()[pid]
    #     h0.close()
    return ascat_dict


def angular_effect(ascat_dict, key_inc, key_sigma):
    key_sigma_2 = '%s_40' % key_sigma
    ascat_dict[key_sigma_2] = np.zeros(ascat_dict[key_sigma].shape) - 999
    k = ascat_dict[key_inc].shape[0] / 9
    for type0 in [0., 1., 2., 3.]:
        i_type = (ascat_dict['sate_type'] == type0)
        inc0, sigma0, secs0 = ascat_dict[key_inc][:, i_type].reshape(k, 9, -1), \
                              ascat_dict[key_sigma][:, i_type].reshape(k, 9, -1), \
                              ascat_dict['utc_line_nodes'][:, i_type].reshape(k, 9, -1)
        # check how many valid pixels
        inc_valid = inc0[(inc0 > -999) & (inc0 != 0)]
        i_type = (ascat_dict['sate_type'] == type0)
        if inc0.size < 20:
            a0 = -0.12
        else:
            # a0, b0 = np.polyfit(inc0, sigma0, 1)
            sigma_r = [angular_correct(sigma_i, inc_i, sec_i, inc_c=40)
                       for inc_i, sigma_i, sec_i in zip(inc0, sigma0, secs0)]
            sigma_r_nd = np.array([sigma_r]).reshape(ascat_dict[key_sigma_2][:, i_type].shape)
            ascat_dict[key_sigma_2][:, i_type] = sigma_r_nd
    return ascat_dict


def angular_effect_v2(ascat_dict, key_inc, key_sigma):
    # temp check pixel 22
    p22_inc = ascat_dict['inc_angle_trip_mid'][22, :]
    valid = p22_inc > -100
    p22_value = ascat_dict['sigma0_trip_mid'][22, :]
    valid_ind = [np.where((valid) & (ascat_dict['sate_type'] == type0)) for type0 in [0., 1., 2., 3.]]
    a, b = np.polyfit(p22_inc[valid], p22_value[valid], 1)
    key_sigma_2 = '%s_40' % key_sigma
    ascat_dict[key_sigma_2] = np.zeros(ascat_dict[key_sigma].shape) - 999
    k = ascat_dict[key_inc].shape[0] / 9
    # for type0 in [0., 1., 2., 3.]:
    #     i_type = (ascat_dict['sate_type'] == type0)
    for type0 in [-1]:
        i_type = (ascat_dict['sate_type'] > type0)
        inc0, sigma0, secs0 = ascat_dict[key_inc][:, i_type], \
                              ascat_dict[key_sigma][:, i_type], \
                              ascat_dict['utc_line_nodes'][:, i_type]
        # check how many valid pixels
        inc_valid = inc0[(inc0 > -999) & (inc0 != 0) & ~np.isnan(inc0)]
        i_type = (ascat_dict['sate_type'] == type0)
        if inc0.size < 20:
            a0 = -0.12
        else:
            # a0, b0 = np.polyfit(inc0, sigma0, 1)
            sigma_r = [angular_correct(sigma_i, inc_i, sec_i, inc_c=40)
                       for inc_i, sigma_i, sec_i in zip(inc0, sigma0, secs0)]
            sigma_r_nd = np.array([sigma_r]).reshape(ascat_dict[key_sigma_2][:, i_type].shape)
            ascat_dict[key_sigma_2][:, i_type] = sigma_r_nd
    return ascat_dict


def grid_extract(smap_pixel, key='npr'):
    if key == 'npr':
        return (smap_pixel['cell_tb_v_aft'] - smap_pixel['cell_tb_h_aft']) * 1.0 / \
               (smap_pixel['cell_tb_v_aft'] + smap_pixel['cell_tb_h_aft'])
    elif key == 'tb':
        return np.array([smap_pixel['cell_tb_v_aft'], smap_pixel['cell_tb_h_aft']])


def csv2tif(source, target):
    cvs = gdal.Open(source)
    if cvs is None:
        print 'ERROR: Unable to open %s' % source
        return

    geotiff = gdal.GetDriverByName("GTiff")
    if geotiff is None:
        print 'ERROR: GeoTIFF driver not available.'
        return

    options = []
    geotiff.CreateCopy(target, cvs, 0, options)

    # source = 'E:\\test.csv'
    # target = 'E:\\test.tif'
    #
    # csv2tif(source, target)


def csv2tif_projected(source, target, destEPSG, srcEPSG=4326):
    # open CSV source file
    cvs = gdal.Open(source)
    if cvs is None:
        print 'ERROR: Unable to open %s' % source
        return

    # get GeoTIFF driver
    geotiff = gdal.GetDriverByName("GTiff")
    if geotiff is None:
        print 'ERROR: GeoTIFF driver not available.'
        return

    # set source coordinate system of coordinates in CSV file
    src_crs = osr.SpatialReference()
    src_crs.ImportFromEPSG(srcEPSG)

    # set destination projection parameters
    dest_crs = osr.SpatialReference()
    dest_crs.ImportFromEPSG(destEPSG)

    # set coordinate transformation
    tx = osr.CoordinateTransformation(src_crs, dest_crs)

    # get raster dimension related parameters of source dataset
    xo, xs, xr, yo, yr, ys = cvs.GetGeoTransform()
    xsize = cvs.RasterXSize
    ysize = cvs.RasterYSize

    # convert corner coordinates from old to new coordinate system
    (ulx, uly, ulz) = tx.TransformPoint(xo, yo)
    (lrx, lry, lrz) = tx.TransformPoint(xo + xs * xsize + xr * ysize, \
                                        yo + yr * xsize + ys * ysize)

    # create blank in-memory raster file with same dimension as CSV raster
    mem = gdal.GetDriverByName('MEM')
    dest_ds = mem.Create('', xsize, ysize, 1, gdal.GDT_Float32)

    # get new transformation
    dest_geo = (ulx, (lrx - ulx) / xsize, xr, \
                uly, yr, (lry - uly) / ysize)

    # set the geotransformation
    dest_ds.SetGeoTransform(dest_geo)
    dest_ds.SetProjection(dest_crs.ExportToWkt())

    # project the source raster to destination coordinate system
    gdal.ReprojectImage(cvs, dest_ds, \
                        src_crs.ExportToWkt(), dest_crs.ExportToWkt(), \
                        gdal.GRA_Bilinear, 0.0, 0.0)

    # save projected in-memory raster to disk
    geotiff.CreateCopy(target, dest_ds, 0)


def ascii_to_tiff(infile, outfile, refIm):
    """
    Transform an XYZ ascii file without a header to a projected GeoTiff

    :param infile (str): path to infile ascii location
    :param outfile (str): path to final GTiff
    :param refIm (str): path to a reference image made from the lat lon pair centriods

    """

    im = gdal.Open(refIm)
    ima = gdal.Open(refIm).ReadAsArray()
    row = ima.shape[0];
    col = ima.shape[1]

    indata = np.genfromtxt(infile, delimiter=",", skip_header=True, dtype=None)
    lon = indata[:, 0]  # x
    lat = indata[:, 1]  # y
    data = indata[:, 2]

    # create grid
    xmin, xmax, ymin, ymax = [min(lon), max(lon), min(lat), max(lat)]
    xi = np.linspace(xmin, xmax, col)
    yi = np.linspace(ymin, ymax, row)
    xi, yi = np.meshgrid(xi, yi)

    # linear interpolation
    zi = ml.griddata(lon, lat, data, xi, yi, interp='linear')
    final_array = np.asarray(np.rot90(np.transpose(zi)))

    # projection
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(outfile, col, row, 1, gdal.GDT_Float32)
    dst_ds.GetRasterBand(1).WriteArray(final_array)
    prj = im.GetProjection()
    dst_ds.SetProjection(prj)

    gt = im.GetGeoTransform()
    dst_ds.SetGeoTransform(gt)
    dst_ds = None

    final_tif = gdal.Open(outfile).ReadAsArray()

    return final_tif


def vrt_write(x_size, y_size, source_path, source_band=1, x_offset=0, y_offset=0,
              x_source_size=256, y_source_size=256, dest_x_offset=0, dest_y_offset=0, x_dest_size=90, y_dest_size=90,
              layer0='onset'):
    drv = gdal.GetDriverByName("VRT")
    vrt = drv.Create("test2.vrt", x_size, y_size, 0)

    vrt.AddBand(gdal.GDT_Float32)
    band = vrt.GetRasterBand(1)

    # Changed `x_size` and `y_size` to `x_source_size` and `y_source_size` on the "SourceProperties" line, since the
    # `RasterXSize` and `RasterYSize` attributes should correspond to this source file's pixel size.
    simple_source = '<SourceFilename relativeToVRT="1">%s</SourceFilename>' % source_path + \
                    '<SourceBand>%i</SourceBand>' % source_band + \
                    '<SourceProperties RasterXSize="%i" RasterYSize="%i" DataType="Real"/>' % (
                    x_source_size, y_source_size) + \
                    '<SrcRect xOff="%i" yOff="%i" xSize="%i" ySize="%i"/>' % (
                    x_offset, y_offset, x_source_size, y_source_size) + \
                    '<DstRect xOff="%i" yOff="%i" xSize="%i" ySize="%i"/>' % (
                    dest_x_offset, dest_y_offset, x_dest_size, y_dest_size)

    # simple_source = '<OGRVRTDataSource>' + \
    #                     '<OGRVRTLayer name="%s">' % layer0 + \
    #                     "<SrcDataSource>%s</SrcDataSource>" % source_path + \
    #                     "<GeometryType>wkbPoint</GeometryType>" + \
    #                     '<GeometryField encoding="PointFromColumns" x="Easting" y="Northing" z="Elevation"/' + \
    #                     '</OGRVRTLayer>' + \
    #                     '</OGRVRTDataSource>'

    band.SetMetadataItem("SimpleSource", simple_source)
    # Changed from an integer to a string, since only strings are allowed in `SetMetadataItem`.
    band.SetMetadataItem("NoDataValue", '-9999')


def peak_find(var_xy, p=1.0):
    var_y = var_xy[1] * p
    mean0 = np.abs(np.nanmean(var_y))
    peaks_iter = np.round(np.log10(mean0)) * 0.1 + 0.1
    max_value, min_value = peakdetect.peakdet(var_y, peaks_iter, var_xy[0])
    ini_secs = bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12])
    thaw_window = [ini_secs + doy0 * 3600 * 24 for doy0 in [60, 150]]
    max_value_thaw = max_value[(max_value[:, 1] > thaw_window[0]) & (max_value[:, 1] < thaw_window[1])]
    if max_value_thaw.size == 0:
        # with open('onset_map0.txt', 'a-') as writer0:
        #     writer0.writelines('no thaw onset was find at: %d' % i0)
        thaw_onset0 = 150
        print 'warning: no thaw onset was found'
    else:
        thaw_onset0 = max_value_thaw[:, 1][max_value_thaw[:, -1].argmax()]  # thaw onset from npr
    return thaw_onset0


def zero_find(var_xy, w=9, th=0.5, year0=2016):
    ini_date = 1
    k = np.zeros(w) + 1.0 / w
    mv_average = np.convolve(var_xy[1], k, mode='same')
    out_index = np.where(mv_average < th)[0]
    # remove if the zero cross occurs too early
    out_index = out_index[out_index > 15]
    out_x = var_xy[0][out_index]
    if out_x.size < 1:
        return ini_date
    else:
        return out_x[0]


def zero_find_lt(var_xy, w=9, th=0.5, year0=2016):
    ini_date = 1
    k = np.zeros(w) + 1.0 / w
    mv_average = np.convolve(var_xy[1], k, mode='same')
    out_index = np.where(mv_average < th)[0]
    # remove if the zero cross occurs too early
    out_index = out_index[out_index > 15]
    out_x = var_xy[0][out_index]
    if out_x.size < 1:
        return ini_date
    else:
        return out_x[0]


def zero_find_gt(var_xy, w=9, th=0.5, year0=2016):
    ini_date = 1
    k = np.zeros(w) + 1.0 / w
    mv_average = np.convolve(var_xy[1], k, mode='same')
    out_index = np.where(mv_average > th)[0]
    # remove if the zero cross occurs too early (before doy 60)
    tuple0 = bxy.time_getlocaltime(var_xy[0, 5:10], ref_time=[2000, 1, 1, 0], t_source='US/Alaska')
    year_ref = np.mean(tuple0[0]).astype(int)
    out_index = out_index[var_xy[0][out_index] > bxy.get_total_sec('%d0301' % year_ref)]
    out_x = var_xy[0][out_index]
    if out_x.size < 1:
        return ini_date
    else:
        return out_x[0]


def get_insitu(site_no, time_onset, m_name, window=2):
    '''
    :param time:
    :param name:
    :param window:
    :return:
        the variation during the time window, including mean, max, min, standard deviation
    '''
    t_tup = bxy.time_getlocaltime(time_onset)
    doy_onset = t_tup[3]
    pass_hr = t_tup[-1]
    doy_window = np.zeros([2 * window + 1, doy_onset.size])
    pass_hr_mat = np.matmul(np.zeros([1, doy_window.size]).T + 1, pass_hr.reshape(1, pass_hr.size))
    i_onset = 0
    for doy0 in doy_onset:
        doy_window[:, i_onset] = np.arange(366 + doy0 - window, 366 + doy0 + window + 1)
        i_onset += 1
    doy_window_1d = doy_window.ravel()
    if m_name in ['snow']:
        m_value, m_date = read_site.read_measurements(str(site_no), m_name, doy_window_1d, hr=0)
    else:
        m_value, m_date = read_site.read_measurements(str(site_no), m_name, doy_window_1d, hr=pass_hr_mat.ravel())
    m_value[m_value < -90] = np.nan
    m_value_array = m_value.reshape(doy_window.shape[0], doy_window.shape[1])
    return np.array([np.nanmean(m_value_array, axis=0), np.nanmax(m_value_array, axis=0),
                     np.nanmin(m_value_array, axis=0), np.nanstd(m_value_array, axis=0)])


def get_period_insitu(site_no, time_onset, m_name, window=[3, 3]):
    '''
    :param site_no: int form site number
    :param time_onset: t1 t1, t2, t2, t3, t3, in units of seconds
    :param name: the Soil Moisture Percent -2in (pct) like str
    :param window: useless
    :return:
        the variation during the time window, including mean, max, min, standard deviation
    '''
    t_tup = bxy.time_getlocaltime(time_onset)
    y = t_tup[0][0]
    doy_onset = t_tup[3]
    pass_hr = t_tup[-1]
    # doy_window = np.zeros([window.size, doy_onset.size])
    # pass_hr_mat = np.matmul(np.zeros([1, doy_window.size]).T+1, pass_hr.reshape(1, pass_hr.size))
    i_onset = 0
    # period:
    window_t1_sec = sorted([doy_onset[0], doy_onset[1]])
    window_t2_sec = sorted([doy_onset[2], doy_onset[3]])
    window_t1t2_npr_sec = sorted([doy_onset[0], doy_onset[2]])
    window_t1t2_ascat_sec = sorted([doy_onset[1], doy_onset[3]])
    periods = [window_t1_sec, window_t2_sec, window_t1t2_npr_sec, window_t1t2_ascat_sec]
    result_array = np.zeros([len(periods), 4])  # 4 for mean, max, min, and standard deviation
    for period0 in periods:
        # doy_window[:, i_onset] = np.arange(366+doy0-window, 366+doy0+window+1)
        period = np.arange(period0[0] - 1, period0[1] + 2)
        doy_window_1d = period
        if m_name in ['snow']:
            m_value, m_date = read_site.read_measurements(str(site_no), m_name, doy_window_1d, hr=0, year0=y)
        else:
            m_value, m_date = read_site.read_measurements(str(site_no), m_name, doy_window_1d, hr=18, year0=y)
        m_value[m_value < -90] = np.nan
        result_array[i_onset] = np.array([np.nanmean(m_value), np.nanmax(m_value),
                                          np.nanmin(m_value), np.nanstd(m_value)])
        i_onset += 1
    return result_array


def measurements_slice(mea_value, peroid=np.array([0, 10]), year0=2016, unit='sec'):
    """
    :param mea_value: nd.array, row 0: time in secs, 1: measurements such as air temperature
    :param peroid: the interested period, e.g.: from t_air 0 date to npr increasing date
    :param year0:
    :return:
    """
    if unit != 'sec':
        peroid = (peroid - 1) * 24 * 3600 + bxy.get_total_sec('%d0101' % year0)
    value_slice = mea_value[:, (mea_value[0] > peroid[0]) & (mea_value[0] < peroid[1])]
    return value_slice


def get_ascat_dict(doy_array, y=2016,
                   ascat_atts=['inc_angle_trip_mid', 'utc_line_nodes', 'sate_type', 'sigma0_trip_mid']):
    """
    :param doy_array: search gridded ascat files by date
    :param y:
    :param ascat_atts:
    :return: a dict with keys given in parameters
    """
    path_ascat = []
    for doy0 in doy_array:
        time_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=y)
        match_name = 'result_08_01/ascat_resample_all/ascat_*%s*.h5' % time_str0
        path_ascat += glob.glob(match_name)
    lon_samp, lat_smap = get_base('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151102.h5',
                                  'cell_lon', 'cell_lat')  # lat/lon grid of alaska
    lons_125, lats_125 = get_base('result_08_01/ascat_resample_all/ascat_metopB_20160101_11_A.h5',
                                  'longitude', 'latitude')
    lons_1d = lon_samp.ravel()
    lats_1d = lat_smap.ravel()
    # read ascat_data into dictionary. each key0 corresponded to the key0 of the ascat h5 file (300, 300, time)
    start0 = bxy.get_time_now()
    ascat_dict = ascat_alaska_grid(ascat_atts, path_ascat)  # keys contain 'sate_type'
    start1 = bxy.get_time_now()
    print("----read ascat part: %s seconds ---" % (start1 - start0))
    return ascat_dict


def get_ascat_dict_v2(doy_array, y=2016,
                      ascat_atts=['inc_angle_trip_mid', 'utc_line_nodes', 'sigma0_trip_mid'],
                      file_path='ascat_resample_all',
                      p_ids=np.array([3770])):
    """
    read ascat pixels from gridded ascat data
    :param doy_array: search gridded ascat files by date
    :param p_ids: the specific pixel id
    :param ascat_atts:
    :return: a dict with keys given in ascat_atts, sate_type and distance are added in to the dict
    """
    path_ascat = []
    for doy0 in doy_array:
        time_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=y)
        match_name = 'result_08_01/%s/ascat_*%s*.h5' % (file_path, time_str0)
        path_ascat += glob.glob(match_name)
    # lon_samp, lat_smap = get_base('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151102.h5',
    #                                        'cell_lon', 'cell_lat')  # lat/lon grid of alaska
    # read ascat_data into dictionary. each key0 corresponded to the key0 of the ascat h5 file (300, 300, time)
    start0 = bxy.get_time_now()
    ascat_dict = ascat_alaska_grid_v2(ascat_atts, path_ascat, pid=p_ids)  # keys contain 'sate_type'
    start1 = bxy.get_time_now()
    print("----read ascat part: %s seconds ---" % (start1 - start0))
    return ascat_dict


def get_smap_dict(doy_array, y=2016,
                  smap_atts=['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft']):
    """
    :param doy_array:
    :param y:
    :param smap_atts:
    :return: array 9000 * time, gridded smap measurements, ascending and descending orbits
    """
    date_str = []  # period of interest
    for doy0 in doy_array:
        date_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=y)
        date_str.append(date_str0)
    start0 = bxy.get_time_now()
    lon_samp, lat_smap = get_base('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151102.h5',
                                  'cell_lon', 'cell_lat')  # lat/lon grid of alaska
    smap_dict = smap_alaska_grid(date_str, smap_atts, 'A', lon_samp.size)
    smap_dict_d = smap_alaska_grid(date_str, smap_atts, 'D', lon_samp.size)
    start1 = bxy.get_time_now()
    return smap_dict, smap_dict_d


def dict_concatenate(atts, yr0, yr1, yr2):
    dict_all = {}
    for att0 in atts:
        if len(yr0[att0].shape) > 1:
            dict_all[att0] = np.concatenate((yr0[att0], yr1[att0], yr2[att0]), axis=1)
        else:
            dict_all[att0] = np.concatenate((yr0[att0], yr1[att0], yr2[att0]))
    return dict_all


def distance_interpolate(dis, i2, ascat_angle, ascat_value, inc_01=[30, 50]):
    # set weights
    time_length = ascat_angle.shape[1]
    dis_lim = dis < 15
    station_slice = np.arange(i2 * 9, (i2 + 1) * 9)
    dis_15 = dis[dis_lim]
    weight_15 = 1.0 / dis_15 ** 2
    weight_15_array = np.matmul(weight_15.reshape(weight_15.size, -1),
                                np.zeros([1, time_length]) + 1)
    # correct weighs using incidence angle
    angles0 = ascat_angle[station_slice, :]
    angles0 = angles0[dis_lim, :]
    weight_15_array[(angles0 < inc_01[0]) | (angles0 > inc_01[1])] = 0
    # ascat back scatters for pixels within the distance
    values0 = ascat_value[station_slice, :]
    values0 = values0[dis_lim, :]
    # weighted average
    mean0 = np.sum(weight_15_array * values0, axis=0) / np.sum(weight_15_array, axis=0)

    # check too large/small incidence angle
    # check0 = np.sum(weight_15_array, axis=0)
    # check0_indice = np.where((check0 > 0) & (check0 < 0.1))
    # check0_angle = angles0[:, check0_indice]
    return mean0


def distance_interpolate_v2(dis, ascat_angle, ascat_value, utc_value, inc_01=[30, 50], NN=5):
    """

    :param dis:
    :param ascat_angle:
    :param ascat_value:
    :param utc_value:
    :param inc_01:
    :param NN:
    :return: 0, value, 1, inc angle, 2, time
    """
    # set weights
    time_length = ascat_angle.shape[1]
    dis_lim = dis < 15
    # dis_15 = dis[dis_lim]
    dis_15 = dis[:, 0:NN]
    dis_15_array = np.matmul(dis_15.reshape(dis_15.size, -1),
                             np.zeros([1, time_length]) + 1)
    dis_15_array.shape = -1, 5, time_length


    # # check matrix
    # check_arrary = weight_15_array[109, 0, :]
    # check_mat2 = np.abs(np.mean(weight_15_array, axis=2) - weight_15_array[:, :, 7]) < 1e-10
    # correct weighs using incidence angle
    angles0 = ascat_angle.reshape(-1, 9, time_length)[:, 0:5, :]
    values0 = ascat_value.reshape(-1, 9, time_length)[:, 0:5, :]
    utc0 = utc_value.reshape(-1, 9, time_length)[:, 0:5, :]
    dis_15_array[(angles0 < inc_01[0]) | (angles0 > inc_01[1]) | (utc0 < 1e5)] = 999
    out_angles = np.where(np.mean(dis_15_array, axis=1) == 999)
    weight_15_array = 1.0 / dis_15_array ** 2

    # ascat back scatters for pixels within the distance

    # weighted average value, angle, and utc times.
    mean0 = np.sum(weight_15_array * values0, axis=1) / np.sum(weight_15_array, axis=1)
    mean_angle = np.sum(weight_15_array * angles0, axis=1) / np.sum(weight_15_array, axis=1)
    mean_t = np.sum(weight_15_array * utc0, axis=1) / np.sum(weight_15_array, axis=1)
    mean0[out_angles], mean_angle[out_angles], mean_t[out_angles] = -999.0, -999.0, -999.0
    # check too large/small incidence angle
    # check0 = np.sum(weight_15_array, axis=0)
    # check0_indice = np.where((check0 > 0) & (check0 < 0.1))
    # check0_angle = angles0[:, check0_indice]
    return np.array([mean0, mean_angle, mean_t])


def zone_intiation(year0):
    melt_zone0 = np.array([bxy.get_total_sec(str0) for str0 in ['%d0301' % year0, '%d0701' % year0]])  # in unit of secs
    summer_window = np.array([bxy.get_total_sec(str0) for str0 in ['%d0701' % year0, '%d0901' % year0]])
    thaw_window = melt_zone0.copy()
    winter_window = np.array([bxy.get_total_sec(str0) for str0 in ['%d0101' % year0, '%d0301' % year0]])
    return melt_zone0, summer_window, thaw_window, winter_window

def two_series_detect_v2(id_array, series0, series1, year0,
                         gk=[7, 7, 7], pid_smap=np.array([4547, 3770]), angular=True, melt_buff=7,
                         pixel_save=False):
    '''

    :param id_array: the 1d indices of ascat pixel, and the corresponded smap pixel indices.
                     dimension/shape: Pixel ascat X Pixel smap (N X 2)
    :param series0: [npr, t]: dimension/shape of npr: 2: Pixels X time.
    :param series1: [sigma0, inc, t]: dimension/shape of sigma0: 2: Pixels X time
    :param secs_winter:
    :param secs_summer:
    :param thaw_window:
    :param year0:
    :param melt_zone0:
    :param gk:
    :param pid_smap: the 1d id of smap pixel (pixel0 ~ pixel1603)
    :param angular: do correction if true
    :return:
        onset value: [initiation, main event, end] each element is ndarray
        ascat out:
        [0 onset_array_ascat, 1 melt_end_ascat, 2 conv_ascat_array, 3 lvl_array,
         4 mean_winter,       5 mean_summer,    6 npr_on_smap_melt_date_ak, 7 mean_melt_a,
         8 std_winter_a,      9 std_summer_a,   10 std_melt_a,      11 coef_a,      12 coef_b, 13 time_zero_conv
         14 winter_edge,      15 sigma0_on_edge, 16 a list contains stat of winter convolutions
         17 min sigma0 from melt to melt_end,   18 sigma0_5d_after_onset, 19 pixel_kernels], \
        smap pixel: time series of a given pixel/station, [time, value, convolution results], in forms of list
                    due to the elements of this list has different size or shape.
    '''
    # from variables
    npr, t_tb = series0[0], series0[1]
    secs_0601 = bxy.get_total_sec('%d0601' % year0)
    # initial the winodws
    melt_zone0 = np.array([bxy.get_total_sec(str0) for str0 in ['%d0301' % year0, '%d0701' % year0]])  # in unit of secs
    summer_window = np.array([bxy.get_total_sec(str0) for str0 in ['%d0701' % year0, '%d0901' % year0]])
    thaw_window = melt_zone0.copy()
    winter_window = np.array([bxy.get_total_sec(str0) for str0 in ['%d0101' % year0, '%d0301' % year0]])
    # initial output variables for smap
    smap_pixel = []
    onset_array_smap = np.zeros(npr.shape[0]) - 999
    smap_winter, smap_summer, npr_on_smap_melt_date_ak \
        = onset_array_smap.copy(), onset_array_smap.copy(), onset_array_smap.copy()
    smap_peak = onset_array_smap.copy()
    # intial output variables for ascat
    sigma0_all, inc0_all, times0_all = series1[0], series1[1], series1[2]
    conv_ascat_array = np.zeros(sigma0_all.shape[0]) - 999
    melt_events_time_list = np.zeros([sigma0_all.shape[0], 8]) - 999

    # loops # checking
    for l0 in np.arange(0, npr.shape[0]):
        smap_pack, thaw_secs_npr, npr_on_smap_melt_date, mean0 = \
            smap_melt_initiation(npr[l0], t_tb[l0], winter_window, summer_window, year0, gk=gk[0])
        if npr_on_smap_melt_date.size > 0:
            npr_on_smap_melt_date_ak[l0] = npr_on_smap_melt_date
        else:
            npr_on_smap_melt_date_ak[l0] = 0
        smap_pixel.append(smap_pack)
        onset_array_smap[l0] = thaw_secs_npr
        smap_winter[l0] = mean0[0]
        smap_summer[l0] = mean0[1]
        smap_peak[l0] = mean0[2]

    if len(sigma0_all.shape) > 1:
        ascat_pixels_indices = range(0, sigma0_all.shape[0])  # number of pixels and their ids
    else:
        ascat_pixels_indices = [0]
    # ascat_pixels_indices = [2316, 2317]
    # ascat_pixels_indices = range(3000, 3010)
    # list original
    # all_args_list = [(l0, sigma0_all, inc0_all, times0_all,
    #                  coef_a, coef_b, pixel_kernels, mean_winter, mean_summer, mean_melt_a, std_winter_a,
    #                  std_summer_a, std_melt_a, onset_array_ascat, conv_ascat_array, lvl_array, sigma0_pixel,
    #                  melt_events_time_list, melt_events_conv_list, melt_end_ascat, time_zero_conv,
    #                  sigma0_on_edge, sigma0_min_edge, sigma0_5d_after_onset, winter_edge, winter_conv_mean,
    #                  winter_conv_min, winter_conv_std, min_melt_a,
    #                  id_array, n125_n360, melt_zone0.copy(), thaw_window, winter_window, summer_window,
    #                  melt_buff, onset_array_smap, pid_smap, gk, angular) for l0 in ascat_pixels_indices]
    if len(ascat_pixels_indices) < 2:
        p = 0
        convolution_output = two_series_sigma_process(0, sigma0_all, inc0_all, times0_all,
                                 onset_array_smap[0],
                                 melt_zone0, thaw_window, winter_window, summer_window, melt_buff, gk, angular,
                                 save_path='prepare_files/npy_ascat_one_station', is_return=True)
        return [onset_array_smap, smap_winter, smap_summer,
                smap_peak, smap_pixel, npr_on_smap_melt_date_ak], \
               convolution_output
    else:
        all_args_list = [(l0, sigma0_all[l0], inc0_all[l0], times0_all[l0],
                         np.mean(onset_array_smap[pid_smap == id_array[1][l0]]), melt_zone0, thaw_window, winter_window, summer_window,
                         melt_buff, gk, angular) for l0 in ascat_pixels_indices]
    # print 'the corresponded smap onset: ', np.mean(onset_array_smap[pid_smap == id_array[1][l0]])
        print 'the number of variables: ', len(all_args_list)
        pool0 = multiprocessing.Pool()
        pool0.map(two_series_sigma_unwrapp_args, all_args_list)
        return [onset_array_smap, smap_winter, smap_summer, smap_peak, smap_pixel, npr_on_smap_melt_date_ak], 0


def two_series_detect_quick(id_array, series0, series1, year0,
                         gk=[7, 7, 7], pid_smap=np.array([4547, 3770]), angular=True, melt_buff=7,
                         pixel_save=False):
    '''

    :param id_array: the 1d indices of ascat pixel, and the corresponded smap pixel indices.
                     dimension/shape: Pixel ascat X Pixel smap (N X 2)
    :param series0: [npr, t]: dimension/shape of npr: 2: Pixels X time.
    :param series1: [sigma0, inc, t]: dimension/shape of sigma0: 2: Pixels X time
    :param secs_winter:
    :param secs_summer:
    :param thaw_window:
    :param year0:
    :param melt_zone0:
    :param gk:
    :param pid_smap: the 1d id of smap pixel (pixel0 ~ pixel1603)
    :param angular: do correction if true
    :return:
    '''
    # from variables
    npr, t_tb = series0[0], series0[1]
    # initial the winodws
    melt_zone0 = np.array([bxy.get_total_sec(str0) for str0 in ['%d0301' % year0, '%d0701' % year0]])  # in unit of secs
    summer_window = np.array([bxy.get_total_sec(str0) for str0 in ['%d0701' % year0, '%d0901' % year0]])
    thaw_window = melt_zone0.copy()
    winter_window = np.array([bxy.get_total_sec(str0) for str0 in ['%d0101' % year0, '%d0301' % year0]])
    # initial output variables for smap
    smap_pixel = []
    onset_array_smap = np.zeros(npr.shape[0]) - 999
    smap_winter, smap_summer, npr_on_smap_melt_date_ak \
        = onset_array_smap.copy(), onset_array_smap.copy(), onset_array_smap.copy()
    smap_peak = onset_array_smap.copy()
    sigma0_all, inc0_all, times0_all = series1[0], series1[1], series1[2]
    # smap detection, out: onset, smap
    for l0 in np.arange(0, npr.shape[0]):
        smap_pack, thaw_secs_npr, npr_on_smap_melt_date, mean0 = \
            smap_melt_initiation(npr[l0], t_tb[l0], winter_window, summer_window, year0, gk=gk[0])
        if npr_on_smap_melt_date.size > 0:
            npr_on_smap_melt_date_ak[l0] = npr_on_smap_melt_date
        else:
            npr_on_smap_melt_date_ak[l0] = 0
        smap_pixel.append(smap_pack)
        onset_array_smap[l0] = thaw_secs_npr
        smap_winter[l0] = mean0[0]
        smap_summer[l0] = mean0[1]
        smap_peak[l0] = mean0[2]

    if len(sigma0_all.shape) > 1:
        ascat_pixels_indices = range(0, sigma0_all.shape[0])  # number of pixels and their ids
    else:
        ascat_pixels_indices = [0]
    if len(ascat_pixels_indices) < 2:
        p = 0
        two_series_sigma_process(0, sigma0_all, inc0_all, times0_all,
                                 onset_array_smap[0],
                                 melt_zone0, thaw_window, winter_window, summer_window, melt_buff, gk, angular,
                                 save_path='prepare_files/npy_ascat_one_station', is_return=True)
        return [onset_array_smap, smap_winter, smap_summer, smap_peak, smap_pixel, npr_on_smap_melt_date_ak], 0
    # print 'the corresponded smap onset: ', np.mean(onset_array_smap[pid_smap == id_array[1][l0]])


def get_other_variables():
    p = 0


def multi_two_series_detect(args0):
    return two_series_sigma_process(*args0)


def two_series_simga_parepare(onset_array_smap, id_array, sigma0_all, inc0_all, times0_all, n125_n360,
                              gk, melt_buff, pid_smap, melt_zone0):
    '''
    currently useless
    :return:
    '''
    # part 0
    current_id = id_array[1][l0]  # id of the smap pixel that is looped now
    data_count = times0[times0>0].size  # number of valid data
    if data_count > 600:
        g1, g2 = gk[1], gk[2]
        g_max = 30
    elif times0.size < 2:
        print 'no data in this pixel ', n125_n360[0][l0]
    else:
        g1, g2 = gk[1], gk[2]
        g_max = 15
    buff_seconds = melt_buff * 24 * 3600  # a time buff to reset melt/thaw window
    melt_zone0[0] = np.mean(onset_array_smap[pid_smap == current_id]) - buff_seconds
    return


def two_series_sigma_unwrapp_args(args):
    return two_series_sigma_process(*args)


def two_series_sigma_process(l0, sigma0, inc0, times0,
                             smap_onset, melt_zone0, thaw_window, winter_window, summer_window,
                             melt_buff, gk, angular, save_path='npy_file', is_return=False):
    """
    detect the onsets based on ascat series, using the npr thawing onsets as the starts of melt time window
    :param l0:
    :param sigma0:
    :param inc0:
    :param times0:
    :param smap_onset:
    :param melt_zone0:
    :param thaw_window:
    :param winter_window:
    :param summer_window:
    :param melt_buff:
    :param gk:
    :param angular:
    :param save_path:
    :param is_return: default: False, return values if true
    :return:
        convolution series [time, convolution value]
        convolution event [id of local max/min, time, convolution value]
    """
    # find [winter_conv_mean, winter_conv_min, winter_conv_std]
    # part 0
    data_count = times0[times0>0].size  # number of valid data
    # g1, g2 = gk[1], gk[2]
    if data_count > 600:
        g1, g2 = gk[1], gk[2]
        g_max = 30
    elif times0.size < 2:
        print 'no series in pixel %d' % l0
    else:
        g1, g2 = gk[1], gk[2]
        g_max = 15
    # a time buff to reset melt/thaw window
    melt_zone0[0] = smap_onset - melt_buff * 24 * 3600

    # part 1
    ordered_ind = times0.argsort()
    sigma0, inc0, times0 = sigma0[ordered_ind], inc0[ordered_ind], times0[ordered_ind]
    # angular correction, mean value in seasons
    if angular:
        if times0.size < 1:
            sigma0_correct, a, b = sigma0, -0.11, -5
        else:
            sigma0_correct, a, b = angular_correct(sigma0, inc0, times0, inc_c=40, coef=True)
    else:
        sigma0_correct, a, b = sigma0, -0.11, -5
    coef_a, coef_b = a, b

    if data_count < 150:
        max_ascat, min_na, conv_ascat_thaw = edge_detect(times0, sigma0_correct, g1,
                                                                 seriestype='sig', is_sort=True)
        max_na, min_ascat, conv_ascat_melt = edge_detect(times0, sigma0_correct, g2,
                                                                 seriestype='sig', is_sort=True, long_short=True)
        thaw_onset0 = get_positive_edge(max_ascat, thaw_window, smap_onset)
        melt_zone0[1] = thaw_onset0
        pixel_kernels = np.array([-1, -1])
    else:
        # local maximum, positive edge
        edge_count = 10
        while (g1 < g_max) & (edge_count > 4):
            # update g1 each loop
            edge_out, g1, edge_count = \
                edge_iteration_v2(times0, sigma0_correct, g1, 3, [melt_zone0[0], thaw_window[1]], is_negative=0)
            g1 += 1
        max_ascat, min_na, conv_ascat_thaw = edge_out[0], edge_out[1], edge_out[2]
        thaw_onset0 = get_positive_edge(max_ascat, thaw_window, smap_onset)
        melt_zone0[1] = thaw_onset0
        g_short = g1/2
        if g_short > 3:
            g_short = 3
        edge_count = 10
        # local minimum
        while (g_short < 10) & (edge_count > 4) & (g_short < g1):
            # using new g1 in the last step, and update g2 each loop
            edge_out, g1, edge_count = \
                edge_iteration_v2(times0, sigma0_correct, g1, g_short, melt_zone0, is_negative=1)
            g_short += 1
        max_na, min_ascat, conv_ascat_melt = edge_out[0], edge_out[1], edge_out[2]
        pixel_kernels = np.array([g1, g_short])

    # ascat_pixel = np.array([times0, sigma0_correct, conv_ascat_melt, conv_ascat_thaw, max_ascat, min_ascat])
    ascat_pixel = -999
    melt_end_ascat = get_positive_edge(max_ascat, thaw_window, smap_onset)


    # sigma0 during each period, winter, summer, melting
    sigma0_mean_winter, sigma0_std_winter = mean_std_check([times0, sigma0_correct], winter_window, fill0=-99)
    sigma0_mean_summer, sigma0_std_summer = mean_std_check([times0, sigma0_correct], summer_window)
    sigma0_mean_melt, sigma0_std_melt = mean_std_check([times0, sigma0_correct], melt_zone0)
    negative_edge_winter = min_ascat[(min_ascat[:, 1] > winter_window[0]) &
                                     (min_ascat[:, 1] < winter_window[1])]
    local_min_winter = negative_edge_winter[:, -1][negative_edge_winter[:, -1] < 0]
    if local_min_winter.size < 1:
        noise_negative_winter = -99
    else:
        noise_negative_winter = np.nanmean(local_min_winter)
    negative_edge_melt_zone = min_ascat[(min_ascat[:, 1] > melt_zone0[0]) &
                                        (min_ascat[:, 1] < melt_zone0[1])]
    negative_edge_winter = min_ascat[(min_ascat[:, 1] > winter_window[0]) &
                                     (min_ascat[:, 1] < winter_window[1])]

    if negative_edge_winter.size == 0:
        conv_min_winter = -0.5
    else:
        conv_min_winter = np.min(negative_edge_winter)
    melt_onset0, conv_on_melt_date, lvl_on_melt_date, melt_array, number_onset = \
        get_negative_edge(min_ascat, noise_negative_winter, melt_zone0)
    # print 'the local minimum detected in winter: \n ', negative_edge_winter
    # print 'melt secs window: ', melt_zone0
    # print 'the melt zone: \n', bxy.time_getlocaltime(melt_zone0)
    # print 'the noise level in winter of seireis no. %d: ' % l0, noise_negative_winter
    # print 'temporally print the detected melt onset: ', melt_onset0
    # print 'temporally print the local minimum:\n ', min_ascat
    melt_events_time_list = melt_array[0]
    melt_events_conv_list = melt_array[1]

    # value initiation
    sigma0_on_melt_date, sigma0_min_melt_zone, sigma0_5d_after_onset = -999, -999, -999
    if conv_ascat_melt.size > 1:
        # time of zero cross of convolutions
        time_zerox_negative = get_onset_zero_x(conv_ascat_melt, melt_onset0, zero_x=-5e-3)
        # sigma0 on the edge
        if number_onset == -1:
            sigma0_on_melt_date = -9999
        else:
            # sigma0_on_melt_date = np.mean(sigma0_correct[times0 == melt_onset0])
            sigma0_on_melt_date = sigma0_correct[times0 == melt_onset0]
            # np.save('ascat_pixel_npy_%d.npy' % (l0), np.array([times0, sigma0_correct]))
            if sigma0_on_melt_date.size > 1:
                print 'the onset time repeated on ', melt_onset0
                print 'the pixel id, ', l0
                # sys.exit()
        # calculate min sigma0 around melting edge
        sigma0_around_edge = sigma0_correct[(times0 > melt_zone0[0]) & (times0 < melt_zone0[1])]
        # the mean sigma0 of a 5-day-period after the detected onset
        sigma0s_5d = sigma0_correct[(times0 > melt_onset0) & (times0 < melt_onset0+5*24*3600)]
        if sigma0s_5d.size < 1:
            sigma0_5d_after_onset = -999
        else:
            sigma0_5d_after_onset = np.mean(sigma0s_5d)

        if sigma0_around_edge.size > 0:
            sigma0_min_melt_zone = np.min(sigma0_around_edge)
        else:
            sigma0_min_melt_zone = -9999
        # calculate winter convolution
        winter_conv = conv_ascat_melt[1][(conv_ascat_melt[0] > winter_window[0]) &
                                        (conv_ascat_melt[0] < winter_window[1] + 20*3600*24)]
        if winter_conv.size > 0:
             winter_conv_mean, winter_conv_min, winter_conv_std = np.nanmean(winter_conv), np.nanstd(winter_conv),\
                                                                  np.nanmin(winter_conv)
        else:
            winter_conv_mean, winter_conv_min, winter_conv_std = -9999, -9999, -9999
    else:
        time_zerox_negative = -9999
        winter_conv_mean, winter_conv_min, winter_conv_std = -9999, -9999, -9999

    time_zero_conv = time_zerox_negative
    min_melt_a = 0
    # potential return lis
    # 0 l0, 1 coef_a, 2 coef_b, 3 pixel_kernels, 4 sigma0_mean_winter, 5 sigma0_mean_summer, 6 sigma0_mean_melt, 7 sigma0_std_winter, \
    # 8 sigma0_std_summer, 9 sigma0_std_melt, 10 melt_onset0, 11 conv_on_melt_date, 12 lvl_on_melt_date, \
    # 13 melt_end_ascat, 14 time_zero_conv, \
    # 15 sigma0_on_melt_date, 16 sigma0_min_melt_zone, 17 sigma0_5d_after_onset, 18 conv_min_winter, 19 winter_conv_mean, \
    # 20 winter_conv_min, 21 winter_conv_std, 22 min_melt_a, \
    # 23 ascat_pixel, 24 melt_events_time_list, 25 melt_events_conv_list
    # print 'the detected simga0 on snow melt date, ', sigma0_on_melt_date
    # x = 0
    # list_check = {}
    # for i_x0, ix in enumerate([l0, coef_a, coef_b, pixel_kernels, sigma0_mean_winter, sigma0_mean_summer, sigma0_mean_melt,
    #                         sigma0_std_winter, sigma0_std_summer, sigma0_std_melt, melt_onset0, conv_on_melt_date,
    #                         lvl_on_melt_date, melt_end_ascat,
    #                         time_zero_conv, sigma0_on_melt_date, sigma0_min_melt_zone, sigma0_5d_after_onset,
    #                         conv_min_winter, winter_conv_mean, winter_conv_min, winter_conv_std, min_melt_a,
    #                         ascat_pixel, melt_events_time_list, melt_events_conv_list]):
    #     if type(ix) is np.ndarray:
    #         sz0 = ix.size-1
    #         if ix.size == 0:
    #             print i_x0
    #     else:
    #         sz0 = 1
    #     string0 = '%d--%d' % (x, x+sz0)
    #     list_check[string0] = ix
    #     x += sz0
    # print 'when in id is %d: \n' % (l0), list_check
    save_array = np.hstack((l0, coef_a, coef_b, pixel_kernels, sigma0_mean_winter, sigma0_mean_summer,  # 0-6, k.size 2
                            sigma0_mean_melt, sigma0_std_winter, sigma0_std_summer, sigma0_std_melt, melt_onset0,  # ~11
                            conv_on_melt_date, lvl_on_melt_date, melt_end_ascat,
                            time_zero_conv, sigma0_on_melt_date, sigma0_min_melt_zone, sigma0_5d_after_onset,
                            conv_min_winter, winter_conv_mean, winter_conv_min, winter_conv_std, min_melt_a,
                            melt_onset0, melt_events_time_list, melt_events_conv_list))
    # print 'the id: %d has a shape' % l0, save_array.shape
    np.save('%s/file%d.npy' % (save_path, l0), save_array)
    if is_return:
        return [conv_ascat_thaw, conv_ascat_melt], [max_ascat, min_ascat]
    # return l0, coef_a, coef_b, pixel_kernels, sigma0_mean_winter, sigma0_mean_summer, sigma0_mean_melt, \
    #        sigma0_std_winter, sigma0_std_summer, sigma0_std_melt, melt_onset0, conv_on_melt_date, lvl_on_melt_date, \
    #        melt_end_ascat, time_zero_conv, \
    #        sigma0_on_melt_date, sigma0_min_melt_zone, sigma0_5d_after_onset, conv_min_winter, winter_conv_mean, \
    #        winter_conv_min, winter_conv_std, min_melt_a, \
    #        ascat_pixel, melt_events_time_list, melt_events_conv_list


def two_series_sigma():
    return 0


def edge_iteration(times0, sigma0_correct, g, g2, window, is_negative=0):
    '''

    :param times0:
    :param sigma0_correct:
    :param g:
    :param window:
    :param is_negative: 0, searching positive edge, no long-short kernel is used
    :return:
    '''
    edge_out = test_def.edge_detect(times0, sigma0_correct, g, g2,
                                    seriestype='sig', is_sort=True, long_short=is_negative)
    # edge out 0, 1, 2 max_ascat, min_ascat, conv_ascat
    edge_ascat = edge_out[is_negative]
    pos_edge_ind = (edge_ascat[:, 1] > window[0]) & (edge_ascat[:, 1] < window[1])
    pos_edge_count = sum(pos_edge_ind)
    if is_negative:
        if (pos_edge_count > 4) & (g2 < 15):
            medge_out, g = edge_iteration(times0, sigma0_correct, g, g/2+1, window, is_negative=is_negative)
            iter = 0  # do iteration
        else:
            return edge_out, g
    else:
        if (pos_edge_count > 2) & (g < 15):
            medge_out, g = edge_iteration(times0, sigma0_correct, g+1, g2, window, is_negative=is_negative)
            iter = 0  # do iteration
        else:
            return edge_out, g


def edge_iteration_v2(times0, sigma0_correct, g, g2, window, is_negative=0):
    '''
    detect edge on [times0, sigma0_correct]. count the number of local maximum/minimum within the window
    :param times0:
    :param sigma0_correct:
    :param g:
    :param window:
    :param is_negative: 0, searching positive edge, no long-short kernel is used
    :return:
    '''
    edge_out = edge_detect(times0, sigma0_correct, g, g2,
                                    seriestype='sig', is_sort=True, long_short=is_negative)
    edge_ascat = edge_out[is_negative]   # is_negative: 0: local max, 1: local minimum
    edge_count = sum((edge_ascat[:, 1] > window[0]) & (edge_ascat[:, 1] < window[1]))
    return edge_out, g, edge_count


def re_detection(sz0, id_array, i0, t0, kernels=[7, 7, 7], sigma0_type=4):
    '''
    For one single given pixel, applying a new detection.
    :param sz0:
    :param id_array:
    :param i0:
    :param t0:
    :param kernels:
    :param sigma0_type:
    :return:
    '''
    series_npr = [sz0['smap_pixel'][i0][1], sz0['smap_pixel'][i0][0]]

    if sigma0_type >= 4:
        series_ascat = [sz0['ascat_pixel'][i0][1], sz0['ascat_pixel'][i0][0] * 0,
                        sz0['ascat_pixel'][i0][0]]
    else:
        dB_threshold = 0
        print 'set the backscatter threshold: < ', dB_threshold
        type_ind = (sz0['sate_type'] <= sigma0_type) & (sz0['ascat_pixel'][i0][1] < dB_threshold)
        series_ascat = [sz0['ascat_pixel'][i0][1][type_ind], sz0['ascat_pixel'][i0][0][type_ind] * 0,
                        sz0['ascat_pixel'][i0][0][type_ind]]
    bxy.reshape_element0(series_npr), bxy.reshape_element0(series_ascat)
    # onset_s, onset_a, melt_end, array0, lvl, s_pixel, \
    # a_pixel, m_winter, m_summer, melt_sigma, melt_npr, a_r, b_r, smap_seasonal \
    smap_out, ascat_out, pixels = two_series_detect_v2(id_array, series_npr, series_ascat, t0,
                            kernels, pid_smap=id_array[1], angular=False)
    # return [onset_s, onset_a, melt_end], [s_pixel, a_pixel]
    # all output of two_series_detect_v2
    # [onset_array_smap, smap_winter, smap_summer, smap_peak], \
    #        [0onset_array_ascat, 1melt_end_ascat, 2conv_ascat_array, 3lvl_array,
    #         4mean_winter, 5mean_summer, 6smap_melt_signal, 7mean_melt_a, 8std_winter_a, 9std_summer_a, 10std_melt_a,
    #         11coef_a, 12coef_b]
    ini_onset, main_onset, end_onset = smap_out[0], ascat_out[20][0], ascat_out[1]
    sigma0_winter, sigma0_melt = [ascat_out[4], ascat_out[8]], \
                                 [ascat_out[7], ascat_out[2]]
    return [ini_onset, main_onset, end_onset, sigma0_winter, sigma0_melt], pixels


def re_detection_v2(series_npr, series_ascat, id_array, i0, t0, kernels=[7, 7, 7], sigma0_type=4):
    '''
    For one single given pixel, applying a new detection.
    :param series_npr: [time, npr_values]
    :param id_array: [1d indices of ascat, 1d indices of smap]
    :param i0: [id in saved interested pixels]
    :param t0: year
    :param series_ascat: [value, inc, times]
    :param sigma0_type:
    :return:
    '''
    # series_npr = [sz0['smap_pixel'][i0][1], sz0['smap_pixel'][i0][0]]
    # if sigma0_type >= 4:
    #     series_ascat = [sz0['ascat_pixel'][i0][1], sz0['ascat_pixel'][i0][0] * 0,
    #                     sz0['ascat_pixel'][i0][0]]
    # else:
    #     dB_threshold = 0
    #     print 'set the backscatter threshold: < ', dB_threshold
    #     type_ind = (sz0['sate_type'] <= sigma0_type) & (sz0['ascat_pixel'][i0][1] < dB_threshold)
    #     series_ascat = [sz0['ascat_pixel'][i0][1][type_ind], sz0['ascat_pixel'][i0][0][type_ind] * 0,
    #                     sz0['ascat_pixel'][i0][0][type_ind]]
    # call two series detect v2
    bxy.reshape_element0(series_npr), bxy.reshape_element0(series_ascat)
    smap_out, dummy0 = two_series_detect_v2(id_array, series_npr, series_ascat, t0,
                                                        kernels, pid_smap=id_array[1], angular=False)
    # load saved npy file
    ini_onset, main_onset, end_onset = smap_out[0], ascat_out[20][0], ascat_out[1]
    sigma0_winter, sigma0_melt = [ascat_out[4], ascat_out[8]], \
                                 [ascat_out[7], ascat_out[2]]
    return [ini_onset, main_onset, end_onset, sigma0_winter, sigma0_melt], pixels


def mean_std_check(series, window, fill0=-99):
    sub_series = series[1][(series[0] > window[0]) &
                           (series[0] < window[1]) &
                           (series[1] > fill0)]
    if sub_series.size < 1:
        mean0, std0 = fill0, fill0
    else:
        mean0, std0 = np.nanmean(sub_series), np.nanstd(sub_series)
    return mean0, std0


def get_positive_edge(max_value, window, ref_sec):
    '''
    criteria: 1. within the thaw window and gt the smap onset
              2. the local max gt 1 dB
    :param max_value:
    :param window:
    :param ref_sec:
    :return:
    '''
    # the right bound of melt zone
    year0 = bxy.time_getlocaltime(window)[0][1]
    default_onset0 = bxy.get_total_sec('%d0701' % year0)
    all_local_max = max_value[:, 1]
    # criterion 1
    thaw_ascat = max_value[(all_local_max > window[0]) &
                           (all_local_max < window[1]) &
                           (all_local_max > ref_sec)]
    # thaw_ascat: col 0, id in the series, 1: time in secs, 2: local maximum
    if thaw_ascat.size > 0:
        value_local_max, time_local_max = thaw_ascat[:, -1], thaw_ascat[:, 1]
        # thaw_percent = value_local_max/value_local_max.max()
        # criterion 2
        time_valid_max = time_local_max[value_local_max > 1]
        if time_valid_max.size > 0:
            thaw_onset_correct = time_valid_max[-1]
        else:
            thaw_onset_correct = time_local_max[value_local_max.argmax()]
        if value_local_max[time_local_max == thaw_onset_correct] < 0.5:
            thaw_onset_correct = default_onset0
    else:
        thaw_onset_correct = default_onset0
        # print 'no positive edge was detected, used defaulted right bound: %d0701' % year0
    return thaw_onset_correct


def get_negative_edge(min_edge, noise, window):
    # min_conv_winter_mean
    '''
    find the minimum of negative edge (smallest negative value), which is considered as snowmelt
    :param min_edge: 
    :param noise: the mean negative convolution output in winter
    :param window: 
    :return:
    '''
    # initials
    melt_onset0_array = np.zeros(8) - 9999
    melt_conv_array = np.zeros(8) - 9999
    melt_onset0 = window[0]
    melt_conv = -9999
    melt_lvl0 = -9999
    negative_edge_snowmelt = min_edge[(min_edge[:, 1] > window[0]) &
                                      (min_edge[:, 1] < window[1])]
    number_onset = 0
    if negative_edge_snowmelt[:, 2].size < 1:
        pr = 0
        # print 'no negative edge was found within the window'
    else:
        # valid_index_melt = (levels > 1) & (negative_edge_snowmelt[:, 2] < 0)
        valid_index_melt = negative_edge_snowmelt[:, 2] < 0
        valid_melt_edge = negative_edge_snowmelt[valid_index_melt]
        if valid_melt_edge.shape[0] > 0:  # edges based on global minimum within the window
            onset_row0 = valid_melt_edge[valid_melt_edge[:, 2].argmin()]
            melt_onset0 = onset_row0[1]
            melt_conv = onset_row0[2]
            melt_lvl0 = melt_conv/noise
        # save onset/convolution arrays
            melt_onsets = valid_melt_edge[:, 1]
            conv_on_melt = valid_melt_edge[:, 2]
            if melt_onsets.size > 8:
                lowest_8th = melt_conv_array.argsort()
                eight_id = np.sort(lowest_8th[0: 8])
                melt_onset0_array = melt_onsets[eight_id]
                melt_conv_array = conv_on_melt[eight_id]
            else:
                melt_onset0_array[0: valid_melt_edge[:, 1].size] = melt_onsets
                melt_conv_array[0: valid_melt_edge[:, 2].size] = conv_on_melt
    if melt_onset0 == window[0]:
        number_onset = -1
    return melt_onset0, melt_conv, melt_lvl0, \
           [melt_onset0_array, melt_conv_array], \
           number_onset


def re_detection_plot(id_array, series_npr, series_ascat,
                      pixel_type='interest', detectors=[7, 7, 7], i0=0):
    ascat_satellite = 4
    prefix = 'prepare_files/npz'
    # keys: [ini_onset, main_onset, end_onset, sigma0_winter, sigma0_melt], pixels
    # npz file loaded outside the re_detection function
    station_z_2016 = np.load('%s/ascat/ascat_interest_pixel_series_%d.npz' % (prefix, 2016))
    series_ascat = station_z_2016['xxxx']
    station_z_2017 = np.load('%s/ascat/ascat_interest_pixel_series_%d.npz' % (prefix, 2017))
    station_z_2018 = np.load('%s/ascat/ascat_interest_pixel_series_%d.npz' % (prefix, 2018))
    # prepare time series [value, inc, times]
    # re_detection
    print 'do detection in year 2016'
    onset_2016_new, pixel_2016_new = re_detection_v2(station_z_2016['sigma0_trip_fore'][i0],
                                                     np.array([[ascat_id], [smap_id]]), i0, 2016,
                                                     kernels=detectors, sigma0_type=ascat_satellite)
    # call two_series_detect directly
    smap_out, dummy0 = two_series_detect_v2(id_array, series_npr, series_ascat, t0,
                                                        kernels, pid_smap=id_array[1], angular=False)
    print 'do detection in year 2017'
    onset_2017_new, pixel_2017_new = re_detection_v2(station_z_2017,
                                                               np.array([[ascat_id], [smap_id]]), i0, 2017,
                                                               kernels=detectors, sigma0_type=ascat_satellite)
    print 'do detection in year 2018'
    onset_2018_new, pixel_2018_new = re_detection_v2(station_z_2018,
                                                               np.array([[ascat_id], [smap_id]]), i0, 2018,
                                                               kernels=detectors, sigma0_type=ascat_satellite)
    # pixel index guide: [ascat/smap][pixel_no][t/value/convolutions]; onset index: [pixel_no][keys]
    npr_plot = np.array([pixel_2016_new[0][0][0], pixel_2016_new[0][0][1],
                         pixel_2017_new[0][0][0], pixel_2017_new[0][0][1],
                         pixel_2018_new[0][0][0], pixel_2018_new[0][0][1]])
    npr_conv_plot = [pixel_2016_new[0][0][2],
                     pixel_2017_new[0][0][2],
                     pixel_2018_new[0][0][2]]
    npr_conv_bar = np.vstack((pixel_2016_new[0][0][3],
                              pixel_2017_new[0][0][3],
                              pixel_2018_new[0][0][3]))
    t_2016, v_2016 = bxy.remove_unvalid_time(pixel_2016_new[1][0][0], pixel_2016_new[1][0][1])
    t_2017, v_2017 = bxy.remove_unvalid_time(pixel_2017_new[1][0][0], pixel_2017_new[1][0][1])
    t_2018, v_2018 = bxy.remove_unvalid_time(pixel_2018_new[1][0][0], pixel_2018_new[1][0][1])
    sigma_plot = np.array([t_2016, v_2016, t_2017, v_2017, t_2018, v_2018])
    sigma_conv_plot = [pixel_2016_new[1][0][2],
                       pixel_2017_new[1][0][2],
                       pixel_2018_new[1][0][2]]
    simga_conv_plot_more = [pixel_2016_new[1][0][3],
                           pixel_2017_new[1][0][3],
                           pixel_2018_new[1][0][3]]
    sigma_conv_bar_max = np.vstack((pixel_2016_new[1][0][4],
                                    pixel_2017_new[1][0][4],
                                    pixel_2018_new[1][0][4]))
    sigma_conv_bar_min = np.vstack((pixel_2016_new[1][0][5],
                                    pixel_2017_new[1][0][5],
                                    pixel_2018_new[1][0][5]))
    sigma_qa_w = [onset_2016_new[3], onset_2017_new[3], onset_2018_new[3]]
    sigma_qa_m = [onset_2016_new[4], onset_2017_new[4], onset_2018_new[4]]
    text_qa_w, text_qa_m = ['%.2f$\pm$\n%.2f' % (list0[0], list0[1]) for list0 in sigma_qa_w], \
                           ['%.2f$\pm$\n%.2f' % (list0[0], list0[1]) for list0 in sigma_qa_m]
    # calculate data quality indicator: [(mean1-std1)-mean0]/std0
    indicator0 = [[(list_m[0] - list_m[1] - list_w[0])/list_w[1]][0][0]
                  for list_w, list_m in zip(sigma_qa_w,  sigma_qa_m)]
    ind_array = np.zeros(5)
    ind_array[0: 2] = np.array([int(sno0), smap_id])
    ind_array[2:] = indicator0
    ind_mat.append(ind_array)
    # np.savetxt('sigma_variation_indicator.txt', np.array([]))
    # simga_conv_plot.shape = 6, -1
    print onset_2016_new
    positive_edge = np.array([onset_2016_new[0][0], onset_2017_new[0][0], onset_2018_new[0][0]])
    negative_edge = np.array([onset_2016_new[1][0], onset_2017_new[1][0], onset_2018_new[1][0]])
    positive_edge2 = np.array([onset_2016_new[2][0], onset_2017_new[2][0], onset_2018_new[2][0]])
    # plotting
    # value for timing
    doy_p, doy_n, doy_end = bxy.time_getlocaltime(positive_edge)[-2], \
                            bxy.time_getlocaltime(negative_edge, ref_time=[2000, 1, 1, 0])[-2], \
                            bxy.time_getlocaltime(positive_edge2, ref_time=[2000, 1, 1, 0])[-2]
    doy_all = np.concatenate((doy_p, doy_n, doy_end))
    sec_all = np.concatenate((positive_edge, negative_edge, positive_edge2))

    v_line_local = [sec_all, ['k-', 'k-', 'k-', 'r-', 'r-', 'r-', 'b-', 'b-', 'b-'],
                    ['p', 'p', 'p', 'n', 'n', 'n', 'p1', 'p1', 'p1']]
    # read measurement
    if pixel_type == 'interest':
        if site_plot == 'single':
            insitu_plot2 = get_3_year_insitu(int(sno0), m_name='snow')
            insitu_plot = get_3_year_insitu(int(sno0), m_name="air")
            plot_funcs.plot_subplot([npr_plot, sigma_plot, insitu_plot[0:2]],
                                    [[npr_conv_plot[0], npr_conv_plot[1], npr_conv_plot[2]],
                                     [sigma_conv_plot[0], sigma_conv_plot[1], sigma_conv_plot[2]],
                                     insitu_plot2[0:2]],
                                    main_label=['npr', '$\sigma^0$ mid', 'snow'],
                                    figname='ms_pixel_test_%d_%d' % (sno0, smap_id), x_unit='doy', vline=v_line_local,
                                    vline_label=doy_all, h_line=[[-1], [0], [':']], y_lim=[[1], [[-20, -6]]]
                                    )
        # plotting 3*2
        elif site_plot == 'grid':
            subplot_loc = np.unravel_index(site_order, (3, 2))
            ax = plt.subplot2grid((3, 2), subplot_loc)
            ax_2 = ax.twinx()
            ax.plot(npr_plot[0], npr_plot[1]*100, 'k.',  npr_plot[2], npr_plot[3]*100, 'k.',
                    npr_plot[4], npr_plot[5]*100, 'k.', markersize=2)
            ax_2.plot(sigma_plot[0], sigma_plot[1], 'b.', sigma_plot[2], sigma_plot[3], 'b.',
                      sigma_plot[4], sigma_plot[5], 'b.', markersize=2)
            ax.set_ylim([-5, 5])
            ax_2.set_ylim([-15, 10])
            ax.text(0.5, 0.5, str(sno0), transform=ax.transAxes, va='top', ha='left', fontsize=16)
            # ax_2.plot(**sigma_plot)
            if site_order == 5:
                plt.savefig('gridded_plot')

    elif pixel_type == 'interest':
        valid0 = npr_plot[1] > 0
        if detectors:
            plot_funcs.plot_subplot([npr_plot, sigma_plot, sigma_plot],
                                    [[npr_conv_bar[:, 1], npr_conv_bar[:, 2]],
                                     [sigma_conv_bar_max[:, 1], sigma_conv_bar_max[:, 2]],
                                     [sigma_conv_bar_min[:, 1], sigma_conv_bar_min[:, 2]]],
                                    main_label=['npr', '$\sigma^0$ mid', '$\sigma^0$'],
                                    figname='ms_pixel_test_%d_%d' % (sno0, smap_id), x_unit='doy',
                                    vline=v_line_local, vline_label=doy_all,
                                    h_line2=[[0, 1, 2], [0.01, 1, -1], [':', ':', ':']],
                                    annotation_sigma0=[text_qa_w, text_qa_m],
                                    y_lim=[[0, 1, 2], [[0, 0.1], [-18, -4], [-18, -4]]],
                                    y_lim2=[[0, 1, 2], [[0, 0.05], [0, 6], [-6, 0]]],
                                    type_y2='bar'
                                    )
        else:
            valid0 = npr_plot[1] > 0
            plot_funcs.plot_subplot([npr_plot[:, valid0], sigma_plot],
                                    [npr_conv_plot, sigma_conv_plot],
                                    main_label=['npr', '$\sigma^0$ mid'],
                                    figname='ms_pixel_test_%d_%d' % (sno0, smap_id), x_unit='doy', vline=v_line_local, vline_label=doy_all,
                                    h_line=[[-1], [0], [':']]
                                    )
        return 0