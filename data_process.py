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
from osgeo import gdal
import glob

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


def cal_emi(sno, tb, doy, hrs='emi'):
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
    if hrs == 'emi':
        tb_doy = get_doy(doy) + 0.75
        date_sno = sno[0][:]
        t_5 = sno[1][:]
        ind = np.argwhere(np.in1d(date_sno, tb_doy))
        # consider the complement sets: elements in tb_doy but not in date_no, {tb_doy - date_no}
        missing_date = np.setdiff1d(tb_doy, np.intersect1d(date_sno, tb_doy))  # this missing date of in situ mearsurement
        miss_ind = np.in1d(tb_doy, missing_date).nonzero()  # or try ind = np.in1d(date_sno, tb_doy).nonzero()
        tb_doy = np.delete(tb_doy, miss_ind[0])
        tb = np.delete(tb, miss_ind[0])
        emissivity = tb/(t_5[ind].T + 273.12)
        emissivity = emissivity[0][:]
        return emissivity
    else:
        # time pass of satellite
        if type(hrs) is int:
            tb_doy = doy + np.round(hrs)/24.0
        else:
            tb_doy = [doy[i0] + hrs[i0]/24.0 for i0 in range(0, doy.size)]
            pause = 0
        date_sno = sno[0][:]
        data = sno[1][:]

        #ind = np.append(ind, np.argwhere(np.in1d(date_sno, tb_doy)))
        index0 = np.array([])
        for t in tb_doy:
            ind0 = np.where(np.abs(date_sno - t) < 1e-2)
            if ind0[0].size < 1:
                pause = 0
            else:
                index0=np.append(index0, ind0[0])
        #ind2 = np.int_(np.sort(ind, axis=0))

        # remove the unreal sudden change
        index = index0.astype(int)
        # i_valid_change = np.where(np.abs(np.diff(data[index]))<75)
        data_valid = data[index]  #[i_valid_change[0] + 1]
        date_valid = date_sno[index]  #[i_valid_change[0] + 1]
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
        dis = (lat - site_info[1])**2 + (lon - site_info[2])**2
        near_ind = np.argsort(dis)
        near0, near1, near2 = near_ind[0], near_ind[1], near_ind[2]
        # coordinate and value of center and 2 direction side
        ref = np.array([lon[near0], lat[near0]])
        side0, side1 = np.array([lon[near1], lat[near1]]), np.array([lon[near2], lat[near2]])
        # vector of two sides of envelope square
        vec0 = np.array([side0[0] - ref[0], side0[1] - ref[1]])
        vec1 = np.array([side1[0] - ref[0], side1[1] - ref[1]])
        modul0, modul1 = np.sqrt(vec0[0]**2 + vec0[1]**2), np.sqrt(vec1[1]**2 + vec1[0]**2)
        cos0_1 = np.dot(vec0, vec1)/modul0/modul1

        #  select the diagonal point to the ref from the remained points
        diags = np.array([lon[near_ind[3: ]] - ref[0], lat[near_ind[3: ]] - ref[1]])  # vectors of other points as [xn, yn].T
        diags_mod = np.sqrt(diags[0, :]**2 + diags[1, :]**2)  # modules of the above vectors
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
            dig_cosine = np.dot(diag_vec, diags)/(diags_mod * diag_mod)
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
        loc4 = np.array([[lon_u[ind_l], lat_u[ind_up]], [lon_u[ind_l], lat_u[ind_low]], [lon_u[ind_r], lat_u[ind_up]], [lon_u[ind_r], lat_u[ind_low]]])
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
    line_x = (locs[2][1] - locs[0][1]) * x - (locs[2][0] - locs[0][0]) * y - locs[0][0] * locs[2][1] + locs[2][0] * locs[0][1]
    line_y = (locs[1][1] - locs[0][1]) * x - (locs[1][0] - locs[0][0]) * y - locs[0][0] * locs[1][1] + locs[1][0] * locs[0][1]
    d2xline = line_x**2/((locs[2][1] - locs[0][1])**2 + (locs[2][0] - locs[0][0])**2)  # squared distance to x line
    d2yline = line_y**2/((locs[1][1] - locs[0][1])**2 + (locs[1][0] - locs[0][0])**2)  # squared distance to y line
    dx = math.sqrt((locs[0][0] - locs[2][0])**2 + (locs[0][1] - locs[2][1])**2)  # length of x line
    dy = math.sqrt((locs[0][0] - locs[1][0])**2 + (locs[0][1] - locs[1][1])**2)  # length of y line
    d = (locs[0][0] - site_info[2])**2 + (locs[0][1] - site_info[1])**2  # square of station to up-left
    disx1 = math.sqrt(d - d2xline)
    disx = np.array([dx - disx1, disx1])
    disy2 = math.sqrt(d - d2yline)
    disy = np.array([[disy2], [dy - disy2]])
    fmn = np.array([[vars[1], vars[0]], [vars[3], vars[2]]])
    # var = 1/(dx * dy) * disx * fmn * disy
    var = 1/(dx*dy) * np.dot(np.dot(disx, fmn), disy)
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
        print '%s was not included in \n %s' % (attributes, filename[48: ])
        return [-9999 + i*0 for i in range(len(atts[1]))], -9999, -9999, -1
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
    if dis_inner.size>0:  # not empty
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
        return [-9999 + i*0 for i in range(len(atts[1]))], -9999, -9999, -1


def read_all_pixel(filename, site_no, prj='np', dat='smap_tb', disref=0.5):
    # the lat/lon of pixel
    path0 = 'result_07_01/txtfiles/SMAP_pixel/'
    txt_list=glob.glob(path0+'subcent_'+site_no+'*.txt')
    attributes = 'North_Polar_Projection'
    atts = site_infos.get_attribute('np', sublayer=dat)
    hf_l0 = h5py.File(filename, 'r')
    if attributes not in hf_l0:
        hf_l0.close()
        stat = -1
        print '%s was not included in \n %s' % (attributes, filename[48: ])
        return -1, -1, -1
    hf = hf_l0[attributes]  # open second layer
    lat = hf['cell_lat'].value
    lon = hf['cell_lon'].value
    # read TB within each pixels
    # initial
    dis_list  = np.array([-1, -1, -1, -1])
    tb_pixels = np.zeros([len(txt_list), len(atts[1])]) - 9999
    for n_p, pixel0_path in enumerate(txt_list):
        txt_fname = pixel0_path.split('/')[-1]
        pixel_dis = txt_fname.split('_')[-1].split('.')[0]  # the distance from center to site, as id for each pixel
        dis_list[n_p] = float(pixel_dis)
        pixel_info = np.loadtxt(pixel0_path, delimiter=',')
        s_info = [0, pixel_info[1, 4], pixel_info[0, 4]]
        dis = bxy.cal_dis(s_info[1], s_info[2], lat, lon)
        inner = np.where(dis<1)
        dis_inner = dis[inner[0]]
        interpolated_value = []
        if dis_inner.size>0:  # not empty
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
        print '%s was not included in \n %s' % (attributes, filename[48: ])
        return [-9999 + i*0 for i in range(len(atts[1]))], -9999, -9999, -1
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
    if dis_inner.size>0:  # not empty
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
        return [-9999 + i*0 for i in range(len(atts[1]))], -9999, -9999, -1
    return 0


def interpolation_spatial(values, lon, lat, site_no, disref=0.25):
    s_info = site_infos.change_site(site_no)
    dis = bxy.cal_dis(s_info[1], s_info[2], lat, lon)
    nn = np.where(dis < disref)  # the nearest neighbours within a radius distance "dis_ref"
    dis_nn = dis[nn[0]]
    if dis_nn.size>0:  # not empty
        nn_id = nn[0]
        print site_no, 'has neighbor: ', nn_id.size
        values_nn = values[nn_id]
        value_interp = bxy.dis_inter(dis_nn, values_nn)  # perform 2d spatial interpolation
        return value_interp, dis_nn, [lon[nn_id], lat[nn_id]], 0
    else:
        return 0,0,0,-1


def interp_geo(x, y, z, orb, site_no, disref=0.5, f=None, incidence=None):
    f_bad = np.where(f > 0)
    if any(f_bad[0]>0):
        print 'not usable at %s' % site_no
    s_info = site_infos.change_site(site_no)
    v_interp = []
    v_ind = []
    for orbit in [0, 1]:
        i_orb = np.where(orb==orbit)
        if len(i_orb)> 0:
            xo, yo = x[i_orb], y[i_orb]
            dis = (yo-s_info[1])**2 + (xo - s_info[2])**2
            if dis.size < 1:
                continue
            inner = np.where(dis < disref**2)[0]
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
                di = 1.0/dis_inner
                v1 = np.sum(di*z_inner, axis=1)
                vn = np.sum(di)
                if vn > 0:
                    v_interp.append(v1 * 1.0/vn)
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
            ft[ind[0]:  ] = sig_s
            sums.append(np.nansum((y2 - ft)**2))
            date_sums.append(timing)
        else:
            "print not such a date"
    ts = np.argmin(sums)
    ind = np.where(doy == date_sums[ts])
    sig_w = np.nanmean(y2[0: ind[0]])
    sig_s = np.nanmean(y2[ind[0]:])
    ft = np.zeros(y2.size)
    ft[0: ind[0]] = sig_w
    ft[ind[0]:  ] = sig_s
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
    crit = np.where(diff > d_mean+3*d_std)  # warnings if there are nan value in diff
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
        series[crit[0].astype(int)+1] = np.nan
        for idx in id0[crit[0].astype(int)+1]:
            series[idx] = (np.nanmean(series[idx-2: idx]) + np.nanmean(series[idx+1: idx+3]))/2.0
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
    w = np.ones(n,)/n
    r = np.convolve(series, w, mode='same')
    valid = np.linspace((n-1)/2, series.size-(n+1)/2, series.size - n+1, dtype=int)
    return r, valid


def n_convolve(series, n):  # this is useless!!!!!!!!!!!!!!!
    win = np.ones(n, )
    k = series.size + win.size - 1
    n = win.size
    m = series.size
    w = np.zeros(k)
    for i in range(0, k):
        wsize = range(max(0, i+1-n), min(i, m-1))  # the index saved in moving window
        tp = series[wsize]
        num_nan = np.count_nonzero(np.isnan(tp))  # find nans in the window
        mw = win/(n - num_nan)
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
    k = series.size + win.size - 1
    n = win.size
    m = series.size
    w = np.zeros(k)
    for i in range(0, k):
        if np.isnan(series[min(i, m-1)]):
            w[i] = np.nan
            continue
        wsize = range(max(0, i+1-n), min(i, m-1)+1)  # the index saved in moving window
        num_nan = np.count_nonzero(np.isnan(series[wsize]))  # find nans in the window
        mw = win/(n - num_nan)  # set new mean value for this window
        for j in wsize:
            if np.isnan(series[j]):
                tp = 0  # nan value set as 0
            else:
                tp = series[j]
            w[i] += tp * mw[i-j]
    valid = np.linspace(n-1, m-1, m-n+1, dtype=int)
    return w, valid


def gauss_conv(series, sig=1, size=17):
    """

    :param series:
    :param sig:
    :return:
        ig: the true/false index of value that is not a nan
    """
    size = 6*sig+1
    ig = ~np.isnan(series)
    x = np.linspace(-size/2+1, size/2, size)
    # filterz = (np.sqrt(1/(2*sig**2*np.pi)))*((-x)/sig**2)*np.exp(-x**2/(2*sig**2))
    filterz = ((-x)/sig**2)*np.exp(-x**2/(2*sig**2))
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
    return f1, ig


def gauss_cov_snr(series, peaks_iter, t_series, t_window=[[80, 150], [245, 350]], sig=1, size=17, s_type='other'):
    ons1 = -1
    ons2 = -1
    for sig in np.arange(5, 11, 1):
        size = 6*sig+1
        gsize = 6*sig/2
        ig = ~np.isnan(series)
        x = np.linspace(-size/2+1, size/2, size)
        filterz = ((-x)/sig**2)*np.exp(-x**2/(2*sig**2))
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
        if (ons1!=-1) & (ons2!=-1):
            return np.array([ons1, ons2]), f1, ig, convs, max_convs, min_convs, sig
    return np.array([ons1, ons2]), -1, -1, -1. -1, -1


def find_infleclt_signal(tabs, window, convs, t_convs):
    onset = 0
    snr_threshold = 0
    t_date = t_convs-365
    convs_win = convs[(t_date>10) & (t_date<70)]  # winter window: 10~70
    mean_wind, std_win = np.nanmean(convs_win), np.nanstd(convs_win)
    snr_threshold = [mean_wind-3*std_win, mean_wind+2*std_win]
    snr_thershold = [100, -100]
    dayz = tabs[:, 1]-365
    window_trans = (dayz > window[0]) & (dayz < window[1])
    v_z = tabs[:, -1][window_trans]
    onset_all = dayz[window_trans][(v_z>snr_threshold[1])|(v_z<snr_threshold[0])]
    if onset_all.size > 0:
        onset = onset_all[0]
    else:
        onset = -1

    # np.set_printoptions(precision=5, suppress=True)
    # print 'Filter width: ', w0
    # trans2winter[i0] = np.abs(np.nanmean(extremum[window_trans])/conv_std_win)
    return onset


def gauss2nd_conv(series, sig=1):
    size = 6*sig+1
    ig = ~np.isnan(series)
    x = np.linspace(-size/2+1, size/2, size)
    # filterz = (np.sqrt(1/(2*sig**2*np.pi)))*((-x)/sig**2)*np.exp(-x**2/(2*sig**2))
    filterz = (1.0*x**2/sig**4 - 1.0/sig**2)*np.exp(-1.0*x**2/(2.0*sig**2))
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


def find_inflect(maxtab, mintab, typez='tri', typen='sig'):
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
            thaw = find_infleclt_anual(maxtab, [60, 180], -1)
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
    min = np.nanmean(series[sz/10: sz*3/10+1])
    max = np.nanmean(series[sz*7/10: sz*9/10+1])
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
        valid_dateh = t_h[(w-1)/2: t_h.size-(w+1)/2 + 1]  # date --> smooth[valid]
        ffnpr_s, valid = n_convolve2(ffnpr, w)
        ffnpr_s = ffnpr_s[valid]  # x: valid_dateh, y: ffnpr_s[valid]
        g_npr, i_gaussian = gauss_conv(ffnpr_s)  # ffnpr, t_h
        g_npr_valid_n = (g_npr[g_size: -g_size] - np.nanmin(g_npr[g_size: -g_size]))\
                                /(np.nanmax(g_npr[g_size: -g_size]) - np.nanmin(g_npr[g_size: -g_size]))   # normalized
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
    lenv += (num-2)*(buff*2+1)
    A1 = np.zeros([len(insitu)+1, min(lenv, (num-1) * (buff*2+1))])
    j = 1
    for att in insitu:  # include the date
        A0 = np.array([])
        for i in range(1, num, 1):
            att_valid = att[ind_bool][ind_size: -ind_size]
            A0 = np.append(A0, att_valid[int(onset[i, 0])-buff: int(onset[i, 0])+1+buff])
            # print onset[i, 0]
        A1[j] = A0
        j += 1
    for i in range(1, num, 1):  # add a onset flag, if 1 it is the onset
        A1[0][A1[1] == onset[i, 1]] = 1.11
    # set the txt write format
    # A1.shape[0]
    formt = ['%1.2f', '%1d', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.8f', '%1.8f']
    np.savetxt('onset_test'+id+'.txt', np.transpose(A1), delimiter=',', fmt=formt,
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
        sm_win = sm[np.where((datez>15)&(datez<60))]
        sm_ref = np.nanmean(sm_win)  # reference moisture: DOY 15 to 60
        if sm_ref == 0:
            sm_ref == 0.01
        i_th = np.where((datez < 150) & (datez > 75))[0]  # for thawing index
        i_frz = np.where((datez > 250) & (datez < 330))[0]

        sm_th = sm[i_th]
        d_sm0 = np.diff(sm_th)  # the daily change in thawing period
        t_th = t[i_th]
        rcs = (sm_th - sm_ref)/sm_ref  # change of soil moisture during thawing
        n = 0
        sm_window = 5
        for rc in rcs:
            if rc:
                rc_weeky, sm_weeky = rcs[n: n+7], sm_th[n: n+sm_window]
                d_sm_weeky = d_sm0[n: n+6]
                if sm_weeky.size < sm_window:  # array length protection, we consider the next 7 days's value after the possible onsets
                    break
                d5 = (np.max(sm_weeky) - sm_weeky[0])
                d3 = np.max(sm_weeky[0: 3]) - sm_weeky[0]
                d_sm = sm_weeky - sm_ref
                day_th = np.where(rc_weeky>0.2)[0]
                if (d5>10)|(d3>5):  # (np.mean(sm_weeky)-sm_ref>5) (day_th.size > 3) &
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
        weeky_insitu = np.array([datez[i_th[n]+t_0[0]: i_th[n]+t_0[1]],
                                 t[i_th[n]+t_0[0]: i_th[n]+t_0[1]],
                                 sm[i_th[n]+t_0[0]: i_th[n]+t_0[1]]])
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
                if i2+3 > t_th.size:
                    print 'cant find the -1 degC timing'
                elif np.mean(t_th[i2: i2+4]) > -1: # 4-day > -1 degC
                    onset_temperature = [datez[i_th[i2]]]
                    t_in = 1
            elif t0 > 1:
                if i2+3 > t_th.size:
                    print 'cant find the 1 degC timing'
                elif np.mean(t_th[i2: i2+4]) > 1:
                    day_temperature2 = [datez[i_th[i2]]]
                    t_in = 0
                    break


        # freezing detect
        sm_fr = sm[i_frz]
        t_fr = t[i_frz]
        rcd = (sm_fr[1: ] - sm_fr[0: -1])/sm_fr[0: -1]
        frz_time = 5
        n = 0
        for rc in rcd:
            if t_fr[n]<1:
                if rc<-0.05:
                    rc_weeky, sm_weeky = rcd[n: n+7], sm_fr[n: n+frz_time]
                    print 'sm change is %.1f - %.1f' % (sm_fr[n+1], sm_fr[n])
                    for sm_i in sm_weeky:
                        print sm_i
                    day_frz = np.where(rc_weeky<0.01)[0]
                    d3 = sm_weeky[0]-np.min(sm_weeky[0: 3])
                    print sm_weeky.size
                    d10 = sm_weeky[0]-np.min(sm_weeky)
                    if (sm_weeky[0]<10)|(np.mean(sm_fr[n: ]) < 10):  # ((day_frz.size>3) | (np.mean(sm_fr[n+1: n+8]) < 10)):
                        # ((d10 > 10)|(sm_weeky[0]<10))&(np.mean(sm_fr[n: ]) < 10):
                        if t[i_frz[n+1]] > 1:
                            print 'the soil temperature is two high: %f.1' % t[i_frz[n+1]]
                            n += 1
                            continue
                        else:
                            break
                # elif (t[i_frz[n+1]] < 1)&((sm[i_frz[n+1]] - sm[i_frz[n+7]]) > 10):
                #     print t[n], rcd[n], sm[n]
                #     break

            n += 1

        if n+1 >= i_frz.size:
            n -= 1
        onset.append(datez[i_frz[n+1]])
        # temperature constraint
        # greater than -1 celsius for 4 or more days
        i2 = -1

        t_frz = t[i_frz]
        for t0 in t_frz:
            i2 += 1
            if t0 < 1:
                # check if the date is beyond the thawing period
                if i2+3 > t_frz.size:
                    print 'cant find the -1 degC timing'
                elif all(t_frz[i2: i2+3] < 1):  # 4-day > -1 degC
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


def pass_zone_plot(input_x, input_y, value, pre_path, fname=[], z_min=-20, z_max=-4, prj='merc', odd_points=[],
                   title_str=' '):
    """
    :param input_x:
    :param input_y:
        the cordinate value, x is longitude and y is latitude
    :param value: the value to be plot on map
    :param fname: 'spatial_ascat'+ID_description_passzone.png
    :return:
    """
    fig = plt.pyplot.figure(figsize=[8, 8])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title(title_str)
    if prj == 'merc':
        m = Basemap(llcrnrlon=-170, llcrnrlat=54, urcrnrlon=-140, urcrnrlat=72,
                    resolution='l', area_thresh=1000., projection=prj,
                    lat_ts=50., ax=ax)
    elif prj == 'aea':
        # m = Basemap(width=3e6, height=3e6, resolution='l', projection=prj, lat_ts=62, lat_0=62, lon_0=-150., ax=ax)
        m = Basemap(width=15e5, height=2e6, resolution='l', projection=prj, lat_1=55, lat_2=65, lon_0=-154., lat_0=63, ellps='WGS84', ax=ax)
    elif prj == 'ease':
        m = Basemap(llcrnrlon=-170, llcrnrlat=54, urcrnrlon=-140, urcrnrlat=72, resolution='l', area_thresh=10000, lat_ts=30, projection='cea')
    im = m.pcolormesh(input_x, input_y, value, vmin=z_min, vmax=z_max, latlon=True)
    m.drawcoastlines()
    m.drawcountries()
    parallel = np.arange(50, 80, 5)
    meridian = np.arange(-170, -130, 5)
    m.drawparallels(parallel, labels=[1, 0, 0, 0], fontsize=18)
    m.drawmeridians(meridian, labels=[0, 0, 0, 1], fontsize=18)
    cb = m.colorbar(im)
    if fname == 'thaw_14_7' or fname =='freeze_14_7':
        cb.set_label('Days')
    else:
        cb.set_label('DOY 2016')
    # if len(odd_points) > 0:
    if odd_points.size < 20:
        if type(odd_points[0]) is list:
            x = np.array([odd_points[i][0] for i in range(0, len(odd_points))])
            y = np.array([odd_points[i][1] for i in range(0, len(odd_points))])

            # m.scatter(odd_points[0], odd_points[1], marker='x', color='k', latlon=True)
        else:
            x, y = odd_points[0], odd_points[1]
        print 'odd point: ', x, y
        m.scatter(x, y, marker='x', color='k', latlon=True)
        # m.scatter(x[-7:], y[-5:], marker='s', color='k', latlon=True)
    else:
        for s0 in odd_points:
            x, y = s0[-2], s0[-3]
            m.scatter(x, y, marker='x', color='k', latlon=True)
    plt.rcParams.update({'font.size': 18})
    plt.pyplot.savefig(pre_path+fname+'.png', dpi=120)
    # add_site(m, pre_path+'spatial_ascat'+fname+'_site')
    plt.pyplot.close()
    return 0


def add_site(m, fname):
        site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177', '2210',
                    '1089', '1233', '2212', '2211']
        s_lat, s_lon, s_name = [], [], []
        for no in site_nos:
            s_name.append(site_infos.change_site(no)[0])
            s_lat.append(site_infos.change_site(no)[1])
            s_lon.append(site_infos.change_site(no)[2])
        m.scatter(s_lon, s_lat, 5, marker='*', color='k', latlon=True)
        plt.pyplot.savefig(fname+'.png', dpi=120)


def ascat_nn(x, y, z, orb, site_no, disref=0.5, f=None, center=False):
    """
    return the indices of pixels within a distance from centroid to site.
    :return: min_inds: 0: id of NN at ascending, 1: id of NN at des
    """
    f_bad = np.where(f > 0)
    if any(f_bad[0]>0):
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
            xo, yo = x[i_orb], y[i_orb]
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
    return min_inds, np.append(dis_list[0], dis_list[1])


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
        t_win = [time2[i], time2[i]+1]
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
        '1175', '950', '2065', '967', '2213', '949', '950', '960', '962', '968','1090',  '1177',  '2081', '2210', '1089', '1233', '2212', '2211']
    angle_range = np.arange(25, 65, 0.1)
    prefix = site_infos.get_data_path('_07_01')
    for site in siteno:
        txtname = './result_05_01/ascat_point/ascat_s'+site+'_2016.npy'
        txt_table = np.load(txtname)
        id_orb = txt_table[:, -1] == 0
        out = txt_table[id_orb, :].T
        xt, y1t, y2t = np.array(out[0]), np.array(out[1]), np.array(out[2])
        x_inc, y_sig = out[5: 8], out[2: 5]
        tx = out[0]
        # 4 periods
        p1, p2, p3, p4 = tx<80, (tx>90)&(tx<120), (tx>160)&(tx<250), tx>270
        p_no = 0
        p_con = p1 | p3
        for p in [p1, p2, p3, p4, p_con]:
            p_no += 1
            fig = pylt.figure(figsize=[4, 3])
            ax = fig.add_subplot(111)
            inci1 = x_inc[:, p]
            sig1 = y_sig[:, p]
            if any(inci1[0] > 1e2):
                inci1*=1e-2
            if any(sig1[0] < -1e4):
                sig1*=1e-6
            ax.plot(inci1[0], sig1[0], 'bo')
            plot_funcs.plt_more(ax, inci1[1], sig1[1])
            plot_funcs.plt_more(ax, inci1[2], sig1[2], symbol='go')
            # linear regression
            x = inci1.reshape(1, -1)[0]
            y = sig1.reshape(1, -1)[0]
            a, b = np.polyfit(x, y, 1)
            f = np.poly1d([a, b])
             # r squared
            y_mean = np.sum(y)/y.size
            sstot = np.sum((y-y_mean)**2)
            ssres = np.sum((y-f(x))**2)
            r2 = 1 - ssres/sstot
            fig.text(0.15, 0.15, '$y = %.2f x + %.f$\n $r^2 = %.4f$' %(a, b, r2))
            plot_funcs.plt_more(ax, x, f(x), symbol='r-', fname=prefix+'Incidence_angle_p'+str(p_no))
            fig.clear()
            pylt.close()
        fig2 = pylt.figure(figsize=[4, 3])
        ax = fig2.add_subplot(111)
        n = 0
        for p in [p1, p3]:
            inci1 = x_inc[:, p]
            sig1 = y_sig[:, p]
            if any(inci1[0] > 1e2):
                inci1*=1e-2
            if any(sig1[0] < -1e4):
                sig1*=1e-6
            # linear regression
            x = inci1.ravel()
            y = sig1.ravel()
            a, b = np.polyfit(x, y, 1)
            f = np.poly1d([a, b])
             # r squared
            y_mean = np.sum(y)/y.size
            sstot = np.sum((y-y_mean)**2)
            ssres = np.sum((y-f(x))**2)
            r2 = 1 - ssres/sstot
            if n < 1:
                ax.plot(inci1[0], sig1[0], 'ro', markersize=5)
                plot_funcs.plt_more(ax, inci1[1], sig1[1], symbol='go', marksize=5)
                plot_funcs.plt_more(ax, inci1[2], sig1[2], symbol='bo', marksize=5)
                #ax.set_ylim([-16, -6])
                plot_funcs.plt_more(ax, angle_range, f(angle_range), symbol='r-', marksize=5)
                fig2.text(0.15, 0.15, '$y = %.2f x + %.f$\n $r^2 = %.4f$' %(a, b, r2))
            else:
                plot_funcs.plt_more(ax, inci1[0], sig1[0], symbol='r^', marksize=5)
                plot_funcs.plt_more(ax, inci1[1], sig1[1], symbol='g^', marksize=5)
                plot_funcs.plt_more(ax, inci1[2], sig1[2], symbol='b^', marksize=5)
                plot_funcs.plt_more(ax, angle_range, f(angle_range), symbol='b-', marksize=5)
                fig2.text(0.45, 0.75, '$y = %.2f x + %.f$\n $r^2 = %.4f$' %(a, b, r2))
            n += 1
        ax.set_ylim([-18, -4])
        pylt.savefig(prefix+'Incidence_angle_'+site+'.png', dpi=120)
        pylt.close()
    return 0


def ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g = 5, pp=False, norm=False,
                      txt_path='./result_05_01/ascat_point/', is_sub=False, order=1):  # orbit default is 0, ascending, night pass
    # read series sigma
    site = site_no
    site_nos = site_infos.get_id(site, mode='single')
    for si0 in site_nos:
        if is_sub is False:
            txtname = glob.glob(txt_path+'ascat_s'+si0+'*')[0]
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
        doy_tp2 = times_ascat[-2] + (times_ascat[0]-2015)*365 + np.max(np.array([(times_ascat[0]-2016), np.zeros(times_ascat[0].size)]), axis=0)
        passhr = times_ascat[-1]*1.0

        # angular normalization
        # tx = doy_tp[1]+1 -365
        tx = doy_tp2 + 1 -365
        p = (tx > 0) & (tx < 60)
        x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
        a, b = np.polyfit(x, y, 1)
        print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)
        f = np.poly1d([a, b])

        # angular dependency for each ob mode
        p_win, p_su = (tx>0) & (tx<60), (tx>150)&(tx<260)
        x_trip_win, x_trip_su = ascat_ob[p_win, 5: 8], ascat_ob[p_su, 5: 8]
        y_trip_win, y_trip_su = ascat_ob[p_win, 2: 5], ascat_ob[p_su, 2: 5]
        a_array, b_array, d_array = np.zeros(6)-1, np.zeros(6)-1, np.zeros(3) - 99
        for i1 in range(0, 3):
            a_w, b_w = np.polyfit(x_trip_win[:, i1], y_trip_win[:, i1], 1)
            a_s, b_s = np.polyfit(x_trip_su[:, i1], y_trip_su[:, i1], 1)
            a_array[i1], b_array[i1] = a_w, b_w
            a_array[i1+3], b_array[i1+3] = a_s, b_s
            d_array[i1] = np.mean(y_trip_su[:, i1]) - np.mean(y_trip_win[:, i1])
        a_coef_name = 'ascat_linear_a.txt'
        b_coef_name = 'ascat_linear_b.txt'
        with open(a_coef_name, 'a') as t_file:
            t_file.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, a_array[0], a_array[1], a_array[2],
                                                              a_array[3], a_array[4], a_array[5], d_array[0], d_array[1], d_array[2]))
        with open(b_coef_name, 'a') as t_file:
            t_file.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, b_array[0], b_array[1], b_array[2],
                                                              b_array[3], b_array[4], b_array[5], d_array[0], d_array[1], d_array[2]))

        sig_m = ascat_ob[:, 3]
        inc_m = ascat_ob[:, 6]
        # sig_m = ascat_ob[:, 2]  # changed using the forwards mode to increase the winter summer difference
        # inc_m = ascat_ob[:, 5]
        sig_mn = sig_m - (ascat_ob[:, 6]-45)*a
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
        g_size = 6*sig_g/2
        if order == 1:
            g_sig, ig2 = gauss_conv(sig_mn, sig=sig_g, size=2*g_size+1)  # non-smoothed
        elif order == 2:
            g_sig, ig2 = gauss2nd_conv(sig_mn, sig=sig_g)
        g_sig_valid = 2*(g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                    /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))-1
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
        return [tx[ig2][g_size: -g_size]+365, g_sig_valid, g_sig_valid_non], \
               [tx, sig_mn, inc_d], \
               onset,\
               pass_hr_d,\
               [u_doy+365, pass_hr_d],\
               [max_gsig_s, min_gsig_s]  # g_sig[g_size: -g_size] g_sig_valid
    # read site data
    # plot
    return 0


def get_ascat_series(site_no, orb_no=0, inc_plot=False, sigma_g = 5, pp=False, norm=False,
                      txt_path='./result_05_01/ascat_point/', is_sub=False, order=1):  # orbit default is 0, ascending, night pass
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
            txtname = glob.glob(txt_path+'ascat_s'+si0+'*')[0]
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
        doy_tp2 = times_ascat[-2] + (times_ascat[0]-2015)*365
        passhr = times_ascat[-1]*1.0

        # angular normalization
        # tx = doy_tp[1]+1 -365
        tx = doy_tp2 + 1 -365
        p = (tx > 0) & (tx < 60)
        x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
        a, b = np.polyfit(x, y, 1)
        print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)
        f = np.poly1d([a, b])

        # angular dependency for each ob mode
        p_win, p_su = (tx>0) & (tx<60), (tx>150)&(tx<260)
        x_trip_win, x_trip_su = ascat_ob[p_win, 5: 8], ascat_ob[p_su, 5: 8]
        y_trip_win, y_trip_su = ascat_ob[p_win, 2: 5], ascat_ob[p_su, 2: 5]
        a_array, b_array, d_array = np.zeros(6)-1, np.zeros(6)-1, np.zeros(3) - 99
        for i1 in range(0, 3):
            a_w, b_w = np.polyfit(x_trip_win[:, i1], y_trip_win[:, i1], 1)
            a_s, b_s = np.polyfit(x_trip_su[:, i1], y_trip_su[:, i1], 1)
            a_array[i1], b_array[i1] = a_w, b_w
            a_array[i1+3], b_array[i1+3] = a_s, b_s
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
        sig_mn = sig_m - (ascat_ob[:, 6]-45)*a
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



def ascat_order2(site_no, orb_no=0, inc_plot=False, sigma_g = 5, pp=False, norm=False,
                      txt_path='./result_05_01/ascat_point/', is_sub=False):
    # read sigmas
    txtname = glob.glob(txt_path+'ascat_s'+site_no+'*')[0]
    ascat_all = np.load(txtname)
    if ascat_all.size < 1:
        return [-1, -1, -1], [-1, -1, -1], [-1, -1], -1, [-1, -1], [-1, -1]
    id_orb = ascat_all[:, -1] == orb_no
    ascat_ob = ascat_all[id_orb]
    # transform utc time to local time
    doy_passhr = bxy.time_getlocaltime(ascat_all, 0)
    doy = doy_passhr[1]
    doy[doy_passhr[0]<2016]-=365
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
    sig_mn = sig_m - (ascat_ob[:, 6]-45)*a
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
    g_size = 6*sig_g/2
    g_sig, ig2 = gauss2nd_conv(sig_mn, sig=sig_g)  # non-smoothed
    g_sig_valid = 2*(g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))-1
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
        pylt.savefig('./result_07_01/inc_mid_'+site_no+'.png')
        fig.clear()
        pylt.close()
        # ons_site = sm_onset(sm5_date-365, sm5_daily, t5_daily)
    # onset based on ascat

    # actual pass time
    # sec_ascat = bxy.timetransform(ascat_ob[:, 1], '20000101 00:00:00', '%Y%m%d %X', tzone=True)
    # doy, passhr = np.modf((sec_ascat)/3600.0/24.0)[1] + 1, np.round(np.modf((sec_ascat)/3600.0/24.0)[0]*24)
    return [tx[ig2][g_size: -g_size]+365, g_sig_valid, g_sig_valid_non], \
           [tx, sig_mn, inc_d], \
           onset,\
           pass_hr_d,\
           [u_doy+365, pass_hr_d],\
           [max_gsig_s, min_gsig_s]  # g_sig[g_size: -g_size] g_sig_valid
    return 0


def ascat_alaska_onset(ob='AS', norm=False, std=3, version='old', target00=[142, 237]):
    if version == 'new':
        f_doc = 'all_year_observation'
        doy0 = get_doy('20151001')-365
        doy0 = get_doy('20151001')-365
        doy_range = np.arange(doy0, doy0+500)-1
    else:
        doy_range = np.arange(0, 365)
        f_doc = 'all_year_observation_old'
    if norm:
        isnorm = 'norm'
        folder_path = './result_05_01/ascat_resample_norms/ascat_resample_'+ob+'/'
    else:
        isnorm = 'orig'
        folder_path = './result_05_01/ascat_resample_'+ob+'/'
    print 'the sigma data is %s, %s' % (isnorm, ob)
    d0 = datetime.date(2016, 1, 1)  # change from 2016 1 1
    # initial a base grid
    base0 = np.load(folder_path+'ascat_20160101_resample.npy')
    sigma_3d = np.zeros([base0.shape[0], base0.shape[1], doy_range.size])
    inc_3d = np.zeros([base0.shape[0], base0.shape[1], doy_range.size])
    onset_2d = np.zeros([base0.shape[0], base0.shape[1]])
    lg_grid, lat_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                        np.load('./result_05_01/other_product/lat_ease_grid.npy')
    fpath0 = './result_05_01/onset_result/%s/' % f_doc  # saving path
    if not os.path.exists(fpath0+'ascat_all_2016_'+isnorm+'_'+ob+'.npy'):
        # find the daily sigma and mask files of AK
        i=0
        for doy in doy_range:
            d_daily = d0 + timedelta(doy)
            d_str = d_daily.strftime('%Y%m%d')
            ak_file = bxy.find_by_date(d_str, folder_path)
            for f1 in ak_file:
                if f1.find('resample') > 0:
                    v = np.load(folder_path+f1)
                    sigma_3d[:, :, i] = v
                # if f1.find('incidence') > 0:
                #     agl = np.load(folder_path+f1)
                #     inc_3d[:, :, i] = agl
            i+=1
        np.save(fpath0+'ascat_all_2016_'+isnorm+'_'+ob, sigma_3d)
        # np.save(fpath0+'ascat_inc_2016_'+ob, inc_3d)
    else:
        sigma_3d = np.load(fpath0+'ascat_all_2016_'+isnorm+'_'+ob+'.npy')
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
    row_test, col_test = np.where(mask0)[0], np.where(mask0)[1] # test of target point
    nodata_count, land_count = 0, 0
    onset_1d = [[], []]
    for s1 in sigma_land:
        # edge detection
        g_size = 3*std
        i_sig_valid = (s1 != 0)
        doy_i = doy_range[i_sig_valid]
        sigma_i = s1[i_sig_valid]
        if sigma_i.size >= 120:
            g_sig, ig2 = gauss_conv(sigma_i, sig=std, size=6*std+1)  # non-smoothed
            g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                        /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
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
                plot_funcs.pltyy(doy_i, sigma_i, fpath0+'odd_point', '$\sigma_0$ (dB)',
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
    np.save(fpath0+'ascat_onset_0_2016_'+isnorm+'_'+ob+'_w'+str(std), onset_2d)
    onset_2d[mask0] = onset_1d[1]
    np.save(fpath0+'ascat_onset_1_2016_'+isnorm+'_'+ob+'_w'+str(std), onset_2d)

    # mask_1d = mask0.reshape(1, -1)
    # sigma_2d = sigma_3d.reshape(1, -1, 365)
    # i_land = np.where(mask_1d == True)
    # for l0 in i_land[1]:
    #     sig_land0 = sigma_2d[0, l0]
    pause = True


def smap_alaska_onset(mode='tb', std=3, version='old'):
    if version=='new':
        f_doc = 'all_year_observation'
        f_doc2 = 'all'
        doy0 = get_doy('20151001')-365
        doy_range = np.arange(doy0, doy0+500) - 1
    else:
        f_doc = 'all_year_observation_old'
        f_doc2 = 'all'
        doy_range = np.arange(0, 365)
    fpath0 = './result_05_01/onset_result/%s/' % f_doc
    ob = 'AS'
    folder_path = './result_05_01/smap_resample_'+ob+'/%s/' % f_doc2
    base0 = np.load(folder_path+'smap_20160105_tbv_resample.npy')
    d0 = datetime.date(2016, 1, 1)
    tbv_3d = np.zeros([base0.shape[0], base0.shape[1], doy_range.size])
    tbh_3d = np.zeros([base0.shape[0], base0.shape[1], doy_range.size])
    onset_2d = np.zeros([base0.shape[0], base0.shape[1]])
    if not os.path.exists(fpath0+'smap_all_2016_tbv_'+ob+'.npy'):
        # find the daily sigma and mask files of AK
        print 'file not found: ', fpath0+'smap_all_2016'+'_'+mode+'_'+ob+'.npy'
        i=0
        for doy in doy_range:
            d_daily = d0 + timedelta(doy)
            d_str = d_daily.strftime('%Y%m%d')
            ak_file = bxy.find_by_date(d_str, folder_path)
            for f1 in ak_file:
                if f1.find('v_resample') > 0:
                    v = np.load(folder_path+f1)
                    tbv_3d[:, :, i] = v
                if f1.find('h_resample') > 0:  # other attributs
                    agl = np.load(folder_path+f1)
                    tbh_3d[:, :, i] = agl
            i += 1
        np.save(fpath0+'smap_all_2016_tbv_'+ob, tbv_3d)
        np.save(fpath0+'smap_all_2016_tbh_'+ob, tbh_3d)
        # np.save('./result_05_01/onset_result/smap_all_2016_tbh_'+ob, tbh_3d)
    else:
        tbv_3d = np.load(fpath0+'smap_all_2016_tbv_'+ob+'.npy')
        tbh_3d = np.load(fpath0+'smap_all_2016_tbh_'+ob+'.npy')
    # read mask for land
    mask0 = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
    landid = mask0 == 1
    v_land = tbv_3d[landid, :]
    if mode == 'npr':
        tbv_3d_ma = np.ma.masked_array(tbv_3d, mask=[tbv_3d < 100])
        tbh_3d_ma = np.ma.masked_array(tbh_3d, mask=[tbh_3d < 100])
        npr = (tbv_3d_ma - tbh_3d_ma)/(tbv_3d_ma + tbh_3d_ma)
        v_land = npr[landid]
        npr_test = npr[51, 68, :]
    mask_row, mask_col = np.where(landid)[0], np.where(landid)[1]
    test_id = np.where((mask_row==51) & (mask_col==68))
    nodata_count = 0
    land_count = 0
    onset_1d = [[], []]
    # npr_test = npr[21, 52, :]
    for s1 in v_land:
        # edge detection
        g_size = 3*std
        if mode == 'npr':
            i_tbv_valid = (s1 != 0) & (s1 != -9999) & (s1 != 1) & (s1 != -1)
        else:
            i_tbv_valid = ((s1 != 0) & (s1 != -9999))
        doy_i = doy_range[i_tbv_valid]
        tbv_i = s1[i_tbv_valid]
        if tbv_i.size >= 120:
            g_sig, ig2 = gauss_conv(tbv_i, sig=std, size=2*g_size+1)  # non-smoothed
            g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                        /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
            if mode=='npr':
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
    np.save(fpath0+'smap_onset_0_2016'+'_'+mode+'_'+ob+'_w'+str(std), onset_2d)
    onset_2d[landid] = onset_1d[1]
    np.save(fpath0+'smap_onset_1_2016'+'_'+mode+'_'+ob+'_w'+str(std), onset_2d)
    return 0


def ascat_onset_map(ob, odd_point=[], product='ascat', mask=False, std=4, mode=['_norm_'], version='old',
                    f_win=[0, 0], t_win=[0, 0]):
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
        if len(odd_point)>0:
            if type(odd_point[0]) is not list:
                odds = np.array(odd_point).T
                print odds[1]
                dis_odd = bxy.cal_dis(odds[1], odds[0], lats_grid.ravel(), lons_grid.ravel())
                index = np.argmin(dis_odd)
                row = int(index/lons_grid.shape[1])
                col = index - (index/lons_grid.shape[1]*lons_grid.shape[1])
                print row, col, lons_grid[row, col], lats_grid[row, col]
        for key in ob:
            for m in mode:
                onset_0_file = prefix+'ascat_onset_0'+'_2016'+m+key+'_w'+str(std)+'.npy'
                onset0 = np.load(onset_0_file)
                onset_1_file = prefix+'ascat_onset_1'+'_2016'+m+key+'_w'+str(std)+'.npy'
                onset1 = np.load(onset_1_file)
                if mask is True:
                    mask_snow = np.load(anc_direct+'snow_mask_125s.npy')
                    onset0 = np.ma.masked_array(onset0, mask=[mask_snow==0])
                    onset1 = np.ma.masked_array(onset1, mask=[mask_snow==0])
                fpath1 = 'result_08_01/'
                pass_zone_plot(lons_grid, lats_grid, onset0, fpath1,
                               fname='onset_0'+m+key+'_w'+str(std),
                               z_max=180, z_min=50, odd_points=odd_point)
                pass_zone_plot(lons_grid, lats_grid, onset1, fpath1,
                               fname='onset_1'+m+key+'_w'+str(std),
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
            out_bound = lons_grid>-141.0
            onset0[out_bound], onset1[out_bound], onset0_14[out_bound], onset1_14[out_bound] = 0, 0, 0, 0

            if odd_point.size == 4:
                odd_lon = odd_point[2]
                odd_lat = odd_point[3]
                odd_onset = onset0[odd_point[0], odd_point[1]]
            else:
                print 'add all location with labels'
            tbv0 = h0[u'cell_tb_v_aft'].value
            tbv0[tbv0<0] = 0
            mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
            onset0 = np.ma.masked_array(onset0, mask=[(onset0==0)|(mask==0)])
            onset1 = np.ma.masked_array(onset1, mask=[(onset1==0)|(mask==0)])
            onset0_14 = np.ma.masked_array(onset0_14, mask=[(onset0_14==0)|(mask==0)])
            onset1_14 = np.ma.masked_array(onset1_14, mask=[(onset1_14==0)|(mask==0)])
            # mask the snow cover
            mask_snow = np.load('./result_05_01/other_product/snow_mask_360_2.npy')
            onset0 = np.ma.masked_array(onset0, mask=[mask_snow!=0])
            onset1 = np.ma.masked_array(onset1, mask=[mask_snow!=0])
            onset0_14 = np.ma.masked_array(onset0_14, mask=[mask_snow!=0])
            onset1_14 = np.ma.masked_array(onset1_14, mask=[mask_snow!=0])
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
            pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname=thawname, z_max=60, z_min=150, prj='aea',
                           odd_points=odd_point[:, [0, 2, 3, 4]], title_str=thawtitle)  # fpath1
            pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname=frname, z_max=250, z_min=340, prj='aea',
                           odd_points=odd_point[:, [1, 2, 3, 4]], title_str=frtitle)

            # pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname=thawname, z_max=60, z_min=150, prj='aea',
            #                odd_points=[odd_lon, odd_lat], title_str=thawtitle)  # fpath1
            # pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname=frname, z_max=250, z_min=340, prj='aea',
            #                odd_points=[odd_lon, odd_lat], title_str=frtitle)
            # pass_zone_plot(lons_grid, lats_grid, onset1_14-onset1, fpath1, fname='freeze_14_7', z_max=-20, z_min=20, prj='aea',
            #                odd_points=[odd_lon, odd_lat], title_str='Freezing onsets bias')
            # pass_zone_plot(lons_grid, lats_grid, onset0_14-onset0, fpath1, fname='thaw_14_7', z_max=-20, z_min=20, prj='aea',
            #                odd_points=[odd_lon, odd_lat], title_str='Thawing onsets bias')

    else:
        lons_grid = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
        lats_grid = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
        indicators = [product]
        for ind in indicators:
            onset0 = np.load(fpath1+'smap_onset_0_2016_'+ind+'_AS'+'_w'+str(std)+'.npy')
            onset1 = np.load(fpath1+'smap_onset_1_2016_'+ind+'_AS'+'_w'+str(std)+'.npy')
            if mask is True:
                mask_snow = np.load(anc_direct+'snow_mask_360s.npy')
                onset0 = np.ma.masked_array(onset0, mask=[mask_snow==0])
                onset1 = np.ma.masked_array(onset1, mask=[mask_snow==0])
            fpath1 = 'result_08_01/'
            pass_zone_plot(lons_grid, lats_grid, onset0, fpath1, fname='onset_0_smap_'+ind+'_w'+str(std),
                                   z_max=180, z_min=50, odd_points=odd_point)  # fpath1
            pass_zone_plot(lons_grid, lats_grid, onset1, fpath1, fname='onset_1_smap_'+ind+'_w'+str(std),
                                   z_max=360, z_min=250, odd_points=odd_point)


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
    onset_path = prefix+ft+'_2016'+mode+key+'_w%s.npy' % str(std)
    onset_ak = np.load(onset_path)
    if area == 'area_0':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (onset_ak < 60) & (lons_grid < -155) & (lons_grid > -160)
    elif area == 'area_1':
        odd_thaw = (onset_ak > 0) & (lats_grid < 60) & (onset_ak < 80) & (lons_grid < -160)
    elif area == 'area_2':
        odd_thaw = (lats_grid > 55) & (lats_grid < 60) & (lons_grid < -155) & (lons_grid > -160) & (onset_ak < 75) & (onset_ak > 0)
    elif area == 'area_3':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid < -150) & (lons_grid > -155) & (onset_ak > 140) & (onset_ak > 0)
    elif area == 'area_4':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid > -150) & (lons_grid < -145) & (onset_ak < 100)
    elif area == 'area_5':
        odd_thaw = (lats_grid > 60) & (lats_grid < 63) & (lons_grid > -160) & (lons_grid < -155) & (onset_ak > 120)
    elif area == 'area_6':
        odd_thaw = (lats_grid > 65) & (lats_grid < 67) & (lons_grid > -160) & (lons_grid < -155) & (onset_ak > 130) & (onset_ak > 0)
    elif area == 'area_7':
        odd_thaw = (lats_grid > 66) & (lats_grid < 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_ak > 135) & (onset_ak > 0)
    elif area == 'area_8':
        odd_thaw = (lats_grid > 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_ak > 120) & (onset_ak > 0)
    elif area == 'area_88':
        odd_thaw = (lats_grid > 70) & (lons_grid > -155) & (lons_grid < -150) & (onset_ak > 140) & (onset_ak > 0)
    elif area == 'area_9':
        odd_thaw = (lats_grid > 67) & (lats_grid < 70) & (lons_grid > -150) & (lons_grid < -145) & (onset_ak > 150) & (onset_ak < 179)
    elif area == 'area_5f':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid > -145) & (lons_grid < -140) & (onset_ak >345) & (onset_ak > 0)
    elif area == 'area_11':
        odd_thaw = (lats_grid > 68) & (lats_grid < 70) & (lons_grid > -145) & (lons_grid < -140) & (onset_ak < 140) & (onset_ak > 0)
    elif area == 'area_12':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -163) & (lons_grid < -160) & (onset_ak > 125) & (onset_ak < 150)
    id_thaw = np.where(odd_thaw)
    odd_row, odd_col = id_thaw[0], id_thaw[1]
    sigma_path = ('./result_05_01/onset_result/%s/ascat_all_2016' % result_doc) +mode+key+'.npy'
    sigma_3d = np.load(sigma_path)
    inc_path = ('./result_05_01/onset_result/%s/ascat_inc_2016_' % result_doc) + key+'.npy'
    inc_3d = np.load(inc_path)
    odd_sigma = sigma_3d[odd_thaw]
    odd_inc = inc_3d[odd_thaw]
    odd_lon, odd_lat = lons_grid[odd_thaw], lats_grid[odd_thaw]
    id_odd = np.arange(0, odd_lon.size)
    odd_value = onset_ak[odd_thaw]
    np.save('./result_05_01/onset_result/odd_thaw_'+area, odd_sigma)
    np.save('./result_05_01/onset_result/odd_thaw_inc'+area, odd_inc)
    # np.save()
    fpath = './result_08_01/onset_result/odd_thaw_%s.txt' % area
    np.savetxt(fpath,
               np.array([id_odd.T, odd_value.T, odd_lon.T, odd_lat.T, odd_row.T, odd_col.T]).T, fmt='%d, %d, %.8f, %.8f, %d, %d')

    # find the rows and cols number in smap_map
    dis_grid = bxy.cal_dis(odd_rc[1], odd_rc[0], lats_grid, lons_grid)
    min_dis = np.min(dis_grid)
    row_col = np.where(dis_grid==min_dis)
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
    series_tbv_2016 = prefix+'smap_all_2016_tbv_'+orbit+'.npy'
    series_tbh_2016 = prefix+'smap_all_2016_tbh_'+orbit+'.npy'
    odd_value_path = prefix+'odd_value_smap'+'.txt'
    onset_value = prefix+'smap_onset_%s_2016_%s_%s_w%s.npy' % (ft, mode, orbit, str(std))  #smap_onset_0_2016_npr_AS_w4.npy
    odd_series = prefix+'series/'+area+orbit
    fsize = [8, 5]
    # Test onset by specifying an area
    onset_0_file = np.load(onset_value)
    if area == 'area_0':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (onset_0_file < 80) & (lons_grid < -145) & (lons_grid > -150)
    elif area == 'area_1':
        odd_thaw = (onset_0_file > 0) & (lats_grid < 60) & (onset_0_file < 80) & (lons_grid < -160)
    elif area == 'area_2':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid < -145) & (lons_grid > -150) & (onset_0_file > 110) & (onset_0_file > 0)
    elif area == 'area_3':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid < -150) & (lons_grid > -155) & (onset_0_file > 140) & (onset_0_file > 0)
    elif area == 'area_4':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid > -150) & (lons_grid < -145) & (onset_0_file < 100)
    elif area == 'area_5':
        odd_thaw = (lats_grid > 70) & (lats_grid < 80) & (lons_grid > -155) & (lons_grid < -150) & (onset_0_file > 130)
    elif area == 'area_6':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_0_file > 300) & (onset_0_file > 0)
    elif area == 'area_7':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_0_file > 100) & (onset_0_file > 0)
    elif area == 'area_8':
        odd_thaw = (lats_grid > 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_0_file > 120) & (onset_0_file > 0)
    elif area == 'area_88':
        odd_thaw = (lats_grid > 70) & (lons_grid > -155) & (lons_grid < -150) & (onset_0_file > 140) & (onset_0_file > 0)
    elif area == 'area_9':
        odd_thaw = (lats_grid > 67) & (lats_grid < 70) & (lons_grid > -150) & (lons_grid < -145) & (onset_0_file > 150) & (onset_0_file < 179)
    elif area == 'area_100':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -145) & (lons_grid < -140) & (onset_0_file > 300) & (onset_0_file > 0)
    elif area == 'area_11':
        odd_thaw = (lats_grid > 68) & (lats_grid < 70) & (lons_grid > -145) & (lons_grid < -140) & (onset_0_file < 140) & (onset_0_file > 0)
    id_thaw = np.where(odd_thaw)

    #read data
    odd_row, odd_col = id_thaw[0], id_thaw[1]
    tbh_3d = np.load(series_tbh_2016)
    tbv_3d = np.load(series_tbv_2016) ####
    if mode == 'tb':
        indicator_3d = tbv_3d
    else:
        indicator_3d = (tbv_3d - tbh_3d)/(tbv_3d+tbh_3d)
    odd_sigma = indicator_3d[odd_thaw]
    odd_lon, odd_lat = lons_grid[odd_thaw], lats_grid[odd_thaw]
    id_odd = np.arange(0, odd_lon.size)
    odd_value = onset_0_file[odd_thaw]
    np.savetxt(odd_value_path,
               np.array([id_odd.T, odd_value.T, odd_lon.T, odd_lat.T, odd_row.T, odd_col.T]).T, fmt='%d, %d, %.8f, %.8f, %d, %d')

    # find the rows and cols number in smap_map
    dis_grid = bxy.cal_dis(odd_rc[1], odd_rc[0], lats_grid, lons_grid)
    min_dis = np.min(dis_grid)
    row_col = np.where(dis_grid==min_dis)  # row_col[0][0] is row, row_col[1][0] is col
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
            g_size = 3*std
            g_sig, ig2 = gauss_conv(v_valid, sig=std, size=2*g_size+1)
            g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                            /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
            max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, t_valid[g_size: -g_size])
            onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=mode)
            # test the temporal variation
            # print 'the temporal change of tb%s is: ' % pol, v_valid[g_size: -g_size]
            fig = pylt.figure(figsize=fsize)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d, Days: %d, distance: %.1f'
                         % (loni, lati, rowi, coli, onset[0], onset[1], sum(v<0), min_dis))
            plot_funcs.pltyy(t_valid, v_valid, 'test', '$T_{B%s}$ (K)' % pol,
                             t2=t_valid[ig2][g_size: -g_size], s2=g_sig[g_size: -g_size],
                             ylim=[230, 280], symbol=['bo', 'g-'], handle=[fig, ax])
            ax.set_xlim([0, 365])
            pylt.savefig(odd_series+'_'+pol+'_w'+str(std), dpi=300)
    elif mode=='npr':
        iv_valid = (v != 0) & (v != 1) & (v != -1) & (v != -0) & (v != -9999) & (~np.isnan(v))
        # iv_valid = ((v != 0) & (v != -9999))
        # iv_valid = (v != 0) & (v > -999)
        v_valid = v[iv_valid]
        t = np.arange(ini_doy, v.size) - 1
        t_valid = t[iv_valid]
        g_size = 3*std
        g_sig, ig2 = gauss_conv(v_valid, sig=std, size=2*g_size+1)  # non-smoothed
        g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
        max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig[g_size: -g_size], 1e-4, t_valid[g_size: -g_size])
        onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=mode)
        print 'onset in odd_point test:', onset
        # the temporal change of npr/tb
        id_test = int(max_gsig_s[max_gsig_s[:, 1] == onset[0]][0, 0])
        # test the temporal variation
        print 'the temporal change of npr is: ', v_valid[g_size: -g_size][id_test-1: id_test+7]
        fig = pylt.figure(figsize=fsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d distance: %.1f'
                     % (loni, lati, rowi, coli, onset[0], onset[1], min_dis))
        plot_funcs.pltyy(t_valid, v_valid, odd_series+'_'+mode+'_w'+str(std), '$NPR$ (%)',
                         t2=t_valid[ig2][g_size: -g_size], s2=g_sig[g_size: -g_size],
                         ylim=[0, 1.2*np.max(v_valid)], symbol=['bo', 'g-'], handle=[fig, ax])
        ax.set_xlim([0, 365])
        pylt.savefig(odd_series+'_'+mode+'_w'+str(std), dpi=300)

def ascat_test_odd_point_plot(sig_file, area_file, p_id, area='odd_arctic1', odd_point=[], orb='AS', mode=[], ptype='sig', std=4, start_date='20160101'):
    prefix = './result_05_01/onset_result/odd_series_smap/'
    prefix_new = './result_05_01/onset_result/all_year_observation/'
    odd_info = np.loadtxt(area_file, delimiter=',')
    sig = np.load(sig_file)
    # odd_onset = odd_info[:, 1]
    # id of testing pixel
    # p_id = np.argsort(odd_onset)[odd_onset.size//2]
    # p_id = 37
    if len(odd_point) > 0:
        v, loni, lati, rowi, coli, dis = odd_point[0], odd_point[1], odd_point[2], odd_point[3], odd_point[4], odd_point[5]
    else:
        v, loni, lati, rowi, coli = sig[p_id], odd_info[:, 2][p_id], odd_info[:, 3][p_id], \
                                odd_info[:, 4][p_id], odd_info[:, 5][p_id]
    print 'the odd pixel is (%.5f, %.5f)' % (loni, lati), '\n'
    iv_valid = v != 0
    v_valid = v[iv_valid]
    t = np.arange(0, v.size)+get_doy(start_date)-365-1
    t_valid = t[iv_valid]
    g_size = 3*std
    g_sig, ig2 = gauss_conv(v_valid, sig=std, size=2*g_size+1)
    # g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
    #             /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))

    g_sig_valid = g_sig[g_size: -g_size]
    max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, t_valid[g_size: -g_size])
    onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=ptype)
    fig = pylt.figure(figsize=[8, 5])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d, Days: %d, distance: %.1f' % (loni, lati, rowi, coli, onset[0], onset[1], sum(v<0), dis))
    print prefix+area+orb
    plot_funcs.pltyy(t_valid, v_valid, prefix_new+'series/'+area+orb+'_w'+str(std), '$\sigma_0$ (dB)',
                     t2=t_valid[ig2][g_size: -g_size], s2=g_sig_valid,
                     ylim=[1.2*np.min(v_valid), 0.8*np.max(v_valid)], symbol=['bo', 'g-'], handle=[fig, ax])
    ax.set_xlim([0, 365])
    pylt.savefig(prefix_new+'series/'+area+orb+'_w'+str(std), dpi=300)

    return 0


def ascat_test_odd_angular(inc, sig, t=None, period = [], area='test'):
    if len(period) > 0:
        id = (t > period[0]) & (t < period[1])
        sig, inc = inc[id], sig[id]
    a, b = np.polyfit(inc, sig, 1)
    f = np.poly1d([a, b])
    fig = pylt.figure(figsize=[4, 3])
    ax = fig.add_subplot(111)
    sig_mean = np.sum(sig)/sig.size
    sstot = np.sum((sig-sig_mean)**2)
    ssres = np.sum((sig-f(inc))**2)
    r2 = 1 - ssres/sstot
    fig.text(0.15, 0.15, '$y = %.2f x + %.f$\n $r^2 = %.4f$' %(a, b, r2))
    ax.plot(inc, sig, 'k.')
    plot_funcs.plt_more(ax, inc, f(inc), symbol='r-',
                        fname='./result_05_01/onset_result/odd_series/odd_angular_'+area)
    fig.clear()
    pylt.close()


def ascat_gap(ascat_series, id):
    d0 = np.unique(ascat_series[:, 0])
    d1 = np.arange(ascat_series[:, 0][0], d0.size+ascat_series[:, 0][0])
    fig = pylt.figure(figsize=[8, 2])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(d0, d0-d1, '.')
    pylt.savefig('./result_05_01/point_result/ascat/data_gap'+id+'.png')


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
    arys=[]
    for i in xrange(1, ds.RasterCount+1):
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
        row[i] = id0/array_shape[1]
        col[i] = id0 - id0/array_shape[1]*array_shape[1]
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
            pass_hr = np.modf(sec_now/(24*3600.0))[0] * 24
            n, bins, patches = ax0.hist(pass_hr, 50, normed=1, facecolor='green', alpha=0.75)
        ax0.set_ylabel(name0)
        axs.append(ax0)
    pylt.savefig(prefix+'ancplot/'+site_no+orb+'anc.png')


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
    as_fname, des_fname = prefix+'txtfiles/site_tb/tb_'+site_no+'_A_2016.txt', prefix+'txtfiles/site_tb/tb_'+site_no+'_D_2016.txt'
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
                      bxy.timetransform(time_des, '20000101 11:58:56', '%Y%m%d %X', tzone=True)  # seconds from 20150101 (utc or local)
    pass_as, pass_des = np.modf(sec_as/(24*3600.0))[0] * 24, np.modf(sec_des/(24*3600.0))[0] * 24
    as_date, des_date = np.modf(sec_as/(24*3600.0))[1]+1, np.modf(sec_des/(24*3600.0))[1]+1
    tbv_a, tbv_d = [as_date, site_as[:, n_tbv]], [des_date, site_des[:, n_tbv]]
    tbh_a, tbh_d = [as_date, site_as[:, n_tbh]], [des_date, site_des[:, n_tbh]]
    # check the passtime distributionn
    # fig = pylt.figure(figsize=[5, 5])
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot()
    if p=='v':
        obd, tb_date, pass_hr = cal_obd(tbv_a, tbv_d, pass_as, pass_des)
    elif p=='h':
        obd, tb_date, pass_hr = cal_obd(tbh_a, tbh_d, pass_as, pass_des)
    elif p=='npr':
        npr_a, npr_d = (tbv_a[1]-tbh_a[1])/(tbv_a[1]+tbh_a[1]), (tbv_d[1]-tbh_d[1])/(tbv_d[1]+tbh_d[1])
        obd, tb_date, pass_hr = cal_obd(np.array([tbh_a[0], npr_a]), np.array([tbh_d[0], npr_d]), pass_as, pass_des)
    elif p=='vh':
        obd_v, tb_date, pass_hr = cal_obd(tbv_a, tbv_d, pass_as, pass_des)
        obd_h, tb_date, pass_hr = cal_obd(tbh_a, tbh_d, pass_as, pass_des)
    elif p=='vh0':
        obd_v, tb_date, pass_hr = cal_obd(tbv_a, tbv_d, pass_as, pass_des)
        obd_h, tb_date, pass_hr = cal_obd(tbh_a, tbh_d, pass_as, pass_des)
    elif p=='sig':
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
    site_date = np.arange(366, 366+365)
    sm_as, sm_date_as = cal_emi(sm5, ' ', site_date, hrs=passtime[0])
    sm_des, sm_date_des = cal_emi(sm5, ' ', site_date, hrs=passtime[1])
    mm_daily, mm_doy = read_site.read_measurements(site_no, "snow", site_date)
    sm_as[sm_as<-90], sm_des[sm_des<-90] = np.nan, np.nan

    # plotting parameters
    # tb_date, obd_v, obd_h; p1

    sm_date365 = sm_date_as.astype(int)-365  # p3
    sm_change = np.diff(sm_as)
    idd = np.in1d(sm_date365[1:], tb_date)
    tb_date-=365
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
    window0 = (date_npr_Emax>period[0])&(date_npr_Emax<period[1])  # window for thawing: 0~150, freezing: 250 ~350
    max_date_npr = date_npr_Emax[window0]
    i0_npr_Et = np.in1d(gau[0], max_date_npr)
    max_trans = [gau[0][i0_npr_Et], gau[1][i0_npr_Et], gau[2][i0_npr_Et]]  # maximum during transition (minimum for tb): date, norm vlaue, non_norm value
    i_Et_05 = max_trans[1] >= -1
    # the extremum date, normaled extremum, non-normaled ex
    max_date_npr, Emax, Emax_non = max_trans[0][i_Et_05], max_trans[1][i_Et_05], max_trans[2][i_Et_05]
    max_dsm_npr, max_soil_npr, max_swe_npr = \
        np.zeros(max_date_npr.size), np.zeros(max_date_npr.size), np.zeros(max_date_npr.size)
    # some in situ data
    i_dsm = 0
    for d0 in max_date_npr:
        i_window_swe = [np.argmin(np.abs(np.fix(swe[0]-d0+np.fix(3*k_width)))), np.argmin(np.abs(np.fix(swe[0]-d0-np.fix(3*k_width))))]
        i_window_sm = [np.argmin(np.abs(np.fix(sm[0]-d0+np.fix(3*k_width)))), np.argmin(np.abs(np.fix(sm[0]-d0-np.fix(3*k_width))))]
        i_window_soil = [np.argmin(np.abs(np.fix(tsoil[0]-d0+np.fix(3*k_width)))), np.argmin(np.abs(np.fix(tsoil[0]-d0-np.fix(3*k_width))))]
        dsm = (sm[1][i_window_sm[1]] - sm[1][i_window_sm[0]])
        dsoil = np.nanmean(tsoil[1][i_window_soil[0]: i_window_soil[1]+1])
        dswe = (swe[1][i_window_swe[1]] - swe[1][i_window_swe[0]])

        max_dsm_npr[i_dsm], max_soil_npr[i_dsm], max_swe_npr[i_dsm] = dsm, dsoil, dswe
        i_dsm += 1
    # save results during transition
    filedoc = 'result_07_01/methods/'
    h5name1 = 'transition_'+site_no+'_v00.h5'
    narr1 = np.array([max_date_npr, Emax, Emax_non, max_dsm_npr, max_soil_npr, max_swe_npr])  # date, value, change of sm, mean t and change of swe
    print [layers+"/width_"+str(k_width)]
    bxy.h5_writein(filedoc+h5name1, layers+"/width_"+str(k_width), narr1)
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
        trans_date=[70, 80, 150]
    else:
        trans_date=[70, 260, 350]
    # thaw
    # if indic!='tb':
    for w0 in w:
        i0+=1

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
                                                                              tbob='_A_', sig0=w0, order=1, seriestype='tb')  # result npr
            conv0[-1] *= -1
            tp = [peakdate[1], peakdate[0]]
            peakdate = tp
        else:
            tbv1, tbh1, series, conv0, ons1, sitetime, peakdate = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                              tbob='_A_', sig0=w0, order=1)  # result npr
        if ft[0] == 'freeze':
            conv0[-1] *= -1
            peaksz = peakdate[1].T
        elif ft[0] == 'thaw':
            peaksz = peakdate[0].T
        conv_date = conv0[0]
        convs = conv0[-1]
        conv_win = convs[(conv_date>10) & (conv_date<trans_date[0])]
        conv_std_win = np.nanstd(conv_win)

        extremun_date = peaksz[1]
        extremum = peaksz[-1]
        window_trans = (extremun_date > trans_date[1]) & (extremun_date < trans_date[2])
        np.set_printoptions(precision=5, suppress=True)
        conv_trans = convs[((conv_date>75) & (conv_date<trans_date[2]))|((conv_date>260)&(conv_date<350))]  # thawing transition
        conv_trans_std = np.nanstd(conv_trans)
        conv_thaw = convs[(conv_date>75) & (conv_date<trans_date[2])]
        conv_thaw_std = np.nanstd(conv_thaw)
        print 'Filter width: ', w0
        # print 'winter: ',extremum[window_winter], '\n', 'mean: ', mean_winter, 'std: ', np.std(extremum[window_winter]), \
        #     '\n' 'transition: ', extremum[window_trans][ind_trans_valid], \
        #     'mean: ', np.nanmean(extremum[window_trans][ind_trans_valid]), '\n', 'transition date are: ', date_extremum[window_trans][ind_trans_valid], '\n'
        # trans2winter[i0] = np.abs(np.nanmean(extremum[window_trans][ind_trans_valid])/mean_winter)

        # trans2winter[i0] = np.abs(np.nanmean(extremum[window_trans])/conv_std_win)
        trans2winter[i0] = np.abs(conv_trans_std/conv_std_win)
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
        print '%s was not included in \n %s' % (attributes, filename[48: ])
        return [-9999 + i*0 for i in range(len(atts[1]))], -9999
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
    if dis_inner.size>0:  # not empty
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
        return [-9999 + i*0 for i in range(len(atts[1]))], -9999


def read_h5_latlon(h5name, latlon, att, orb=1):  # orb 0 is the AM pass
    """
    :param h5name:
    :param latlon: 0: lon, 1: lat, 2: dis; use distance as unique id
    :param att:
    :return: att_dict_list: a list that contains the <attribute dict> read for target pixels
    """
    # initial
    att_value = np.zeros([latlon.shape[0], len(att)+1]) - 1
    att_dict_list = []
    distance_id = []
    h0 = h5py.File(h5name)
    lats, lons = h0['Freeze_Thaw_Retrieval_Data/latitude'].value[orb].ravel(), h0['Freeze_Thaw_Retrieval_Data/longitude'].value[orb].ravel()
    for i, coord in enumerate(latlon.reshape(-1, 4)):
        index0 = (np.abs(lons - coord[1]) < 0.01) & (np.abs(lats - coord[2]) < 0.01)
        check_index0 = np.where(index0)
        jndex1 = np.where((np.abs(h0['Freeze_Thaw_Retrieval_Data/latitude'].value[orb]-coord[2])<0.01)
                          & (np.abs(h0['Freeze_Thaw_Retrieval_Data/longitude'].value[orb]-coord[1])<0.01))
        att_dict = {}
        for j, att0 in enumerate(att):
            value = h0[att0].value[orb].ravel()[index0]
            if value.size > 0.5:
                att_value[i, j] = h0[att0].value[orb].ravel()[index0]
            else:
                continue
        att_value[i, -1] = coord[3]
    return att_value, 1,  1


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
    hf0 = h5py.File('./tp/'+h5_newfile)
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
    date = [85, 85.875, 95, 95.875, 115.875, 277.875, 279.875,947.000]
    date = [a for a in range(95, 105)]
    date = [98+a*0.125 for a in range(0, 24)]
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
            pixel_id = ft_value[ft_value[:, 0] > -1, -1]  # get the pixel id based on the distance from station to pixel center
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
            pixel_id = ft_value[ft_value[:, 0] > -1, -1]  # get the pixel id based on the distance from station to pixel center
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
    dis_ref = bxy.cal_dis(yc, xc, yc, xc+0.01)
    step_lon = 6.0/dis_ref
    x0 = xc-step_lon*0.01
    x1 = xc+step_lon*0.01
    dis_ref = bxy.cal_dis(yc, xc, yc+0.01, xc)
    step_lat = 6.0/dis_ref
    y0 = yc+step_lat*0.01
    y1 = yc-step_lat*0.01
    locs = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])-np.array([xc, yc])
    theta = theta0/180*np.pi
    rot_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    new_loc = np.matmul(locs, rot_matrix)

    xy = np.array([[1, 1], [1, 2], [2, 2], [2, 1]]) - np.array([1.5, 1.5])
    xy_r1 = np.matmul(xy, rot_matrix)
    xy_new = xy_r1 + np.array([1.5, 1.5])

    theta3 = 37.0/180*np.pi
    rot_matrix2 = np.array([[np.cos(theta3), np.sin(theta3)], [-np.sin(theta3), np.cos(theta3)]])
    xy_new2 = np.matmul(xy_r1, rot_matrix2)+np.array([1.5, 1.5])
    fig0 =pylt.figure()
    ax = fig0.add_subplot(1, 1, 1)
    ax.plot(xy_new[:, 0], xy_new[:, 1], 'r-')
    ax.plot(xy_new2[:, 0], xy_new2[:, 1], 'b--')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    pylt.savefig('test_ascat_corner_roate')
    return new_loc + np.array([xc, yc])


def ascat_corner_rotate_new(pc, theta0):
    half_diag = 12.5/2.0 * np.sqrt(2.0)
    if theta0 < 45:
        theta2 = 45 - theta0
    xc, yc = pc[0], pc[1]
    # x0 = np.sqrt(xc**2-)
    # bxy.cal_dis(xc, yc, a, b)
    dis_lon = bxy.cal_dis(yc, xc, yc, xc+0.01)
    #step_lon = 6.0/dis_ref
    dis_lat = bxy.cal_dis(yc, xc, yc+0.01, xc)
    #step_lat = 6.0/dis_ref

    delta_x1 = half_diag*np.sin(theta2/180.0*np.pi)*1e-2/dis_lon
    delta_x2 = half_diag*np.cos(theta2/180.0*np.pi)*1e-2/dis_lon
    delta_y1 = half_diag*np.cos(theta2/180.0*np.pi)*1e-2/dis_lat
    delta_y2 = half_diag*np.sin(theta2/180.0*np.pi)*1e-2/dis_lat
    coe_matrix = np.zeros([8, 4])
    coe_matrix[[0, 1, 5, 6], [0, 1, 3, 2]] = -1
    coe_matrix[[2, 3, 4, 7], [0, 1, 2, 3]] = 1
    deltas = np.array([delta_x1, delta_x2, delta_y1, delta_y2]).reshape(4, -1)
    rotates = np.matmul(coe_matrix, deltas).reshape(4, -1)
    return np.array(pc) + rotates


def angular_correct(sigma0, inc0, inc_c=45):
    value = sigma0.ravel()
    angle = inc0.ravel()
    a, b = np.polyfit(angle, value, 1)
    print 'the linear regression coefficients, a: %.3f, b: %.3f' % (a, b)
    sig_mn = sigma0 - (inc0-inc_c)*a
    return sig_mn


# updated 0515/2018
def ascat_plot_series_copy(site_no, orb_no=0, inc_plot=False, sigma_g = 5, pp=False, norm=False,
                      txt_path='./result_05_01/ascat_point/', is_sub=False, order=1):  # orbit default is 0, ascending, night pass
    # read series sigma
    site = site_no
    site_nos = site_infos.get_id(site, mode='single')
    for si0 in site_nos:
        if is_sub is False:
            txtname = glob.glob(txt_path+'ascat_s'+si0+'*')[0]
        # txtname = './result_05_01/ascat_point/ascat_s'+si0+'_2016.npy'
        ascat_all = np.load(txtname)
        if ascat_all.size < 1:
            return [-1, -1, -1], [-1, -1, -1], [-1, -1], -1, [-1, -1], [-1, -1]
        id_orb = ascat_all[:, -1] == orb_no
        ascat_ob = ascat_all[id_orb]
        # transform utc time to local time
        sec_ascat = bxy.timetransform(ascat_ob[:, 1], '20000101 00:00:00', '%Y%m%d %X', tzone=True)
        doy_tp = np.modf((sec_ascat)/3600.0/24.0)
        doy = doy_tp[1]+1
        passhr = np.round(doy_tp[0]*24.0)
        times_ascat = bxy.time_getlocaltime(ascat_ob[:, 1], ref_time=[2000, 1, 1, 0])
        doy_tp2 = times_ascat[-2]
        passhr2 = times_ascat[-1]

        # angular normalization
        tx = doy_tp[1]+1 -365
        p = (tx > 0) & (tx < 60)
        x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
        a, b = np.polyfit(x, y, 1)
        print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)
        f = np.poly1d([a, b])

        # angular dependency for each ob mode
        p_win, p_su = (tx>0) & (tx<60), (tx>150)&(tx<260)
        x_trip_win, x_trip_su = ascat_ob[p_win, 5: 8], ascat_ob[p_su, 5: 8]
        y_trip_win, y_trip_su = ascat_ob[p_win, 2: 5], ascat_ob[p_su, 2: 5]
        a_array, b_array, d_array = np.zeros(6)-1, np.zeros(6)-1, np.zeros(3) - 99
        for i1 in range(0, 3):
            a_w, b_w = np.polyfit(x_trip_win[:, i1], y_trip_win[:, i1], 1)
            a_s, b_s = np.polyfit(x_trip_su[:, i1], y_trip_su[:, i1], 1)
            a_array[i1], b_array[i1] = a_w, b_w
            a_array[i1+3], b_array[i1+3] = a_s, b_s
            d_array[i1] = np.mean(y_trip_su[:, i1]) - np.mean(y_trip_win[:, i1])
        a_coef_name = 'ascat_linear_a.txt'
        b_coef_name = 'ascat_linear_b.txt'
        with open(a_coef_name, 'a') as t_file:
            t_file.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, a_array[0], a_array[1], a_array[2],
                                                              a_array[3], a_array[4], a_array[5], d_array[0], d_array[1], d_array[2]))
        with open(b_coef_name, 'a') as t_file:
            t_file.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (site_no, b_array[0], b_array[1], b_array[2],
                                                              b_array[3], b_array[4], b_array[5], d_array[0], d_array[1], d_array[2]))

        sig_m = ascat_ob[:, 3]
        inc_m = ascat_ob[:, 6]
        # sig_m = ascat_ob[:, 2]  # changed using the forwards mode to increase the winter summer difference
        # inc_m = ascat_ob[:, 5]
        sig_mn = sig_m - (ascat_ob[:, 6]-45)*a
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
        g_size = 6*sig_g/2
        if order == 1:
            g_sig, ig2 = gauss_conv(sig_mn, sig=sig_g, size=2*g_size+1)  # non-smoothed
        elif order == 2:
            g_sig, ig2 = gauss2nd_conv(sig_mn, sig=sig_g)
        g_sig_valid = 2*(g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                    /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))-1
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
        return [tx[ig2][g_size: -g_size]+365, g_sig_valid, g_sig_valid_non], \
               [tx, sig_mn, inc_d], \
               onset,\
               pass_hr_d,\
               [u_doy+365, pass_hr_d],\
               [max_gsig_s, min_gsig_s]  # g_sig[g_size: -g_size] g_sig_valid
    # read site data
    # plot
    return 0


def smap_alaska_grid(year=2016):  # not used 0718/2018
    # spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap/', orbit='A')
    # save_dir = './result_08_01/area/smap/SMAP_alaska_A_201601*.h5'
    match_file = './result_08_01/area/SMAP/SMAP_alaska_A_%d*.h5' % year
    h5_list = sorted(glob.glob('./result_08_01/area/SMAP/SMAP_alaska_A_2017*.h5'))
    min_row, min_col, max_row, max_col = 999, 999, 0, 0
    lons, lats, cols, rows = np.array([]), np.array([]), np.array([]), np.array([])
    for h5_file in h5_list:
        h0 = h5py.File(h5_file, 'r')
        h0_north = h0['North_Polar_Projection']
        cell_lon, cell_lat, cell_col, cell_row = \
            h0_north[u'cell_lon'], h0_north[u'cell_lat'], h0_north[u'cell_column'], h0_north[u'cell_row']
        min_row0, min_col0, max_row0, max_col0 = \
            np.min(cell_row), np.min(cell_col), np.max(cell_row), np.max(cell_col)
        min_row, min_col = min(min_row0, min_row), min(min_col0, min_col)
        max_row, max_col = max(max_row0, max_row), max(max_col0, max_col)
        lons, lats, cols, rows = \
            np.append(lons, cell_lon), np.append(lats, cell_lat), np.append(cols, cell_col), np.append(rows, cell_row)
    lat_grid, lon_grid = \
        np.zeros([max_row-min_row+1, max_col-min_col+1]), np.zeros([max_row-min_row+1, max_col-min_col+1])
    TBdir = '/media/327A50047A4FC379/SMAP/SPL1CTB.003/'

    for row0 in range(min_row, max_row+1):
        for col0 in range(min_col, max_col+1):
            idx = np.where((cols == col0) & (rows == row0))
            print idx
            break
        break


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
