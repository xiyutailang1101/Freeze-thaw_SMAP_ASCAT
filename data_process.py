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


def get_doy(date_string):
    """

    :param
        date_string: in the form of 'yyyymmdd'
    :return:
        doy_num: the num of doy
    """
    doy_num_list = []
    for strz in date_string:
        t = datetime.datetime.strptime(strz, '%Y%m%d').timetuple()
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
        tb_doy = doy + np.round(hrs)/24.0
        date_sno = sno[0][:]
        data = sno[1][:]
        # using
        ind = np.append(ind, np.argwhere(np.in1d(date_sno, tb_doy)))
        index = []
        for t in tb_doy:
            ind0 = np.where(np.abs(date_sno - t) < 1e-2)
            if ind0[0].size < 1:
                pause = 0
            else:
                index.append(ind0[0])
        ind2 = np.int_(np.sort(ind, axis=0))
        return data[index], date_sno[index]

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
    atts = site_infos.get_attribute('np', sublayer='smap_tb')
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

def rm_temporal(series):
    """
    <description>
    :param series:
    :return:
    """
    diff = np.diff(series)
    d_std = np.nanstd(diff)
    crit = np.where(np.abs(diff) > 3 * d_std)  # warnings if there are nan value in diff
    return crit


def rm_odd(series):
    crit = rm_temporal(series)
    series_out = series
    series_out[crit] = np.nan
    return series_out


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
    ig = ~np.isnan(series)
    x = np.linspace(-size/2+1, size/2, size)
    filterz = (np.sqrt(1/(2*sig**2*np.pi)))*((-x)/sig**2)*np.exp(-x**2/(2*sig**2))
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
        if max_day[0] > 365:
            max_day -= 365
            min_day -= 365
        if typen == 'tb':
            thaw = find_infleclt_anual(mintab, [60, 150], 0)
            freeze = find_infleclt_anual(maxtab, [240, 360], -1)
        else:
            thaw = find_infleclt_anual(maxtab, [60, 180], -1)
            freeze = find_infleclt_anual(mintab, [240, 360], 0)
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
    if mode == 'annual':
        sm_jan = sm[np.where((datez>15)&(datez<60))]
        sm_ref = np.nanmean(sm_jan)
        i_th = np.where((datez < 150) & (datez > 75))[0]  # for thawing index
        i_frz = np.where((datez > 270) & (datez < 330))[0]
        sm_th = sm[i_th]
        t_th = t[i_th]
        rcs = (sm_th - sm_ref)/sm_ref
        n = 0
        for rc in rcs:
            if rc > 0.2:
                rc_weeky, sm_weeky = rcs[n: n+7], sm_th[n: n+7]
                day_th = np.where(rc_weeky>0.2)[0]
                if (day_th.size > 3) & any(sm_weeky - sm_ref > 5):
                    if t[i_th[n]] < -1:
                        print 'temperature is too low for thawing: %f.1' % t[i_th[n]]
                        n += 1
                        continue
                    else:
                        break
            n += 1
        if n >= i_th.size:
            n -= 1
        print 'Thawing onset: the souil temperature is %f.2' % t[i_th[n]]
        onset = [datez[i_th[n]]]
        sm_fr = sm[i_frz]
        rcd = (sm_fr[1: ] - sm_fr[0: -1])/sm_fr[0: -1]
        n = 0
        for rc in rcd:
            if rc < -0.2:
                rc_weeky, sm_weeky = rcd[n: n+7], sm_fr[n+1: n+8]
                print 'sm change is %.1f - %.1f' % (sm_fr[n+1], sm_fr[n])
                for sm_i in sm_weeky:
                    print sm_i
                day_frz = np.where(rc_weeky<0.01)[0]
                if ((day_frz.size>3) & (np.mean(sm_fr[n+1: n+8])-sm_fr[n] < -4.9))|(any(sm_weeky<5)):
                    if t[i_frz[n+1]] > 1:
                        print 'the soil temperature is two high: %f.1' % t[i_frz[n+1]]
                        n += 1
                        continue
                    else:
                        break
            n += 1

        if n+1 >= i_frz.size:
            n -= 1
        onset.append(datez[i_frz[n+1]])
    return onset


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


def pass_zone_plot(input_x, input_y, value, pre_path, fname=[], z_min=-20, z_max=-4, prj='merc', odd_points=[]):
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
    if prj == 'merc':
        m = Basemap(llcrnrlon=-170, llcrnrlat=54, urcrnrlon=-140, urcrnrlat=72,
                    resolution='l', area_thresh=1000., projection=prj,
                    lat_ts=50., ax=ax)
    elif prj == 'laea':
        m = Basemap(width=4e6, height=4e6, resolution='l', projection=prj, lat_ts=62, lat_0=62, lon_0=-150., ax=ax)
    im = m.pcolormesh(input_x, input_y, value, vmin=z_min, vmax=z_max, latlon=True)
    m.drawcoastlines()
    m.drawcountries()
    parallel = np.arange(50, 80, 5)
    meridian = np.arange(-170, -130, 5)
    m.drawparallels(parallel, labels=[1, 0, 0, 0])
    m.drawmeridians(meridian, labels=[0, 0, 0, 1])
    cb = m.colorbar(im)
    if len(odd_points) > 0:
        if type(odd_points[0]) is list:
            x = [odd_points[i][0] for i in range(0, len(odd_points))]
            y = [odd_points[i][1] for i in range(0, len(odd_points))]
            # m.scatter(odd_points[0], odd_points[1], marker='x', color='k', latlon=True)
        else:
            x, y = odd_points[0], odd_points[1]
        # print 'odd point: ', x, y
        m.scatter(x, y, marker='x', color='k', latlon=True)
    plt.pyplot.savefig(pre_path+'spatial_ascat'+fname+'.png', dpi=120)
    add_site(m, pre_path+'spatial_ascat'+fname+'_site')
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


def ascat_nn(x, y, z, orb, site_no, disref=0.5, f=None, incidence=None):
    """
    return the indices of pixels within a distance from centroid to site.
    :return: min_inds: 0: id of NN at ascending, 1: id of NN at des
    """
    f_bad = np.where(f > 0)
    if any(f_bad[0]>0):
        print 'not usable at %s' % site_no
    s_info = site_infos.change_site(site_no)
    min_inds = [[], []]  # 0: ascending neighbour, 1: descending
    dis_list = [[], []]
    for orbit in [0, 1]:
        i_orb = np.where(orb == orbit)
        if i_orb[0].size > 0:
            xo, yo = x[i_orb], y[i_orb]
            dis = bxy.cal_dis(s_info[1], s_info[2], yo, xo)
            min_ind = dis < 9
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


def ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g = 5, pp=False):  # orbit default is 0, ascending, night pass
    # read series sigma
    site = site_no
    site_nos = site_infos.get_id(site, mode='single')
    for si0 in site_nos:
        txtname = './result_05_01/ascat_point/ascat_s'+si0+'_2016.npy'
        ascat_all = np.load(txtname)
        id_orb = ascat_all[:, -1] == orb_no
        ascat_ob = ascat_all[id_orb]
        # ascat_gap(ascat_ob, si0)
        # angular normalization
        tx = ascat_ob[:, 0] + np.round(ascat_ob[:, 1])/24.0
        p = (tx > 20) & (tx < 80)
        x, y = ascat_ob[p, 5: 8].reshape(1, -1)[0], ascat_ob[p, 2: 5].reshape(1, -1)[0]
        a, b = np.polyfit(x, y, 1)
        print 'the angular dependency: a: %.3f, b: %.3f' % (a, b)
        f = np.poly1d([a, b])

        # # plot the angular dependency
        # fig = pylt.figure(figsize=[4, 3])
        # ax = fig.add_subplot(111)
        # ax.plot(x, y, 'o')
        # y_mean = np.sum(y)/y.size
        # sstot = np.sum((y-y_mean)**2)
        # ssres = np.sum((y-f(x))**2)
        # r2 = 1 - ssres/sstot
        # fig.text(0.15, 0.15, '$y = %.2f x + %.f$\n $r^2 = %.4f$' %(a, b, r2))
        # plot_funcs.plt_more(ax, x, f(x), symbol='r-', fname='./result_05_01/point_result/ascat/Incidence_angle_p'+si0)
        # fig.clear()
        # pylt.close()

        sig_m = ascat_ob[:, 3]
        inc_m = ascat_ob[:, 6]
        sig_mn = sig_m - (ascat_ob[:, 6]-45)*a
        # daily average:
        tdoy = ascat_ob[:, 0]
        u_doy = np.unique(tdoy)
        sig_d, i0 = np.zeros([u_doy.size, 2]), 0
        inc_d = np.zeros(u_doy.size)
        for td in u_doy:
            sig_d[i0][0] = td
            sig_d[i0][1] = np.mean(sig_mn[tdoy == td])
            inc_d[i0] = np.mean(inc_m[tdoy == td])
            i0 += 1
        tx = sig_d[:, 0]
        sig_mn = sig_d[:, 1]

        # one more constraints, based on incidence angle
        # id_inc = bxy.gt_le(inc_d, 30, 35)
        # tx, sig_mn = tx[id_inc], sig_mn[id_inc]

        # edge detect
        sig_g = sigma_g  # gaussian stds
        g_size = 6*sig_g/2
        g_sig, ig2 = gauss_conv(sig_mn, sig=3, size=2*g_size+1)  # non-smoothed
        g_sig_valid = 2*(g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                    /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))-1
        max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, tx[g_size: -g_size])
        onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual')
        # site read
        doy, passhr = ascat_ob[:, 0] + 365, np.round(ascat_ob[:, 1])
        site_type = site_infos.get_type(si0)
        site_file = './copy0519/txt/'+site_type + si0 + '.txt'
        y2_empty = np.zeros(tx.size) + 1
        stats_sm, sm5 = read_site.read_sno(site_file, "Soil Moisture Percent -2in (pct)", si0)  # air tmp
        sm5_daily, sm5_date = cal_emi(sm5, y2_empty, doy, hrs=passhr)
        stats_t, t5 = read_site.read_sno(site_file, "Soil Temperature Observed -2in (degC)", si0)
        t5_daily, t5_date = cal_emi(t5, y2_empty, doy, hrs=passhr)
        if pp:
            stats_swe, swe = read_site.read_sno(site_file, "Precipitation Increment (mm)", si0)
        else:
            stats_swe, swe = read_site.read_sno(site_file, "snow", si0, field_no=-1)
        swe_daily, swe_date = cal_emi(swe, y2_empty, doy, hrs=passhr)
        sm5_daily[sm5_daily < -90], t5_daily[t5_daily < -90], \
        swe_daily[swe_daily < -90] = np.nan, np.nan, np.nan  # remove missing
        # print 'site no is %s' % si0
        print 'station ID is %s' % si0
        if inc_plot is True:
            fig = pylt.figure(figsize=[8, 3])
            ax = fig.add_subplot(111)
            x, y = u_doy, inc_d
            ax.plot(x, y, 'o')
            pylt.savefig('./result_07_01/inc_mid_'+si0+'.png')
            fig.clear()
            pylt.close()
            # ons_site = sm_onset(sm5_date-365, sm5_daily, t5_daily)
        # onset based on ascat
        # test_def.plt_npr_gaussian_ASC(['E(t)', tx[ig2][g_size: -g_size]+365, g_sig_valid],
        #                          # ['TB_v', t_v, var_npv],
        #                          ['Soil moisture (%)', sm5_date, sm5_daily],  # soil moisture
        #                          ['Soil temperature (DegC)', t5_date, t5_daily],
        #                          ['SWE (mm)', swe_date, swe_daily],
        #                          ['$\sigma^0$', tx+365, sig_mn],  # 511
        #                          fa_2=[], vline=onset,  # edge detection, !! t_v was changed
        #                          #ground_data=ons_site,
        #                          fe_2=[],
        #                          figname='./result_05_01/point_result/ascat/onset_based_on_ASCAT'+si0+'2.png',
        #                          mode='annual')
        return [tx[ig2][g_size: -g_size]+365, g_sig_valid], \
               [sm5_date, sm5_daily], \
               [t5_date, t5_daily], \
               [swe_date, swe_daily], \
               [tx+365, sig_mn], \
               onset

        # fig = pylt.figure(figsize=[4, 3])
        # ax = fig.add_subplot(111)
        # ax.plot(ascat_ob[:, 6], sig_m, 'ko')
        # pylt.savefig('./result_05_01/point_result/ascat/ascat_norm0_'+si0+'.png')
        # fig = pylt.figure(figsize=[4, 3])
        # ax = fig.add_subplot(111)
        # ax.plot(ascat_ob[:, 6], sig_mn, 'ko')
        # pylt.savefig('./result_05_01/point_result/ascat/ascat_norm1_'+si0+'.png')
        # print a, np.std(sig_m), np.std(sig_mn), np.mean(np.diff(sig_m)), np.mean(np.diff(sig_mn))


    # read site data
    # plot
    return 0


def ascat_alaska_onset(ob='AS', norm=False):
    if norm:
        isnorm = 'norm'
        folder_path = './result_05_01/ascat_resample_norms/ascat_resample_'+ob+'/'
    else:
        isnorm = 'orig'
        folder_path = './result_05_01/ascat_resample_'+ob+'/'
    print 'the sigma data is %s, %s' % (isnorm, ob)
    doy_range = np.arange(0, 365)
    d0 = datetime.date(2016, 1, 1)
    # initial a base grid
    base0 = np.load(folder_path+'ascat_20160101_resample.npy')
    sigma_3d = np.zeros([base0.shape[0], base0.shape[1], 365])
    inc_3d =  np.zeros([base0.shape[0], base0.shape[1], 365])
    onset_2d = np.zeros([base0.shape[0], base0.shape[1]])
    series_name = 'ascat_all_2016_'+isnorm+'_'+ob+'.npy'
    if not os.path.exists('./result_05_01/onset_result/ascat_all_2016_'+isnorm+'_'+ob+'.npy'):
        # find the daily sigma and mask files of AK
        for doy in doy_range:
            d_daily = d0 + timedelta(doy)
            d_str = d_daily.strftime('%Y%m%d')
            ak_file = bxy.find_by_date(d_str, folder_path)
            for f1 in ak_file:
                if f1.find('resample') > 0:
                    v = np.load(folder_path+f1)
                    sigma_3d[:, :, doy] = v
                if f1.find('incidence') > 0:
                    agl = np.load(folder_path+f1)
                    inc_3d[:, :, doy] = agl
        np.save('./result_05_01/onset_result/ascat_all_2016_'+isnorm+'_'+ob, sigma_3d)
        np.save('./result_05_01/onset_result/ascat_inc_2016_'+ob, inc_3d)
    else:
        sigma_3d = np.load('./result_05_01/onset_result/ascat_all_2016_'+isnorm+'_'+ob+'.npy')
        inc_3d = np.load('./result_05_01/onset_result/ascat_inc_2016_'+ob+'.npy')

    # build mask for land
    mask0 = sigma_3d[:, :, 0] != 0
    for m in range(0, 30):
        maski = sigma_3d[:, :, m] != 0
        mask0 = np.logical_or(mask0, maski)
    sigma_land = sigma_3d[mask0, :]
    np.save('./result_05_01/other_product/mask_ease2_125N', mask0)
    row_test, col_test = np.where(mask0)[0], np.where(mask0)[1]
    nodata_count = 0
    land_count = 0
    onset_1d = [[], []]
    for s1 in sigma_land:
        # edge detection
        g_size = 8
        i_sig_valid = (s1 != 0)
        doy_i = doy_range[i_sig_valid]
        sigma_i = s1[i_sig_valid]
        if sigma_i.size >= 120:
            g_sig, ig2 = gauss_conv(sigma_i, sig=3)  # non-smoothed
            g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                        /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
            max_gsig, min_gsig = peakdetect.peakdet(g_sig_valid, 1e-1, doy_i[g_size: -g_size])
            onset = find_inflect(max_gsig, min_gsig, typez='annual')
            onset_1d[0].append(onset[0])
            onset_1d[1].append(onset[1])
        else:
            onset_1d[0].append(np.array(0))
            onset_1d[1].append(np.array(0))
            nodata_count += 1
        if row_test[land_count] == 66:
            if col_test[land_count] == 229:
                print row_test[land_count], col_test[land_count], onset[0]
        land_count += 1
            # print 'no data count: %f' % nodata_count
    print '%d pixels have no valid data\n' % nodata_count
    onset_2d[mask0] = onset_1d[0]
    np.save('./result_05_01/onset_result/ascat_onset_0_2016_'+isnorm+'_'+ob, onset_2d)
    onset_2d[mask0] = onset_1d[1]
    np.save('./result_05_01/onset_result/ascat_onset_1_2016_'+isnorm+'_'+ob, onset_2d)

    # mask_1d = mask0.reshape(1, -1)
    # sigma_2d = sigma_3d.reshape(1, -1, 365)
    # i_land = np.where(mask_1d == True)
    # for l0 in i_land[1]:
    #     sig_land0 = sigma_2d[0, l0]
    pause = True


def smap_alaska_onset(mode='tbv', sig=3):
    ob = 'AS'
    folder_path = './result_05_01/smap_resample_'+ob+'/'
    base0 = np.load(folder_path+'smap_20160105_tbv_resample.npy')
    doy_range = np.arange(0, 365)
    d0 = datetime.date(2016, 1, 1)
    tbv_3d = np.zeros([base0.shape[0], base0.shape[1], 365])
    tbh_3d = np.zeros([base0.shape[0], base0.shape[1], 365])
    onset_2d = np.zeros([base0.shape[0], base0.shape[1]])
    if not os.path.exists('./result_05_01/onset_result/smap_all_2016'+'_'+mode+'_'+ob+'.npy'):
        # find the daily sigma and mask files of AK
        for doy in doy_range:
            d_daily = d0 + timedelta(doy)
            d_str = d_daily.strftime('%Y%m%d')
            ak_file = bxy.find_by_date(d_str, folder_path)
            for f1 in ak_file:
                if f1.find('v_resample') > 0:
                    v = np.load(folder_path+f1)
                    tbv_3d[:, :, doy] = v
                if f1.find('h_resample') > 0:  # other attributs
                    agl = np.load(folder_path+f1)
                    tbh_3d[:, :, doy] = agl
        np.save('./result_05_01/onset_result/smap_all_2016_tbv_'+ob, tbv_3d)
        np.save('./result_05_01/onset_result/smap_all_2016_tbh_'+ob, tbh_3d)
    else:
        tbv_3d = np.load('./result_05_01/onset_result/smap_all_2016_tbv_'+ob+'.npy')
        tbh_3d = np.load('./result_05_01/onset_result/smap_all_2016_tbh_'+ob+'.npy')
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
        g_size = 8
        if mode == 'npr':
            i_tbv_valid = (s1 != 0) & (s1 != -9999) & (s1 != 1) & (s1 != -1)
        else:
            i_tbv_valid = ((s1 != 0) & (s1 != -9999))
        doy_i = doy_range[i_tbv_valid]
        tbv_i = s1[i_tbv_valid]
        if tbv_i.size >= 120:
            g_sig, ig2 = gauss_conv(tbv_i, sig=sig, size=2*g_size+1)  # non-smoothed
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
    np.save('./result_05_01/onset_result/smap_onset_0_2016'+'_'+mode+'_'+ob, onset_2d)
    onset_2d[landid] = onset_1d[1]
    np.save('./result_05_01/onset_result/smap_onset_1_2016'+'_'+mode+'_'+ob, onset_2d)
    return 0


def ascat_onset_map(ob, odd_point=[], product='ascat', mask=False):
    anc_direct = './result_05_01/other_product/'
    if product == 'ascat':
        mode = ['_orig_', '_norm_']
        prefix = './result_05_01/onset_result/ascat_onset_'
        lons_grid, lats_grid = np.load('./result_05_01/onset_result/lon_ease_grid.npy'), \
                                np.load('./result_05_01/onset_result/lat_ease_grid.npy')
        if odd_point[0] is not list:
            dis_odd = bxy.cal_dis(odd_point[1], odd_point[0], lats_grid.ravel(), lons_grid.ravel())
            index = np.argmin(dis_odd)
            row = int(index/lons_grid.shape[1])
            col = index - (index/lons_grid.shape[1]*lons_grid.shape[1])
            print row, col, lons_grid[row, col], lats_grid[row, col]
        for key in ob:
            for m in mode:
                onset_0_file = prefix+'0'+'_2016'+m+key+'.npy'
                onset0 = np.load(onset_0_file)
                onset_1_file = prefix+'1'+'_2016'+m+key+'.npy'
                onset1 = np.load(onset_1_file)
                if mask is True:
                    mask0 = np.load(anc_direct+'snow_mask_125s.npy')
                    onset0 = np.ma.masked_array(onset0, mask=[mask0==0])
                    onset1 = np.ma.masked_array(onset1, mask=[mask0==0])
                pass_zone_plot(lons_grid, lats_grid, onset0, './result_05_01/onset_result/', fname='onset_0'+m+key,
                               z_max=180, z_min=50, odd_points=odd_point)
                pass_zone_plot(lons_grid, lats_grid, onset1, './result_05_01/onset_result/', fname='onset_1'+m+key,
                               z_max=360, z_min=240, odd_points=odd_point)
    else:
        lons_grid = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
        lats_grid = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
        indicators = ['npr', 'tb']
        for ind in indicators:
            onset0 = np.load('./result_05_01/onset_result/smap_onset_0_2016_'+ind+'_AS.npy')
            onset1 = np.load('./result_05_01/onset_result/smap_onset_1_2016_'+ind+'_AS.npy')
            if mask is True:
                mask0 = np.load(anc_direct+'snow_mask_360s.npy')
                onset0 = np.ma.masked_array(onset0, mask=[mask0==0])
                onset1 = np.ma.masked_array(onset1, mask=[mask0==0])
            pass_zone_plot(lons_grid, lats_grid, onset0, './result_05_01/onset_result/', fname='onset_0_smap_'+ind,
                                   z_max=180, z_min=50, odd_points=odd_point)
            pass_zone_plot(lons_grid, lats_grid, onset1, './result_05_01/onset_result/', fname='onset_1_smap_'+ind,
                                   z_max=360, z_min=250, odd_points=odd_point)


def ascat_result_test(area, key='AS', mode='_norm_', odd_rc=([], []), ft='0'):
    lons_grid, lats_grid = np.load('./result_05_01/onset_result/lon_ease_grid.npy'), \
                            np.load('./result_05_01/onset_result/lat_ease_grid.npy')
    arctic = lats_grid > 66.5
    # check the odd value
    prefix = './result_05_01/onset_result/ascat_onset_'
    onset_0_file = np.load(prefix+ft+'_2016'+mode+key+'.npy')
    if area == 'area_0':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (onset_0_file < 60) & (lons_grid < -155) & (lons_grid > -160)
    elif area == 'area_1':
        odd_thaw = (onset_0_file > 0) & (lats_grid < 60) & (onset_0_file < 80) & (lons_grid < -160)
    elif area == 'area_2':
        odd_thaw = (lats_grid > 55) & (lats_grid < 60) & (lons_grid < -155) & (lons_grid > -160) & (onset_0_file < 75) & (onset_0_file > 0)
    elif area == 'area_3':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid < -150) & (lons_grid > -155) & (onset_0_file < 80) & (onset_0_file > 0)
    elif area == 'area_4':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (lons_grid > -150) & (lons_grid < -145) & (onset_0_file < 100)
    elif area == 'area_5':
        odd_thaw = (lats_grid > 70) & (lats_grid < 80) & (lons_grid > -155) & (lons_grid < -150) & (onset_0_file > 130)
    elif area == 'area_6':
        odd_thaw = (lats_grid > 65) & (lats_grid < 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_0_file < 120) & (onset_0_file > 0)
    elif area == 'area_7':
        odd_thaw = (lats_grid > 67) & (lats_grid < 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_0_file > 135) & (onset_0_file > 0)
    elif area == 'area_8':
        odd_thaw = (lats_grid > 70) & (lons_grid > -160) & (lons_grid < -155) & (onset_0_file > 120) & (onset_0_file > 0)
    elif area == 'area_88':
        odd_thaw = (lats_grid > 70) & (lons_grid > -155) & (lons_grid < -150) & (onset_0_file > 140) & (onset_0_file > 0)
    elif area == 'area_9':
        odd_thaw = (lats_grid > 67) & (lats_grid < 70) & (lons_grid > -150) & (lons_grid < -145) & (onset_0_file > 150) & (onset_0_file < 179)
    elif area == 'area_10':
        odd_thaw = (lats_grid > 68) & (lats_grid < 70) & (lons_grid > -145) & (lons_grid < -140) & (onset_0_file < 140) & (onset_0_file > 0)
    elif area == 'area_11':
        odd_thaw = (lats_grid > 68) & (lats_grid < 70) & (lons_grid > -145) & (lons_grid < -140) & (onset_0_file < 140) & (onset_0_file > 0)
    id_thaw = np.where(odd_thaw)
    odd_row, odd_col = id_thaw[0], id_thaw[1]
    sigma_3d = np.load('./result_05_01/onset_result/ascat_all_2016'+mode+key+'.npy')
    inc_3d = np.load('./result_05_01/onset_result/ascat_inc_2016_'+key+'.npy')
    odd_sigma = sigma_3d[odd_thaw]
    odd_inc = inc_3d[odd_thaw]
    odd_lon, odd_lat = lons_grid[odd_thaw], lats_grid[odd_thaw]
    id_odd = np.arange(0, odd_lon.size)
    odd_value = onset_0_file[odd_thaw]
    np.save('./result_05_01/onset_result/odd_thaw_'+area, odd_sigma)
    np.save('./result_05_01/onset_result/odd_thaw_inc'+area, odd_inc)
    np.savetxt('./result_05_01/onset_result/odd_thaw_'+area+'.txt',
               np.array([id_odd.T, odd_value.T, odd_lon.T, odd_lat.T, odd_row.T, odd_col.T]).T, fmt='%d, %d, %.8f, %.8f, %d, %d')

    # test particular pixels, e.g., median is np.argsort(odd_value)[odd_value.size//2]
    # p_id = np.argsort(odd_value)[odd_value.size//2]
    p_id = 37
    ascat_test_odd_point_plot('./result_05_01/onset_result/odd_thaw_'+area+'.npy',
                              './result_05_01/onset_result/odd_thaw_'+area+'.txt',
                              p_id, area=area, orb=key,
                              odd_point=[sigma_3d[odd_rc], lons_grid[odd_rc], lats_grid[odd_rc], odd_rc[0], odd_rc[1]],
                              mode=mode)
    # test angular dependency
    # id_angular = odd_sigma[p_id] != 0
    # y_sig = odd_sigma[p_id][id_angular]
    # x_inc = odd_inc[p_id][id_angular]
    # ascat_test_odd_angular(x_inc, y_sig, area=area)
    return 0


def smap_result_test(area, key='AS', odd_rc=([], []), ft='0', mode='tb'):
    lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy'), \
                            np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
    # check the odd value
    prefix = './result_05_01/onset_result/smap_onset_'
    # smap_onset_0_2016_tb_AS.npy
    onset_0_file = np.load(prefix+ft+'_2016_'+mode+'_'+key+'.npy')
    if area == 'area_0':
        odd_thaw = (lats_grid > 60) & (lats_grid < 65) & (onset_0_file < 60) & (lons_grid < -155) & (lons_grid > -160)
    elif area == 'area_1':
        odd_thaw = (onset_0_file > 0) & (lats_grid < 60) & (onset_0_file < 80) & (lons_grid < -160)
    elif area == 'area_2':
        odd_thaw = (lats_grid > 55) & (lats_grid < 60) & (lons_grid < -155) & (lons_grid > -160) & (onset_0_file < 75) & (onset_0_file > 0)
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
    sigma_3d2 = np.load('./result_05_01/onset_result/smap_all_2016_tbh_'+key+'.npy')
    sigma_3d1 = np.load('./result_05_01/onset_result/smap_all_2016_tbv_'+key+'.npy') ####
    if mode == 'tb':
        sigma_3d = sigma_3d1
    else:
        sigma_3d = (sigma_3d1 - sigma_3d2)/(sigma_3d1+sigma_3d2)
    odd_sigma = sigma_3d[odd_thaw]
    odd_lon, odd_lat = lons_grid[odd_thaw], lats_grid[odd_thaw]
    id_odd = np.arange(0, odd_lon.size)
    odd_value = onset_0_file[odd_thaw]
    np.save('./result_05_01/onset_result/odd_thaw_smap_'+area, odd_sigma)
    np.savetxt('./result_05_01/onset_result/odd_thaw_smap_'+area+'.txt',
               np.array([id_odd.T, odd_value.T, odd_lon.T, odd_lat.T, odd_row.T, odd_col.T]).T, fmt='%d, %d, %.8f, %.8f, %d, %d')
    p_id = 37
    sig_file = './result_05_01/onset_result/odd_thaw_smap_'+area+'.npy'
    area_file = './result_05_01/onset_result/odd_thaw_smap_'+area+'.txt'
    orb=key
    odd_point=[sigma_3d[odd_rc], lons_grid[odd_rc], lats_grid[odd_rc], odd_rc[0], odd_rc[1]]
    prefix = './result_05_01/onset_result/odd_series_smap/'
    odd_info = np.loadtxt(area_file, delimiter=',')
    sig = np.load(sig_file)
    # odd_onset = odd_info[:, 1]
    # id of testing pixel
    # p_id = np.argsort(odd_onset)[odd_onset.size//2]
    # p_id = 37
    if len(odd_point) > 0:
        v, loni, lati, rowi, coli = odd_point[0], odd_point[1], odd_point[2], odd_point[3], odd_point[4]
    else:
        v, loni, lati, rowi, coli = sig[p_id], odd_info[:, 2][p_id], odd_info[:, 3][p_id], \
                                odd_info[:, 4][p_id], odd_info[:, 5][p_id]

    print 'the odd pixel is (%.5f, %.5f)' % (loni, lati), '\n'
    if mode == 'tb':
        for pol in ['V', 'H']:
            if pol == 'H':
                v = sigma_3d2[odd_rc]
            iv_valid = (v > 100)
            v_valid = v[iv_valid]
            t = np.arange(0, v.size)
            t_valid = t[iv_valid]
            g_size = 8
            g_sig, ig2 = gauss_conv(v_valid, sig=4, size=2*g_size+1)
            g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                            /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
            max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, t_valid[g_size: -g_size])
            onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=mode)
            # test the temporal variation
            print 'the temporal change of tb%s is: ' % pol, v_valid[g_size: -g_size]
            fig = pylt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d, Days: %d' % (loni, lati, rowi, coli, onset[0], onset[1], sum(v<0)))
            plot_funcs.pltyy(t_valid, v_valid, prefix+area+orb+'_'+pol, '$T_{B%s}$ (K)' % pol,
                             t2=t_valid[ig2][g_size: -g_size], s2=g_sig[g_size: -g_size],
                             ylim=[200, 280], symbol=['bo', 'g-'], handle=[fig, ax])
    else:
        iv_valid = (v != 0) & (v != 1) & (v != -1) & (v != -0) & (v != -9999) & (~np.isnan(v))
        # iv_valid = ((v != 0) & (v != -9999))
        # iv_valid = (v != 0) & (v > -999)
        v_valid = v[iv_valid]
        t = np.arange(0, v.size)
        t_valid = t[iv_valid]
        g_size = 8
        g_sig, ig2 = gauss_conv(v_valid, sig=4, size=2*g_size+1)  # non-smoothed
        g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
        max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig[g_size: -g_size], 1e-4, t_valid[g_size: -g_size])
        onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=mode)
        print 'onset in odd_point test:', onset
        # the temporal change of npr/tb
        id_test = int(max_gsig_s[max_gsig_s[:, 1] == onset[0]][0, 0])
        # test the temporal variation
        print 'the temporal change of npr is: ', v_valid[g_size: -g_size][id_test-1: id_test+7]
        fig = pylt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d' % (loni, lati, rowi, coli, onset[0], onset[1]))
        plot_funcs.pltyy(t_valid, v_valid, prefix+area+orb, '$NPR$ (%)',
                         t2=t_valid[ig2][g_size: -g_size], s2=g_sig[g_size: -g_size],
                         ylim=[0, 0.1], symbol=['bo', 'g-'], handle=[fig, ax])


def ascat_test_odd_point_plot(sig_file, area_file, p_id, area='odd_arctic1', odd_point=[], orb='AS', mode=[], ptype='sig'):
    prefix = './result_05_01/onset_result/odd_series_smap/'
    odd_info = np.loadtxt(area_file, delimiter=',')
    sig = np.load(sig_file)
    # odd_onset = odd_info[:, 1]
    # id of testing pixel
    # p_id = np.argsort(odd_onset)[odd_onset.size//2]
    # p_id = 37
    if len(odd_point) > 0:
        v, loni, lati, rowi, coli = odd_point[0], odd_point[1], odd_point[2], odd_point[3], odd_point[4]
    else:
        v, loni, lati, rowi, coli = sig[p_id], odd_info[:, 2][p_id], odd_info[:, 3][p_id], \
                                odd_info[:, 4][p_id], odd_info[:, 5][p_id]
    print 'the odd pixel is (%.5f, %.5f)' % (loni, lati), '\n'
    iv_valid = v != 0
    v_valid = v[iv_valid]
    t = np.arange(0, v.size)
    t_valid = t[iv_valid]
    g_size = 8
    g_sig, ig2 = gauss_conv(v_valid, sig=3)  # non-smoothed
    g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
    max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, t_valid[g_size: -g_size])
    onset = find_inflect(max_gsig_s, min_gsig_s, typez='annual', typen=ptype)
    fig = pylt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title('Loc: %.4f, %.4f, %d, %d \n T: %d F: %d, Days: %d' % (loni, lati, rowi, coli, onset[0], onset[1], sum(v<0)))
    print prefix+area+orb
    plot_funcs.pltyy(t_valid, v_valid, prefix+area+orb, '$\sigma_0$ (dB)',
                     t2=t_valid[ig2][g_size: -g_size], s2=g_sig_valid,
                     ylim=[-20, -4], symbol=['bo', 'g-'], handle=[fig, ax])

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

