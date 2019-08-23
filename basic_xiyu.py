import numpy as np
import h5py
from datetime import datetime
from datetime import timedelta
import os
import fnmatch
import re
import pytz
import scipy.signal as signal
import scipy.optimize as opt
import glob

def opt_test(fminfunc, x, h, y, yn, x0):
    return 0



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


def cal_dis_v2(lat0, lon0, lats, lons):

    lamda0 = (lon0/180.0)*np.pi
    lamdas = (lons/180.0)*np.pi
    phi0 = (lat0/180.0)*np.pi
    phis = (lats/180.0)*np.pi
    x = (lamdas-lamda0.reshape(lamda0.size, -1)) * np.cos((phis+phi0.reshape(phi0.size, -1))/2)
    y = phis - phi0.reshape(phi0.size, -1)
    return 6371*np.sqrt(x**2 + y**2)


def dict2npz(fname0, dict0, keys):
    np.savez(fname0, **{dict0[key0] for key0 in keys})

def check_h5(filename):
    hf = h5py.File(filename)
    n_key = hf.keys()[2]
    print hf.keys(), '\n', n_key
    for key in hf[hf.keys()[2]].keys():
        print key
    print hf[n_key]['cell_row'][-1]
    hf.close()


def deci2bin(d_num):
    # return an string with each element represent a bit, from 0 to n
    np.binary_repr(d_num)


def dis_inter(dis, value):
    if type(dis) is int:
        if dis == 0:
            dis += 1
    weighs = 1/(dis)**2
    if type(value) is not np.ndarray:
        if value == -9999:
            weighs = 0
            print 'no valid weighs'
    else:
        weighs[value==-9999] = 0
    return np.sum(weighs*value)/np.sum(weighs)


def get_doy(date_string, year0=2015):
    """
    :param
        date_string: in the form of 'yyyymmdd'
    :return:
        doy_num: the num of doy
    """
    doy_num_list = []
    for strz in date_string:
        t = datetime.strptime(strz, '%Y%m%d').timetuple()
        doy_num_list.append(t.tm_yday + (t.tm_year - year0) * 365)
    doy_num_np = np.array(doy_num_list)
    return doy_num_np


def get_doy_v2(date_string):
    """
    return the doy num, a string in forms of 'yyyymmdd' should be given
    :param
        date_string: in the form of 'yyyymmdd'
    :return:
        doy_num: the num of doy
    """
    doy_num_list = []
    year_no = []
    for strz in date_string:
        t = datetime.strptime(strz, '%Y%m%d').timetuple()
        doy_num_list.append(t.tm_yday)
        year_no.append(t.tm_year)
    doy_num_np = np.array(doy_num_list)
    return doy_num_np, year_no


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


def bitand(bin_array, bits):
    and_result = np.zeros(bin_array.size)
    i = 0
    for bin in bin_array:
        and_result[i] = np.bitwise_and(bin, bits)
        i += 1
    return and_result


def timetransform(time_utc, reftime, fms, tzone=False):
    """
    Get AK local time in seconds from 2015/01/01.
    :param time_utc:
    :param reftime:
    :param fms:
    :param tzone:
    :return:
    """
    sec_2015 = datetime.strptime('20150101 00:00:00', '%Y%m%d %X') \
                       - datetime.strptime(reftime, fms)
    sec_20160312 = datetime.strptime('20160313 10:59:00', '%Y%m%d %X') \
                   - datetime.strptime(reftime, fms)
    sec_20161105 = datetime.strptime('20161106 09:59:00', '%Y%m%d %X') \
                   - datetime.strptime(reftime, fms)
    sec_ascat = time_utc-sec_2015.total_seconds()
    # transform to local time
    if tzone:
        pause = 0
        sec_ascat[(sec_ascat>sec_20160312.total_seconds()) & (sec_ascat<sec_20161105.total_seconds())] -= 8*3600
        sec_ascat[sec_ascat<sec_20160312.total_seconds()] -= 9*3600
        sec_ascat[sec_ascat>sec_20161105.total_seconds()] -= 9*3600
    return sec_ascat


def time_transform_check(time_utc, reftime, fms, tzone=False):
    ref_obj = datetime.strptime(reftime, fms)
    sec_2015 = datetime.strptime('20150101 00:00:00', '%Y%m%d %X') \
                       - datetime.strptime(reftime, fms)
    sec_20160312 = datetime.strptime('20160313 10:59:00', '%Y%m%d %X') \
                   - datetime.strptime(reftime, fms)
    sec_20161105 = datetime.strptime('20161106 09:59:00', '%Y%m%d %X') \
                   - datetime.strptime(reftime, fms)
    sec_ascat = time_utc-sec_2015.total_seconds()
    # transform to local time
    if tzone:
        sec_ascat[(sec_ascat>sec_20160312.total_seconds()) & (sec_ascat<sec_20161105.total_seconds())] -= 8*3600
        sec_ascat[sec_ascat<sec_20160312.total_seconds()] -= 9*3600
        sec_ascat[sec_ascat>sec_20161105.total_seconds()] -= 9*3600
    return sec_ascat


def time_getlocaltime(utc_sec, ref_time=[2000, 1, 1, 12], t_source='utc', t_out='US/Alaska'):  # default ob: asc
    tz_utc = pytz.timezone(t_source)
    tz_ak = pytz.timezone(t_out)
    passtime_obj_list = [datetime(ref_time[0], ref_time[1], ref_time[2], ref_time[3], 0,
                                  tzinfo=tz_utc)+timedelta(seconds=sec_i) for sec_i in utc_sec]
    doy_passhr = np.array([[p_time0.astimezone(tz=tz_ak).timetuple().tm_year,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_mon,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_mday,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_yday,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_hour]
                           for p_time0 in passtime_obj_list]).T
    return doy_passhr  # year, month, day, doy, hour


def time_getlocaltime_v2(utc_sec, ref_time=[2000, 1, 1, 12], t_source='utc', t_out='US/Alaska'):  # default ob: asc
    """
    the nan values in secs series are ignored
    :param utc_sec:
    :param ref_time:
    :param t_source:
    :param t_out:
    :return:
    """
    tz_utc = pytz.timezone(t_source)
    tz_ak = pytz.timezone(t_out)
    nan_id = np.isnan(utc_sec)
    utc_sec[nan_id] = get_total_sec('19991231')
    passtime_obj_list = [datetime(ref_time[0], ref_time[1], ref_time[2], ref_time[3], 0,
                                  tzinfo=tz_utc)+timedelta(seconds=sec_i) for sec_i in utc_sec]
    doy_passhr = np.array([[p_time0.astimezone(tz=tz_ak).timetuple().tm_year,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_mon,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_mday,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_yday,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_hour]
                           for p_time0 in passtime_obj_list]).T
    return doy_passhr  # year, month, day, doy, hour


def get_secs(t_list, reftime=[2000, 1, 1, 0]):
    return (datetime(t_list[0], t_list[1], t_list[2], t_list[3], t_list[4], t_list[5]) -
            datetime(reftime[0], reftime[1], reftime[2], reftime[3], 0, 0)).total_seconds()


def get_total_sec(date_str, fmt='%Y%m%d', reftime=[2000, 1, 1, 0]):
    return (datetime.strptime(date_str, fmt) -
            datetime(reftime[0], reftime[1], reftime[2], reftime[3], 0, 0)).total_seconds()


def zone_trans(secs, zone0, zone1, ref_time=[2000, 1, 1, 0]):
    tz0 = pytz.timezone(zone0)
    tz1 = pytz.timezone(zone1)
    passtime0 = datetime(ref_time[0], ref_time[1], ref_time[2], ref_time[3], 0, tzinfo=tz0) + timedelta(seconds=secs)
    passtime1 = passtime0.astimezone(tz=tz1)
    time1_tuple = passtime1.timetuple()
    return time1_tuple


def coordinate_2_index(array0, origin_shape):
    return array0[0]*origin_shape[0]+array0[1]


def get_time_now():
    return datetime.now()


def slice4plot(var, range):
    """
    slice var with range
    :param var:
    :param range:
    :return:
    """
    return 0


def integ_left(a, b, f, nbins=10):
    h = float(b-a)/10  # dx ?
    sum = 0.0
    for n in range(nbins):
        sum += sum + h*(f[0+n*h])


def integ_10(b, f, nbins=10):
    dx = 1
    sum0=0.0
    for n in range(nbins):
        sum0 += sum0 + dx*(f[b-n*dx])
    return sum0


def h5_writein(fname, layers, narrayz):
    h0 = h5py.File(fname, 'a')
    # layerlist = split_strs(layers, '/')  # number of layers
    # if len(layerlist) > 1:
    #     for key0 in layerlist:
    #         if key0 in h0.keys():
    #             status0 = 'grp: %s existed' % key0
    #             del layerlist[layerlist.index(key0)]
    #             for key1 in layerlist:
    #                 if key1 in h0[key0].keys():
    #                     status1 = 'grp: %s existed' % (key0+'/'+key1)
    #                     h0.close()
    #                     return 0
    h0.create_dataset(layers, data=narrayz)
    h0.close()
    return 0


def split_strs(str_var, symbol):
    p_underline = re.compile(symbol)
    # .append(p_underline.split(tbs)[3])
    return p_underline.split(str_var)


def trim_mean(input_mat):
    output_arr = np.zeros(input_mat.shape[0])
    i = -1
    for mat0 in input_mat:
        i+=1
        # remove nan value
        mat0 = mat0[~np.isnan(mat0)]
        # cal the trimmed average
        mat1 = np.sort(mat0)
        output_arr[i] = np.mean(mat1[1: -1])
    return output_arr


def ll2easegrid(lon, lat):
    x = 0
    y = 0

    return x, y


def gt_nstd(conv_250_350, series, n):
    if conv_250_350.shape[0] > 1:
        conv_value = conv_250_350[:, 2]
        conv_mean = np.mean(series)
        conv_std = np.std(series)
        i1 = np.abs(conv_value) > conv_mean + n*conv_std
    else:
        i1 = 0
    return i1


def check_nan(series):
    return series[0][np.isnan(series[1])]


def im_permitivity(a, b):  # rel and im part
    im_z = np.sqrt(np.sqrt(a**2+b**2)) * (b/np.sqrt((a+np.sqrt(a**2+b**2))**2+b**2))
    return im_z


def los_factor(a, b, omega):
    return 2*np.pi*omega*np.sqrt(1.0*a/2*(np.sqrt(1+(b/a)**2)-1))


def doy2date(doy, year0=2016, fmt="%m%d"):
    d_obj = datetime(year0, 1, 1) + timedelta(doy-1)
    return d_obj.strftime(fmt)


def latlon2index(p_coord, resolution=12.5):
    """
    turn latitude longitude into 1d and 2d indices
    :param p_coord: ndarray, dimension: pixels X coordinates, e.g. [[lon0, lat0], [lon1, lat1], ...]
    :param resolution:
    :return:
    """
    if resolution == 36:
        h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
        h0 = h5py.File(h5_name)
        lons_grid = h0['cell_lon'].value
        lats_grid = h0['cell_lat'].value
    else:
        lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                    np.load('./result_05_01/other_product/lat_ease_grid.npy')
    # turn p_coord to index
    pixel_id = (np.zeros(p_coord.shape[0]).astype(int), np.zeros(p_coord.shape[0]).astype(int))
    p_id_1d = np.zeros(p_coord.shape[0]).astype(int)
    i_pid = 0
    for coord0 in p_coord:
        dis = cal_dis(coord0[1], coord0[0], lats_grid, lons_grid)
        # dis.argmin() to smap 1d index
        p_id_1d[i_pid] = dis.argmin()
        i2d = np.unravel_index(dis.argmin(), dis.shape)
        pixel_id[0][i_pid] = i2d[0]
        pixel_id[1][i_pid] = i2d[1]
        i_pid += 1
    return pixel_id, p_id_1d


def index_match(all_index, target_index):
    return [np.where(all_index==i0)[0] for i0 in target_index]


def get_doy_array(st, en, fmt):
    st_sec = get_total_sec(st, fmt=fmt)
    en_sec = get_total_sec(en, fmt=fmt)
    st_obj = datetime.strptime(st, fmt)
    en_obj = datetime.strptime(en, fmt)
    st_doy, en_doy = st_obj.timetuple().tm_yday, en_obj.timetuple().tm_yday
    year0 = st_obj.timetuple().tm_year
    doy_arry = np.arange(st_doy, en_doy+1)
    return doy_arry, year0


def current_time():
    return datetime.now().timetuple()


def test_read_txt(delimiter=','):
    with open('region_sigma_air.txt') as file0:
        for row0 in file0:
            print row0[0], row0[1], row0[2]
            atts = row0.split(delimiter)
            print atts[0], atts[1]
            break


def get_head_cols(fname, headers=[]):
    if len(headers) == 0:
        print 'input headers required'
    else:
        site_tb = np.loadtxt(fname)
        with open(fname, 'rb') as as0:

            for row in as0:
                row0 = re.split(', |,|\n', row[2:])
                col_num = []
                for head0 in headers:
                    col_num.append(row0.index(head0))
                    # 'cell_tb_time_seconds_aft', 'cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_qual_flag_v_aft',
                    # 'cell_tb_qual_flag_h_aft', 'cell_tb_error_v_aft', 'cell_tb_error_h_aft', 'cell_lon', 'cell_lat'
                return col_num


def time_normalize(time_array):
    return time_array


def trans_in2d(idx_1d, shape):
    # change 1d index to 2d coordinate (column/row)
    row_num = idx_1d/shape[1]
    col_num = idx_1d - idx_1d/shape[1] * shape[1]
    idx_2d = np.array([row_num, col_num])
    return idx_2d


def trans_doy_str(doy_array,  y=2015, form="%Y%m%d"):
    doy_str = []
    for doy0 in doy_array:
        doy_str0 = doy2date(doy0, year0=y, fmt=form)
        doy_str.append(doy_str0)
    return doy_str


def odd_out(out_name, out_value, nodata_id=0):
    """
    save ood value record on an ascii file
    :param out_name:
    :param out_value:
    :param nodata_id:
    :return:
    """
    if type(out_value) == str:
        lines = 'odd value is %s' % out_value
    else:
        lines = 'odd value is %.1f' % out_value
    with open(out_name, 'a') as writer0:
        if nodata_id == 0:
            time0 = datetime.now().timetuple()
            time_str = '%d-%d, %d: %d \n' % (time0.tm_mon, time0.tm_mday, time0.tm_hour, time0.tm_min)
            writer0.write(time_str)
            writer0.write(lines)
            writer0.write('\n')
            nodata_id += 1
        else:
            writer0.write(lines)
            writer0.write('\n')


def geo_2_row(grid, target):
    dis = cal_dis(target[1], target[0], grid[1].ravel(), grid[0].ravel())
    idx_1d = np.argmin(dis)
    rc = trans_in2d(idx_1d, grid[0].shape)
    return rc


def hampel(vals_orig, k=7, t0=3):
    vals = vals_orig.copy()
    L = 1.4826
    rolling_median = np.roll(vals, k).median()
    difference = np.abs(rolling_median-vals)
    median_abs_deviation = np.roll(difference, k).median()


def reject_outliers(data, m=3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s<m


def time_compare(x0, x1):
    signal.resample()


def nan_set(input0, mask_var):
    input0[input0 == mask_var] = np.nan
    return input0


def normal_series(series0, n=0):
            # g_npr_valid_n = 2*(g_npr[g_size: -g_size] - np.nanmin(g_npr[g_size: -g_size]))\
            #             /(np.nanmax(g_npr[g_size: -g_size]) - np.nanmin(g_npr[g_size: -g_size])) - 1  # normalized
    max0, min0 = np.nanmax(series0), np.nanmin(series0)
    if n == 1:
        return 2*(series0 - min0) / (max0 - min0) - 1
    elif n == 0:
        return (series0 - min0) / (max0 - min0)


def get_statics(array):
    """
    :return: mean, max and min
    """
    return np.nanmean(array), np.nanmax(array), np.nanmin(array)

def smap_download_file_list(st, ed, year=2017, m=3, d=18):
    delta_d = 1
    str_list = trans_doy_str(np.arange(st, ed+1), y=year, form='%Y.%m.%d')  # 77, 214, 2017
    with open('smap_folders.txt', 'w') as f0:
        for str0 in str_list:
            f0.write('%s\n' % str0)
    return 0


def reshape_element0(list, d0=1):
    for element0 in list:
        element0.shape = d0, -1


def remove_unvalid_time(t0, value0, t_str='20151231'):
    reference_sec = get_total_sec(t_str)
    valid = t0 > reference_sec
    return t0[valid], value0[valid]


def ind2latlon(points_index, resolution=12.5):
    if resolution == 12.5:
        lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                            np.load('./result_05_01/other_product/lat_ease_grid.npy')
        mask = np.load('/home/xiyu/PycharmProjects/R3/result_05_01/other_product/mask_ease2_125N.npy')
        p_index_sensor = points_index[1]  # ascat index
    elif resolution == 36:
        h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102'
        h0 = h5py.File(h5_name)
        lons_grid = h0['cell_lon'].value
        lats_grid = h0['cell_lat'].value
        mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
        p_index_sensor = points_index[0]  # smap index
    return lons_grid, lats_grid, p_index_sensor


def sort_byline():
    return 0


def get_yearly_files(t_window=[0, 210], year0=2016):
    # ascat_att0=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore',
    #                               'inc_angle_trip_fore', 'sigma0_trip_mid', 'inc_angle_trip_mid']
    # path0='./result_08_01/series/ascat'
    doy_array = np.arange(t_window[0], t_window[1])
    file_path='ascat_resample_all3'
    path_ascat = []
    for doy0 in doy_array:
        time_str0 = doy2date(doy0, fmt='%Y%m%d', year0=year0)
        match_name = 'result_08_01/%s/ascat_*%s*.h5' % (file_path, time_str0)
        path_ascat += glob.glob(match_name)
    return path_ascat