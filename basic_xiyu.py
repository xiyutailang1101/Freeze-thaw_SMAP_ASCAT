import numpy as np
import h5py
from datetime import datetime
from datetime import timedelta
import os
import fnmatch
import re
import pytz

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


def time_getlocaltime(utc_sec, ref_time=[2000, 1, 1, 12], t_source = 'utc', t_out = 'US/Alaska'):  # default ob: asc
    tz_utc = pytz.timezone(t_source)
    tz_ak = pytz.timezone(t_out)
    passtime_obj_list = [datetime(ref_time[0], ref_time[1], ref_time[2], ref_time[3], 0, tzinfo=tz_utc)+timedelta(seconds=sec_i) for sec_i in utc_sec]
    doy_passhr = np.array([[p_time0.astimezone(tz=tz_ak).timetuple().tm_year,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_mon,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_mday,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_yday,
                            p_time0.astimezone(tz=tz_ak).timetuple().tm_hour]
                           for p_time0 in passtime_obj_list]).T
    return doy_passhr  # year, month, day, doy, hour

def get_secs(t_list, reftime=[2000, 1, 1, 0]):
    return (datetime(t_list[0], t_list[1], t_list[2], t_list[3], t_list[4], t_list[5])
            - datetime(reftime[0], reftime[1], reftime[2], reftime[3], 0, 0)).total_seconds()

def zone_trans(secs, zone0, zone1, ref_time=[2000, 1, 1, 0]):
    tz0 = pytz.timezone(zone0)
    tz1 = pytz.timezone(zone1)
    passtime0 = datetime(ref_time[0], ref_time[1], ref_time[2], ref_time[3], 0, tzinfo=tz0) + timedelta(seconds=secs)
    passtime1 = passtime0.astimezone(tz=tz1)
    time1_tuple = passtime1.timetuple()
    return time1_tuple


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
    row_num = idx_1d/shape[1]
    col_num = idx_1d - idx_1d/shape[1] * shape[1]
    idx_2d = np.array([row_num, col_num])
    return idx_2d


def trans_doy_str(doy_array):
    doy_str = []
    for doy0 in doy_array:
        doy_str0 = doy2date(doy0, year0=2015, fmt="%Y%m%d")
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
    d = np.abs(data - np.mean(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s<m



