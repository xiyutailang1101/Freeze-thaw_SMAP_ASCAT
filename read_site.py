"""
This package is for reading single data in or regional data around a site.
"""
import numpy as np
import matplotlib.pyplot as plt
import re, sys
import os
import h5py
import log_write, site_infos
import csv
import datetime
import basic_xiyu as bs
import plot_funcs
import matplotlib.pyplot as plt
import data_process
import basic_xiyu as bxy
abvalue = []


def search_ob_file(date, file_path):
    """
    Search file of a site by date
    date:
        date
    file_path:
        path contains h5 files
    Return:
        A list contains the ascend and descend orbit file
    """
    file_name = []
    for data_a in os.listdir(file_path):
        if re.search(date, data_a):  # data of the date
            file_name.append(file_path + data_a)
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
    """
    read the attribute (dataset) specified
    :param file_name:
    :param site_info:
    :param attribute: [group name, att list]
    :return:
    """

    hf_a = h5py.File(file_name, 'r')
    if attribute[0] not in hf_a:
        hf_a.close()
        stat = -1
        print '%s was not included in \n %s' % (attribute[0], file_name)
        return -9999, -9999, stat
    else:
        value_dict = dict()
        lat = np.array(hf_a[attribute[0]+'/cell_lat'])
        lon = np.array(hf_a[attribute[0]+'/cell_lon'])
        dis = bs.cal_dis(site_info[1], site_info[2], lat, lon)
        min_ind = np.where(dis<25.5)
        if min_ind[0].size < 1:  # no near neighbor
            return -9999, -9999, -1
        else:
            for atti in attribute[1]:
                value = np.array(hf_a[attribute[0]+'/'+atti].value[min_ind])
                value_dict[atti] = value
            value_dis = np.array(dis[min_ind])
            hf_a.close()
            stat = 1
            return value_dict, value_dis, stat



def read_region(file_name, attribute, site_no, disref=0.5):
    """
    Read all the data of an attribute of h5
    :return
        value, flag, lat, lon
    """
    hf_a = h5py.File(file_name, 'r')
    if attribute[0] not in hf_a:
        print 'No sigma data in this site and date'
        print file_name
        hf_a.close()
        return None, -99, -99, -99
    else:
        lat = np.array(hf_a[attribute[1]])
        lon = np.array(hf_a[attribute[2]])
        value = np.array(hf_a[attribute[0]])
        s_info = site_infos.change_site(site_no)
        dis = (lat - s_info[1])**2 + (lon - s_info[2])**2
        inner = np.where(dis < disref**2)
        if attribute[3] in hf_a:
            flag = np.array(hf_a[attribute[3]])
            hf_a.close()
            return value[inner[0]], flag[inner[0]], lat[inner[0]], lon[inner[0]]
        else:
            hf_a.close()
            return value[inner[0]], -99, -99, -99


def get_h5_list(time_st, time_end, site_no, orbit, result_path='_05_01', excep=None):
    """
        A list of h5 files based on a time window functions that read data in array call this function Site No. and orbit are specified
    :param
        time_st:
    :param
        time_end:
    :param
        site_no:
    :param
        orbit:
    :return:
        final_list contains all the available files
        date_list contains the date string corresponding to finalist
    """
    h5_root = site_infos.get_data_path(result_path) + 's' + site_no + '/'  # previous result saved in document named by site no
    h5_list = []
    h5_date_list = []
    for filename in os.listdir(h5_root):  # selected orbit for a period of time
        if re.search(orbit, filename):
            h5_list.append(filename)
            h5_date_list.append(filename[-11:-3])
    h5_list = sorted(h5_list)
    h5_date_list = sorted(h5_date_list)
    date_num = []
    for dtr in h5_date_list:  # get the day of year
        doy = days_2015(dtr)
        date_num.append(doy)
    date_array = np.array(date_num)
    dt_st = days_2015(time_st)
    st_no = np.searchsorted(date_array, dt_st)  # number of start
    dt_end = days_2015(time_end)
    end_no = np.searchsorted(date_array, dt_end)
    final_list = h5_list[st_no:end_no + 1]
    date_list = h5_date_list[st_no:end_no + 1]
    if excep is not None:
        del final_list[date_list.index(excep)]
        del date_list[date_list.index(excep)]
    return final_list, date_list  # the date string list


def cal_OBD(site_no):
    """
    To generate a list containing all the available days of data in one site
    :param site_no:
    :return: ob_difference: col 0 is sigma0 difference
                            col 1 is TB difference
                            col 2 & 3 is quality flag of A & D orbit of radar data
    """
    # date
    data_root = site_infos.get_data_path() + 's' + site_no + '/'
    date_list = []
    for filename in os.listdir(data_root):  # the folder contains h5 results for a site
        date_list.append(filename[-11:-3])  # the date list
    date_list = set(date_list)
    date_list = sorted(date_list)
    site_info = site_infos.change_site(site_no)
    attributes = site_infos.get_attribute('sigma')  # the attribute don't contain the first layer of h5 file
    attributes2 = site_infos.get_attribute('tbn')
    ob_diference = np.zeros([4, len(date_list)])
    print ob_diference.shape
    n = 0
    date_str_tb = []
    date_str_sig = []
    for date_1 in date_list:  # loop dates in date list
        file_list = search_ob_file(date_1, data_root)  # file_list contains h5 files (tb & sig are saved)
        # if date_obj != '20150501':
        #     continue
        if n == 0:
            print 'date    ', 'radar       ', 'tb:    ', site_no
        if len(file_list) > 1:
            # radar data
            sigma0_A, flag_radar_A = site_read(file_list[0], site_info, attributes)
            sigma0_D, flag_radar_D = site_read(file_list[1], site_info, attributes)
            if sigma0_A == -9999 or sigma0_D == -9999:
                differ1 = -9999
            else:
                differ1 = 10 * np.log(sigma0_A) - 10 * np.log(sigma0_D)
                date_str_sig.append(date_1)
            # differ1, flag_radar_A, flag_radar_D = cal_difference(file_list[0], file_list[1], site_info, attributes)
            # tb data
            # print '=========error========='
            # print date_1
            tb_A, flag_tb_A = site_read(file_list[0], site_info, attributes2)
            if date_1 == '20150501':
                pause = 1
            tb_D, flag_tb_D = site_read(file_list[1], site_info, attributes2)
            if tb_A == -9999 or tb_D == -9999:
                differ2 = -9999
            else:
                differ2 = tb_A - tb_D
                date_str_tb.append(date_1)
            # differ2, flag_tb_A, flag_tb_D = cal_difference(file_list[0], file_list[1], site_info, attributes2)
            ob_diference[0][n] = differ1
            ob_diference[1][n] = differ2
            ob_diference[2][n] = flag_radar_A
            ob_diference[3][n] = flag_radar_D
            print date_1, differ1, differ2, flag_radar_A, flag_radar_D
        else:
            ob_diference[0][n], ob_diference[1][n] = -9999, -9999
        n += 1
    print len(date_list)
    return ob_diference, date_list


def read_ref(date, siteno):
    """
    Read data value of a specified site at a date
    """
    site_info = site_infos.change_site(siteno)
    path = site_infos.get_data_path()
    file_path = path + 's' + siteno + '/'
    file_name = search_ob_file(date, file_path)
    attribute = site_infos.get_attribute('sigma')
    if len(file_name) == 1:
        print 'only one orbit in %s' % date
        return -1, -1
    elif len(file_name) == 0:
        print 'No data in %s' % date
        return -1, -1
    else:
        ref_sigma, flag = site_read(file_name[0], site_info, attribute)
        print 'the reference on %s is %f ' % (date, ref_sigma)
        return ref_sigma, flag


def read_series(time_st, time_end, site_no, ob='_A_', data='radar', dat='sig_vv_aft', att_name='cell_tb_v_aft'):
    """

    :param time_st:
    :param time_end:
    :param site_no:
    :param ob:
    :param data: first layer: group. can be
    :param dat:
    :return: the [read data], date strings, distance
    """
    h5_list, d_list = get_h5_list(time_st, time_end, site_no, ob)  # obtain the file names of h5 files
    if len(h5_list) == 0:
        print 'no data in the time window between %s to %s' % (time_st, time_end)
        sys.exit()

    site_info = site_infos.change_site(site_no)
    attribute = site_infos.get_attribute(data, dat)
    path = site_infos.get_data_path('_05_01')
    site_path = path + '/s' + site_no + '/'
    series = []  # array that saves the data
    dis_all = []
    n = 0
    for h5_file in h5_list:

        abs_path = site_path + h5_file  # path of h5 file
        value, dis, status = site_read(abs_path, site_info, attribute)
        # for keyi in value.keys():
        #     print keyi, value[keyi]
        # print dis
        if status > 0:
            # print value.keys()
            # print dat
            v_inter = bs.dis_inter(dis, value[dat])  # interpolation
            series.append(v_inter)
            dis_all.append(dis)
        else:
            del d_list[n]
        n += 1
    return series, d_list, np.array(dis_all)


def get_date_list(site_no):
    """
    :param site_no: No. of site
    :return: date_list: contains all the available dates.
    """
    data_root = site_infos.get_data_path() + 's' + site_no + '/'
    date_list = []
    for filename in os.listdir(data_root):  # the folder contains h5 results for a site
        date_list.append(filename[-11:-3])  # the date list
    date_list = set(date_list)
    return date_list


def show_flag(flag_array):
    """
    Show quality flag in binary form, #display the location of '1'
    :param
        flag_array:
    :return:
        only display, no return
    """
    flag_a = np.unique(flag_array)
    for flag_num in flag_a:
        loc_1 = assess_flag(flag_num)
        ############################################
    return 0


def assess_flag(flag_num):
    """

    :param flag_num:
    :return: the location of '1' for a single data
            -9999 when flag_num is -9999
    """
    if not np.isnan(flag_num):
        if flag_num != -9999:
            flag_bin = bin(int(flag_num))
            # loc_1 is the location of '1' in the binary flag.
            loc_1 = [len(flag_bin) - 1 - pos for pos, char in enumerate(flag_bin) if char == '1']
            # print flag_num, bin(int(flag_num)), loc_1
            return loc_1
        else:
            return []  # flag is null
    else:
        return []  # here is a nan flag


def get_abnormal(flagged_value, flag_abnorm):
    """
    Transform the decimal flag into bin, then figure out the location of '1', then decide whether to set it as
    null(-9999)
    :param flagged_value:
        row0: value,  row1: decimal flag number
    :param flag_abnorm:
        location of '1' in binary flag number
    :return:
        1. value - the unqualified data are masked as -9999
        2. abnormal value - the masked data's original value
    """
    abnorm_value = np.zeros(flagged_value.shape)
    for i in range(0, flagged_value.shape[1]):
        flag_s = flagged_value[1][i]  # row1
        if flag_s != -9999:
            flag_list = assess_flag(flag_s)
            print flag_s, flag_list
            # print flag_list
            if flag_list != -9999:
                if any(True for x in flag_list if x in flag_abnorm):
                    abnorm_value[0][i] = i
                    abnorm_value[1][i] = flagged_value[0][i]
                    flagged_value[0][i] = -9999
                    # print flag_s, 'Quality fail'
            else:
                print '1111'
    return flagged_value, abnorm_value


def search_snotel(site_no, date,
                  att=["Soil Moisture Percent -2in (pct)", "Soil Temperature Observed -2in (degC)",
                         "Air Temperature Observed (degC)", "snow"]):
    if (site_no == '2065') | (site_no == '2081'):
        att[2] = "Air Temperature Average (degC)"
    site_type = site_infos.get_type(site_no)
    sno_file = './result_07_01/txtfiles/site_measure/'+site_type + site_no + '.txt'
    print site_type + site_no + '.txt'
    #
    n_row = -1
    with open(sno_file, 'rb') as f1:
        reader = csv.reader(f1)
        for row in reader:
            n_row += 1
            if n_row == 63:
                if 'Snow Depth (cm)' in row: # check SD or SWE
                    att[-1] = 'Snow Depth (cm)'
                else:
                    att[-1] = 'Snow Water Equivalent (mm)'
                col_no = [row.index(att0) for att0 in att]  # the col number for attributes
            if n_row > 63:
                t = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M").timetuple()
                doy = t.tm_yday + t.tm_hour/24.0
                if (doy == date) & (t.tm_year == 2016):
                    output = [row[i] for i in col_no]
                    break
            else:
                continue
    print t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_yday, date, '\n', att, '\n', output, '\n', '=================================='
    return output


def read_measurements(site_no, measure_name, doy, hr=0, year0=2016, t_unit='day'):
    """
    :param site_no:
    :param measure_name:
    :param doy:
    :param hr:
    :param year0:
    :param t_unit: sec or day
    :return:
    """
    sno_lib = site_infos.get_type(site_no, all_site=True)
    if t_unit =='sec':
        doy = bxy.get_total_sec('%d0101' % year0)+(doy-1)*3600*24
    if site_no not in sno_lib:
        print 'can not find station in the get_type function'
        return np.zeros(doy.size) - 999, doy
    site_type = site_infos.get_type(site_no)
    if year0 == 2016:
        site_file = './copy0519/txt/'+site_type + site_no + '.txt'
    else:
        site_file = './copy0519/txt/'+site_type + site_no + '_new.txt'
    if site_no in ['1183', '961', '952', '948', '1182', '958', '1234', '1232',
                '1266', '987', '1062', '1003', '1064', '1063', '1037', '1092', '1035', '1094',
                '1268', '1055', '1279', '1096', '1093', '1089', '1267', '1091', '1267', '1264', '1092', '954', '957',
                '1103', '1072']:
        site_file = './copy0519/txt/'+site_type + site_no + '_all.txt'
    if measure_name == 'snow':
        f_no = -1
    else:
        f_no = 0
    in_situ_name, t5 = read_sno(site_file, measure_name, site_no, field_no=f_no, t_unit=t_unit)  # t5: doy and measurements
    att=["Soil Moisture Percent -2in (pct)", "Soil Temperature Observed -2in (degC)",
                         "Air Temperature Observed (degC)", "Snow"]
    m_daily, m_doy = data_process.cal_emi(t5, [],  doy, hrs=hr, t_unit=t_unit)
    # check

    return m_daily, m_doy


def read_measurements_v3(site_no, measure_name, doy, hr=0, year0=2016, t_unit='day'):
    """
    v3: return a string of the name of in situ measurements, such as "Soil Moisture Percent -2in (pct)"
    :param site_no:
    :param measure_name:
    :param doy:
    :param hr:
    :param year0:
    :param t_unit: sec or day
    :return:
    """
    sno_lib = site_infos.get_type(site_no, all_site=True)
    if t_unit =='sec':
        doy = bxy.get_total_sec('%d0101' % year0)+(doy-1)*3600*24
    if site_no not in sno_lib:
        print 'can not find station in the get_type function'
        return np.zeros(doy.size) - 999, doy
    site_type = site_infos.get_type(site_no)
    if year0 == 2016:
        site_file = './copy0519/txt/'+site_type + site_no + '.txt'
    else:
        site_file = './copy0519/txt/'+site_type + site_no + '_new.txt'
    if site_no in ['1183', '961', '952', '948', '1182', '958', '1234', '1232',
                '1266', '987', '1062', '1003', '1064', '1063', '1037', '1092', '1035', '1094',
                '1268', '1055', '1279', '1096', '1093', '1089', '1267', '1091', '1267', '1264', '1092', '954', '957',
                '1103', '1072']:
        site_file = './copy0519/txt/'+site_type + site_no + '_all.txt'
    if measure_name == 'snow':
        f_no = -1
    else:
        f_no = 0
    in_situ_name, t5 = read_sno(site_file, measure_name, site_no, field_no=f_no, t_unit=t_unit)  # t5: doy and measurements
    att=["Soil Moisture Percent -2in (pct)", "Soil Temperature Observed -2in (degC)",
                         "Air Temperature Observed (degC)", "Snow"]
    m_daily, m_doy = data_process.cal_emi(t5, [],  doy, hrs=hr, t_unit=t_unit)
    # check

    return m_daily, m_doy, in_situ_name


def read_measurements_v2(site_no, measure_list, doy, hr=0, year0=2016):
    '''
    :param site_no:
    :param measure_list:
    :param doy:  doy should be secs at 0:00, updated 20190303
    :param hr:
    :param year0:
    :return:
    '''
    sno_lib = site_infos.get_type(site_no, all_site=True)
    if site_no not in sno_lib:
        return np.zeros(doy.size) - 999, doy
    site_type = site_infos.get_type(site_no)
    if year0 == 2016:
        site_file = './copy0519/txt/'+site_type + site_no + '.txt'
    else:
        site_file = './copy0519/txt/'+site_type + site_no + '_new.txt'
    stats_t, t5 = read_sno_v2(site_file, measure_list, site_no, field_no=0)  # t5: doy and measurements
    m_daily, m_doy = data_process.cal_emi_v2(t5, [], doy, hrs=hr)
    # pass hour for snow measurements should be 0
    if 'Snow Depth (cm)' in measure_list:
        snow_m = 'Snow Depth (cm)'
    elif ('Snow Water Equivalent (mm)' in measure_list):
        snow_m = 'Snow Water Equivalent (mm)'
    m_daily_2, m_doy_2 = data_process.cal_emi_v2(t5, [], doy, hrs=0)
    snow_data = m_daily_2[measure_list.index(snow_m)]
    valid_i = bxy.reject_outliers(snow_data, m=4)
    snow_data[~valid_i] = -99
    m_daily[measure_list.index(snow_m)] = snow_data
    m_dict, i0 = {}, 0
    for m_name in measure_list:
        m_daily[i0][m_daily[i0] < -90] = np.nan
        m_dict[m_name] = m_daily[i0]
        i0 += 1
    return m_dict, m_doy


def read_sno(sno_file, filed_name, station_id, field_no=0, t_unit='day'):
    """
    <description>
        Read the measurement of snotel sites. According to the formation of snotel txt data, lines 0~5 are the titles. Data reading is based on the station id.
    :param sno_file: filename of snow tel sites
    :param filed_no: the column of field, e.g., 3 for the soil moisture at 5 cm, 6 for temperature at 5 cm
    :param station_id:
    :param t_unit: sec or day
    :return:
        2 by n array, line 0 is day of year, line 1 is the value
    """
    print "reading %s..." % filed_name
    n_inter = 0  # iterate time, rows of sno_files
    n_ab = 0
    filed_value = []
    filed_date = []
    snow_id = 0
    filed_sec = []
    global abvalue
    with open('abvs', 'rb') as fab:
        read1 = csv.reader(fab)
        for row in read1:
            if n_ab > 5:
                abvalue = row[-1]
            n_ab += 1
    with open(sno_file, 'rb') as f1:
        reader = csv.reader(f1)
        index_row = -1
        for row in reader:
            if filed_name == "Snow":  # 'Snow Water Equivalent (mm)' or 'Snow Depth (cm)'
                    field_no = -1
            if row[0] == 'Date':  # the filed name list
                index_row = 0
                if filed_name != "snow":
                # if field_no != -1:  # not the snow reading
                    field_no = row.index(filed_name)
                else:
                    if 'Snow Water Equivalent (mm)' in row:
                        field_no = row.index('Snow Water Equivalent (mm)')
                    else:
                        field_no = row.index("Snow Depth (cm)")
                        snow_id = 1
                print 'col No. is: ', field_no, row[field_no]  # show what filed is read
                field_name = row[field_no]
                # if row[field_no] == 'Snow Depth (cm)':  # special for snow
                #     snow_id = 1
            elif index_row>-1:
                t = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M").timetuple()
                if t.tm_year - 2016 > 0:
                    leap = 1
                else:
                    leap = 0
                doy = t.tm_yday + t.tm_hour/24.0 + 365*(t.tm_year - 2015) + leap
                secs_2_2000 = bxy.get_total_sec(row[0], fmt="%Y-%m-%d %H:%M", reftime=[2000, 1, 1, 0])
                if row[field_no] == abvalue:
                    filed_value.append(float(-99))
                    filed_date.append(doy)
                    filed_sec.append(secs_2_2000)
                else:
                    if snow_id != 1:  # remove the invalid snow depth
                        filed_value.append(float(row[field_no]))
                        filed_date.append(doy)
                        filed_sec.append(secs_2_2000)
                    else:
                        if (float(row[field_no]) > 190):
                            filed_value.append(float(-99))
                            filed_date.append(doy)
                            filed_sec.append(secs_2_2000)
                        else:
                            filed_value.append(float(row[field_no]))
                            filed_date.append(doy)
                            filed_sec.append(secs_2_2000)
            n_inter += 1
    f1.closed
    if t_unit == 'sec':
        return field_name, np.array([filed_sec, filed_value])  # return field_name, rather than 1
    return field_name, np.array([filed_date, filed_value])


def read_sno_v2(sno_file, filed_name, station_id, field_no=0):
    """
    <description>
        Read the measurement of snotel sites. The heads, meta data were skiped
    :param sno_file: filename of snow tel sites
    :param filed_no: the column of field, e.g., 3 for the soil moisture at 5 cm, 6 for temperature at 5 cm
    :param station_id:
    :return:
        2 by n array, line 0 is day of year, line 1 is the value
    """
    site_type = site_infos.get_type(station_id)
    # site_file = './copy0519/txt/'+site_type + site_no + '_new.txt'
    # sno_file = './copy0519/txt/'+site_type + station_id + '.txt'
    print "reading %s..." % filed_name
    n_inter = 0  # iterate time, rows of sno_files
    n_ab = 0
    filed_value = []
    filed_date = []
    filed_sec = []
    snow_id = 0
    if type(filed_name) is not list:
        filed_name = [filed_name]
    global abvalue
    with open('abvs', 'rb') as fab:
        read1 = csv.reader(fab)
        for row in read1:
            if n_ab > 5:
                abvalue = row[-1]
            n_ab += 1
    # np.loadtxt(sno_file, delimiter=',', skiprows=64)
    field_nos = []
    with open(sno_file, 'rb') as f1:
        reader = csv.reader(f1)
        index_row = -1
        for row in reader:
            if filed_name == "Snow":  # 'Snow Water Equivalent (mm)' or 'Snow Depth (cm)'
                    field_no = -1
            if row[0] == 'Date':  # the filed name list
                print 'the snow measurements is SWE', 'Snow Water Equivalent (mm)' in row
                print 'snow to be readed,', "Snow" in filed_name
                print 'the readed fields', filed_name
                if "Snow" in filed_name:
                    if 'Snow Water Equivalent (mm)' in row:
                        filed_name[filed_name.index("Snow")] = "Snow Water Equivalent (mm)"
                    else:
                        filed_name[filed_name.index("Snow")] = "Snow Depth (cm)"


                # print 'the file name:', sno_file
                # print 'all fields are: ', filed_name
                for m_name in filed_name:
                    field_nos.append(row.index(m_name))
                index_row = 0

                for f0 in field_nos:
                    print 'col No. is: %d %s' %(f0, row[f0])  # show what filed is read
                # if row[field_no] == 'Snow Depth (cm)':  # special for snow
                #     snow_id = 1
            elif index_row>-1:
                t = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M").timetuple()
                if t.tm_year - 2016 > 0:
                    leap = 1
                else:
                    leap = 0
                doy = t.tm_yday + t.tm_hour/24.0 + 365*(t.tm_year - 2015) + leap
                # update 20190304, get tolal secs using the str in forms of "%Y-%m-%d %H:%M"
                secs_2_2015 = bxy.get_total_sec(row[0], fmt="%Y-%m-%d %H:%M", reftime=[2015, 1, 1, 0])
                doy_2_2015 = secs_2_2015/24/3600
                secs_2_2000 = bxy.get_total_sec(row[0], fmt="%Y-%m-%d %H:%M", reftime=[2000, 1, 1, 0])
                daily_measure = []
                for f0 in field_nos:
                    if row[f0] == abvalue:
                        row[f0] = -99
                    # print 'the appended value is',
                    daily_measure.append(float(row[f0]))
                filed_value.append(np.array(daily_measure))
                filed_date.append(doy)
                filed_sec.append(secs_2_2000)
                # if row[field_no] == abvalue:
                #     filed_value.append(float(-99))
                #     filed_date.append(doy)
                # else:
                #     if snow_id != 1:  # remove the invalid snow depth
                #         filed_value.append(float(row[field_nos]))
                #         filed_date.append(doy)
                #     else:
                #         if (float(row[field_no]) > 190):
                #         # (float(row[field_no]) in np.array([102,104,107,112,130,140,183,241,231,239])) | \
                #             filed_value.append(float(-99))
                #             filed_date.append(doy)
                #         else:
                #             filed_value.append(float(row[field_no]))
                #             filed_date.append(doy)
            n_inter += 1
    f1.closed
    measurements = np.zeros([len(daily_measure)+1, len(filed_date)])
    # measurements[0] = np.array(filed_date)
    measurements[0] = np.array(filed_sec)
    measurements[1:] = np.array(filed_value).T
    # np.array([np.array(filed_date),  np.array(filed_value).T])
    return 1, measurements


def days_2015(dtr):
    '''
    <introduction>
        get the day of year from 20150101
    :param dtr:
    :return:
    '''
    if type(dtr) is list:
        doy = []
        for dt in dtr:
            t = datetime.datetime.strptime(dt, "%Y%m%d").timetuple()
            doy.append(t.tm_yday + 365 * (t.tm_year - 2015))
        return np.array(doy)
    else:
        t = datetime.datetime.strptime(dtr, "%Y%m%d").timetuple()
        doy = t.tm_yday + 365 * (t.tm_year - 2015)
        return doy


def unsortsearch(list, target):
    for num in list:
        if num < target:
            continue
        else:
            return num


def rm_empty_measure(record):
    record[np.where(record == -99.0)] = np.nan
    return record


def read_ref(csvfile):
    ref = np.array([])
    with open(csvfile) as fsno:
        sno = csv.reader(fsno)
        for row in sno:
            wi_i = [float(x) for x in row]
            ref = np.append(ref, np.array(wi_i), axis=0)
    return ref


def read_ascat_txt(txtfile):
    n_rows = 0
    out = [[], [], [], [], [], []]  # date, lat, lon, sig, inc, orbit
    with open(txtfile, 'rb') as f1:
        reader = csv.reader(f1)
        for row in reader:
            datei, lati, loni = row[0], row[1], row[2]
            tp = row[9: 12]
            inc_tri = [float(x) for x in row[9:12]]
            inc_np = np.array(inc_tri)
            if inc_np[0]>1e2:
                inc_np *=1e-2
            inc0 = np.min(np.abs(inc_np - 45))
            if inc0 > 5:
                continue
            else:
                i0 = np.argmin(np.abs(inc_np - 45))
                out[0].append(float(datei))
                out[3].append(float(row[3+i0]))
                out[4].append(inc_np[i0])
                out[5].append(float(row[12]))
        output = np.array(out[3])
        if output[0] < -1e5:
            output *= 1e-6
        return [out[0], output, out[4], out[5]]  # date, value, inci, orb


def fix_angle_ascat(txtfile):
    # get fix function:
    txt_table = np.loadtxt(txtfile, delimiter=',')
    tx = txt_table[:,0]  # time line
    p1, p2, p3, p4 = np.where(tx<90), np.where((tx>90)&(tx<120)), np.where((tx>150)&(tx<250)), np.where(tx>270)
    f = [];
    for p in [p1]:
        inci1 = txt_table[p, 9:12].reshape(-1, 3)
        sig1 = txt_table[p, 3:6].reshape(-1, 3)
        if sig1[0, 0] < -1e5:
            inci1 *= 1e-2
            sig1 *= 1e-6
        # linear regression
        x = np.concatenate((inci1[:,0].T, inci1[:,1].T, inci1[:,2].T))
        y = np.concatenate((sig1[:,0].T, sig1[:,1].T, sig1[:,2].T))
        a, b = np.polyfit(x, y, 1)
        f.append(np.poly1d([a, b]))
    # f[0] for frozen season, f[1] fro thawed season
    sigf, sigm, sigaf = txt_table[:, 3].T, txt_table[:, 4].T, txt_table[:, 5].T
    incf, incm, incaf = txt_table[:, 9].T, txt_table[:, 10].T, txt_table[:, 11].T
    # fitting
    inc_ref = 45
    # sigf[0: 100] += f[0](incf[0: 100])-f[0](inc_ref)
    # sigf[100: 270] += f[0](incf[100: 270])-f[0](inc_ref)
    # sigf[270: ] += f[0](incf[270: ])-f[0](inc_ref)
    #
    # sigm[0: 100] += f[0](incm[0: 100])-f[0](inc_ref)
    # sigm[100: 270] += f[0](incm[100: 270])-f[0](inc_ref)
    # sigm[270: ] += f[0](incm[270: ])-f[0](inc_ref)
    #
    # sigaf[0: 100] += f[0](incm[0: 100])-f[0](inc_ref)
    # sigaf[100: 270] += f[0](incm[100: 270])-f[0](inc_ref)
    # sigaf[270: ] += f[0](incm[270: ])-f[0](inc_ref)
    if sigf[0]<-5e5:
        sigf *= 1e-6
        sigm *= 1e-6
        sigaf *= 1e-6
        incf *= 1e-2
        incm *= 1e-2
        incaf *= 1e-2
    sigf -= (incf-inc_ref)*a
    print a, np.std(sigm)
    sigm -= (incm-inc_ref)*a
    print np.std(sigm)
    sigaf -= (incaf-inc_ref)*a
    # fig = plt.figure(figsize=[4, 3])
    # ax = fig.add_subplot(111)
    # ax.plot(incm-inc_ref, sigm, 'bo')
    # plt.savefig('ascat_cor_inc.png', dpi=120)
    return [tx, sigf, sigm, sigaf, txt_table[:, -1].T]  # date, sigma triplets, orbit


def read_diurnal(date, hr_as, hr_des, siteno, att_name):
    """

    :param date:
    :param hr_as:
    :param hr_des:
    :param siteno:
    :param att_name: measurement for read, like snow water equavailent
    :return:
    """
    y2_empty = np.zeros(date.size) + 1
    stats_sm, sm5 = read_sno(' ', "Soil Moisture Percent -2in (pct)", siteno)  # air tmp
    sm_as, sm_date_as = data_process.cal_emi(sm5, y2_empty, date, hrs=hr_as)
    return 0


def read_tibet(site_no):
    site_dict = {'20000': 'Ali01_2016.txt', '20001': 'CST05_2016.txt', '20002': 'NST01_2016.txt',
                 '20003': 'SQ02 _2016.txt'}
    path = '/home/xiyu/Data/tibet/%s' % site_dict[site_no]
    m_out = []
    with open(path) as f0:
        row_no = 0
        for row in f0:
            row_list = row.split('\t')
            if row_no > 2:
                row_list = row.split('\t')  # time, vwc, temp
                m_out.append([bxy.get_total_sec(row_list[0], fmt='%m/%d/%Y %H:%M', reftime=[2000, 1, 1, 12]),
                              float(row_list[1]), float(row_list[2])])
            row_no += 1
    return np.array(m_out)


def get_secs_values(site_no, measure_name, doy, ref_date='20160101', nan_value=0, pass_hr=0):
    measure0, date0 = read_measurements(site_no, measure_name, doy+365, hr=pass_hr)
    measure0[measure0 < nan_value] = np.nan
    sec0 = bxy.get_total_sec(ref_date) + (date0-366)*24*3600
    return measure0, sec0
# def read_measurement(t0, var_name, site_no):
#     """
#
#     :param t0: <list>: [day_of_year, pass_hr]
#     :param var_name: <string>: name of measurement
#     :param site_no:
#     :return: <list>: [day_of_year, measurement]
#     """
#     statu, var_all = read_sno('_', var_name, site_no)
#     m_value, m_date = data_process.cal_emi(var_all, ' ', t0[0], hrs=t0[1])
#     return [np.modf(m_date)[1], np.round(np.modf(m_date)[0]*24), m_value]


def get_3_year_insitu(station_id_int, m_name='air'):
    """

    :param station_id_int:
    :param m_name:
    :return: ndarray with shape equals (period length , 2*number of period). For every two elements, the 1st is the time in unit of secs,
    the other is the value.
    """
    # get in situ measurement
    # pixel_num = np.loadtxt('result_agu/result_2019/points_num_tp.txt', delimiter=',')
    # num01 = pixel_num[0][pixel_num[1] == t_station][0].astype(int)
    tair0, tair1, tair2 = in_situ_series(station_id_int, in_situ_measure=m_name), \
                          in_situ_series(station_id_int, y=2017, in_situ_measure=m_name), \
                          in_situ_series(station_id_int, y=2018, in_situ_measure=m_name)
    t_air_all = np.concatenate((tair0, tair1, tair2), axis=1)  # dimension: att(x y), date, pass hr(7 14 18)
    # plot 3 year data
    air_plot = \
        [t_air_all[0, :, 0], t_air_all[1, :, 0],
         t_air_all[0, :, 1], t_air_all[1, :, 1],
         t_air_all[0, :, 2], t_air_all[1, :, 2]]
    # np.savetxt('result_08_01/plot_data/%d_%s_%d.txt' % (num01, m_name, t_station), air_plot)
    return np.array(air_plot)


def in_situ_series(sno, y=2016, in_situ_measure='air', hr=[7, 14, 18]):
    m_name = site_infos.in_situ_vars(sno)
    if in_situ_measure == 'air':
        if sno in [2065, 2081]:
            in_situ_measure = 'Air Temperature Average (degC)'
        else:
            in_situ_measure = 'Air Temperature Observed (degC)'
    secs_insitu_0 = bxy.get_total_sec('%d0101' % y)
    # doy_insitu = np.array([np.arange(1, 366), np.arange(1, 366), np.arange(1, 366)]).T.ravel()
    num_hr = len(hr)
    doy_insitu = np.matmul(np.array(np.zeros([num_hr, 1])+1), np.array([np.arange(1, 366)])).T.ravel()
    hr_array = np.matmul(np.array([hr]).T, np.array([np.zeros(365)+1])).T.ravel()
    # hr_array0 = np.zeros([doy_insitu.size/3, 1])
    # hr_array = np.matmul(hr_array0+1, np.array([hr]))
    air_value_tr, air_date_tr, in_situ_name = \
            read_measurements_v3(str(sno), in_situ_measure, doy_insitu.astype(int),
                                 hr=hr_array.astype(int), t_unit='sec', year0=y)
    # check x time
    x_time = bxy.time_getlocaltime(air_date_tr, ref_time=[2000, 1, 1, 0], t_source='US/Alaska')
    air_value, air_date = air_value_tr.reshape(-1, num_hr), air_date_tr.reshape(-1, num_hr)  # Col 0, 1, 2: hr7, hr14 hr18
    # air_win = 7
    # w, w_valid = data_process.n_convolve3(air_value[0], air_win)
    # air0_index0 = np.where(w>5)
    # for ind0 in air0_index0[0]:
    #     if air_date[ind0] > bxy.get_total_sec('%d0307' % year0):
    #         tair_zero_day = air_date[ind0] - air_win*24*3600
    #         break
    return np.array([air_date, air_value]), in_situ_name