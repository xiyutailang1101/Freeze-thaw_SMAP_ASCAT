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


def read_measurements(site_no, measure_name, doy, hr=0):
    sno_lib = site_infos.get_type(site_no, all_site=True)
    if site_no not in sno_lib:
        return np.zeros(doy.size) - 999, doy
    site_type = site_infos.get_type(site_no)
    site_file = './copy0519/txt/'+site_type + site_no + '.txt'
    if measure_name == 'snow':
        f_no = -1
    else:
        f_no = 0
    stats_t, t5 = read_sno(site_file, measure_name, site_no, field_no=f_no)
    m_daily, m_doy = data_process.cal_emi(t5, [], doy, hrs=hr)
    return m_daily, m_doy


def read_sno(sno_file, filed_name, station_id, field_no=0):
    """
    <description>
        Read the measurement of snotel sites. According to the formation of snotel txt data, lines 0~5 are the titles. Data reading is based on the station id.
    :param sno_file: filename of snow tel sites
    :param filed_no: the column of field, e.g., 3 for the soil moisture at 5 cm, 6 for temperature at 5 cm
    :param station_id:
    :return:
        2 by n array, line 0 is day of year, line 1 is the value
    """
    site_type = site_infos.get_type(station_id)
    sno_file = './copy0519/txt/'+site_type + station_id + '.txt'
    print "reading %s..." % filed_name
    n_inter = 0  # iterate time, rows of sno_files
    n_ab = 0
    filed_value = []
    filed_date = []
    snow_id = 0
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
                # if row[field_no] == 'Snow Depth (cm)':  # special for snow
                #     snow_id = 1
            elif index_row>-1:
                t = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M").timetuple()
                doy = t.tm_yday + t.tm_hour/24.0 + 365*(t.tm_year - 2015)
                if row[field_no] == abvalue:
                    filed_value.append(float(-99))
                    filed_date.append(doy)
                else:
                    if snow_id != 1:  # remove the invalid snow depth
                        filed_value.append(float(row[field_no]))
                        filed_date.append(doy)
                    else:
                        if (float(row[field_no]) > 190):
                        # (float(row[field_no]) in np.array([102,104,107,112,130,140,183,241,231,239])) | \
                            filed_value.append(float(-99))
                            filed_date.append(doy)
                        else:
                            filed_value.append(float(row[field_no]))
                            filed_date.append(doy)
            n_inter += 1
    f1.closed
    return 1, np.array([filed_date, filed_value])


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
