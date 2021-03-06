__author__ = 'xiyu'
import test_def, peakdetect
import csv
import numpy as np
import read_site
import site_infos
import Read_radar
import h5py
import data_process
import os, re, sys
import warnings
from test_def import plt_npr_gaussian_ASC


site_nos = ['2081', '947', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177',  '2210', '1089', '1233', '2212', '2211']
#site_nos = ['947']
# '947', '949', '950', '960', '962', '968','1090', '1175', '1177'
#'967', '2065', '2081', '2210', '2213', '1089', '1233', '2212', '2211',
# site_nos = ['1062']
site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968','1090', '1175', '1177'],
                    'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
# ASCAT process
n_pixel = []
for site_no in site_nos:
    x_time, sigma0, date_list, out_ascat, inc45_55 = [], [], [], [], []
    txt_path = '/home/xiyu/PycharmProjects/R3/result_03_4/' + 's' + site_no + '/'  # 02: descending, 03: ascending
    date_record = []
    n = 0
    n_warning, n_inc = 0, 0
    file_list = sorted(os.listdir(txt_path))
    for txt_file in file_list:
        # get dt
        n+=1
        p_uline = re.compile('_')
        datei = p_uline.split(txt_file)[1][0: 8]
        # with warnings.catch_warnings():
        #     txt_i = np.loadtxt(txt_path + txt_file, delimiter=',')
        #     n_warning += 1
        #     print n_warning
        print txt_file
        txt_i = np.loadtxt(txt_path + txt_file, delimiter=',')
        if txt_i.size < 5:
            n_warning += 1
            print 'no data is %s' % datei
            continue
        if txt_i.size > 5:
            date_list.append(datei)  # get list of date for in situ data readingh
            if len(txt_i.shape) < 2:  # only 1- d
                locs, sig = np.array([txt_i[0:2]*1]), np.array([txt_i[2:5]*1])
                f_u, inc = np.array([txt_i[5:8]]), np.array([txt_i[8:11]])
                orb = np.array([txt_i[-1]])
            elif txt_i.shape[1] > 10:  # with triplets, flag, and inc angles
                locs, sig = txt_i[:,  0:2], txt_i[:, 2:5]
                f_u, inc = txt_i[:,  5:8]*1, txt_i[:, 8:11]
                f_land = txt_i[:, 11:14]
                orb = txt_i[:, -1]

            t1, numz, ascat_ind, status = data_process.interp_geo(locs.T[1], locs.T[0], sig.T, orb.T, site_no, disref=0, f=f_u.T, incidence=inc.T)

            if status>0:
                # if ascat_ind >= 0:
                #     inc_tri = inc[ascat_ind, :]
                #     inc0 = np.min(np.abs(inc_tri-5000))
                #     if inc0 > 500:
                #         n_inc += 1
                #         continue
                #     else:
                #         i0 = np.argmin(np.abs(inc_tri-5000))
                #         sigma_inc = sig[ascat_ind, i0]
                #         inc45_55.append(inc[ascat_ind, i0])
                #         t1 = sigma_inc
                sigma0+=t1
                xtime = np.array([read_site.days_2015(datei) - 365])
                date_record.append(txt_file)
                if ascat_ind >= 0:
                    if len(txt_i.shape) < 2:
                        out_ascat.append(np.concatenate((xtime, txt_i)))
                    else:
                        for i in ascat_ind:
                            tp = np.concatenate((xtime, txt_i[i, :]))
                            out_ascat.append(tp)
            else:
                continue
        #print n
    # n_pixel.append(numz)
    # x_time_np = np.array([x_time])
    # out_ascat_np = np.array(out_ascat)
    # nn_series = np.array(out_ascat)
    # txtname = 'ascat_series_'+site_no+'.txt'
    # np.savetxt(txtname, nn_series, delimiter=',',
    #             fmt='%d, %.6f, %.6f, %.6f, %.6f, %.6f, %d, %d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %d')
    # read from txt, controlled by incidence angle

    txtname = 'ascat_series_'+site_no+'.txt'
    # angle-selected
    # out = read_site.read_ascat_txt(txtname)  # date, sig, inc, orb
    # orb_no = np.array(out[3])
    # ob_as = np.where(orb_no<0.5)
    # ob_des = np.where(orb_no>0.5)
    # x_time = np.array(out[0])[ob_as]
    # x_time_np = np.array(x_time)
    # sigma0_np = np.array(np.array(out[1])[ob_as])
    # angle-corrected
    out = read_site.fix_angle_ascat(txtname)
    orb_no = np.array(out[4])
    ob_as = np.where(orb_no<0.5)
    ob_des = np.where(orb_no>0.5)
    x_time = np.array(out[0])[ob_as]
    x_time_np = np.array(x_time)
    sigma0_np = np.array(np.array(out[2])[ob_as])  # mid-observation
    if sigma0_np[0] < -1e5:
        sigma0_np *= 1e-6
    # angle-corrected

    # sigma0_fore = [fore[0] for fore in sigma0]
    # sigma0_mid = [fore[1] for fore in sigma0]
    # sigma0_aft = [fore[2] for fore in sigma0]
    # sigma0_all = [sigma0_fore, sigma0_mid, sigma0_aft]
    # sigma0_np = np.array(sigma0_all[0])
    # sigma0_np[np.where(sigma0_np<-90)] = np.nan

    # smooth
    sm_window = 7
    o_ind =range(0, len(x_time), 1)
    val_ind =o_ind[(sm_window-1)/2: len(x_time)-(sm_window+1)/2 + 1]  # val. ind for smooth x
    val_time = x_time_np[val_ind]
    sigma0_s, valid = data_process.n_convolve2(sigma0_np, sm_window)  # val. ind for smooth y
    sigma0_s = sigma0_s[valid]
    # plot time-series
    #test_def.ascat_plot1(x_time, sigma0)
    #test_def.ascat_plot1(val_time, sigma0_s, fname='947smooth')

    # edge detect
    g_size = 8
    g_sig_s, i_gaussian = data_process.gauss_conv(sigma0_s, sig=3)  # smoothed sigmas
    g_sig_s_valid = (g_sig_s[g_size: -g_size] - np.nanmin(g_sig_s[g_size: -g_size]))\
                    /(np.nanmax(g_sig_s[g_size: -g_size]) - np.nanmin(g_sig_s[g_size: -g_size]))
    g_sig, ig2 = data_process.gauss_conv(sigma0_np, sig=3)  # non-smoothed
    g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                    /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))

    # get the peak
    max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, x_time_np[g_size: -g_size])
    #print txt_file, n_warning
    onset = data_process.find_inflect(max_gsig_s, min_gsig_s, typez='annual')

    # in situ measurement

    passtime = [21]
    # site_dic = {'sno_': ['947']}
    for key_site in site_dic.keys():
        if site_no in site_dic[key_site]:
            site_type = key_site
            break
        else:
            continue
    site_file = site_type + site_no + '.txt'
    y2_empty = np.zeros(sigma0_np.size) + 1
    stats_sm, sm5 = read_site.read_sno(site_file, "Soil Moisture Percent -2in (pct)", site_no)  # air tmp
    sm5_daily, sm5_date = data_process.cal_emi(sm5, y2_empty, date_list, hrs=passtime)
    stats_t, t5 = read_site.read_sno(site_file, "Soil Temperature Observed -2in (degC)", site_no)
    t5_daily, t5_date = data_process.cal_emi(t5, y2_empty, date_list, hrs=passtime)
    stats_swe, swe = read_site.read_sno(site_file, "snow", site_no, field_no=-1)
    swe_daily, swe_date = data_process.cal_emi(swe, y2_empty, date_list, hrs=passtime)
    sm5_daily[sm5_daily< -90], t5_daily[t5_daily < -90], \
    swe_daily[swe_daily < -90] = np.nan, np.nan, np.nan  # remove missing
    # print 'site no is %s' % site_no
    print 'station ID is %s' % site_no
    ons_site = data_process.sm_onset(sm5_date-365, sm5_daily, t5_daily)
    onset.append(ons_site[0]), onset.append(ons_site[1])
    # test_def.ascat_plot1([val_time, val_time[i_gaussian][g_size: -g_size], sm5_date-365, t5_date-365, swe_date-365],
    #                      [sigma0_s, g_sig_s_valid, sm5_daily, t5_daily, swe_daily], fname='ascat_SG'+site_no, vline=onset)
    test_def.ascat_plot1([x_time, x_time_np[ig2][g_size: -g_size], sm5_date-365, t5_date-365, swe_date-365],
                         [sigma0_np, g_sig_valid, sm5_daily, t5_daily, swe_daily], fname='ascat_OG'+site_no, vline=onset)
    plt_npr_gaussian_ASC(['E(t)', x_time_np[ig2][g_size: -g_size]+365, g_sig_valid],  # npr
                             # ['TB_v', t_v, var_npv],
                             ['Soil moisture (%)', sm5_date, sm5_daily],  # soil moisture
                             ['Soil temperature (DegC)', t5_date, t5_daily],
                             ['SWE (mm)', swe_date, swe_daily],
                             ['$\sigma^0$', x_time+365, sigma0_np],  # 511
                             fa_2=[], vline=onset,  # edge detection, !! t_v was changed
                             #ground_data=ons_site,
                             fe_2=[],
                             figname='onset_based_on_ASCAT'+site_no+'2.png', mode='annual')




