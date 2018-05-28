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
from test_def import plt_npr_gaussian_all
from plot_funcs import pltyy
import plot_funcs
import matplotlib.pyplot as plt
from datetime import datetime
from basic_xiyu import h5_writein


prefix = './result_07_01/'
site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'] #'1089'
site_nos = ['947']
sha = {'947': [[90, 115], [60, 120]], '968': [[120, 145], [90, 150]], '2213': [100, 120], '1177': [[100, 150], [100, 150]]}
# '947', '949', '950', '960', '962', '968','1090', '1175', '1177'
#'967', '2065', '2081', '2210', '2213', '1089', '1233', '2212', '2211',
site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968','1090', '1175', '1177'],
                    'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
stds = [7]
# initiations:
orders = 1
obs = [0, '_A_', 18]  # 0: As, 1:Des
# obs = [1, '_D_', 8]  # 0: As, 1:Des
i_csv = -1
save_h5 = False
row_nums = len(site_nos)
site_result = np.zeros([row_nums, 6])
tb_result = np.zeros([row_nums, 5])
npr_result1st, ascat_result1st = np.zeros([row_nums, 3]), np.zeros([row_nums, 3])
npr_result2nd, ascat_result2nd = np.zeros([row_nums, 5]), np.zeros([row_nums, 5])
# ASCAT process
n_pixel = []
onset_save = []
gau0_tb = []
# dsm_npr: for moisture change in that day
for site_no in site_nos:
    i_csv += 1
    # site read
    pp = False
    doy = np.arange(1, 366) + 365
    si0 = site_no
    site_type = site_infos.get_type(si0)
    site_file = './copy0519/txt/'+site_type + si0 + '.txt'
    y2_empty = 0
    stats_sm, sm5 = read_site.read_sno(site_file, "Soil Moisture Percent -2in (pct)", si0)  # air tmp
    sm5_daily, sm5_date = data_process.cal_emi(sm5, y2_empty, doy, hrs=obs[2])
    stats_t, t5 = read_site.read_sno(site_file, "Soil Temperature Observed -2in (degC)", si0)
    t5_daily, t5_date = data_process.cal_emi(t5, y2_empty, doy, hrs=obs[2])
    # Air Temperature Observed (degC)

    stats_t, tair5 = read_site.read_sno(site_file, "Air Temperature Observed (degC)", si0)
    tair_daily, tair_date = data_process.cal_emi(tair5, y2_empty, doy, hrs=obs[2])
    if site_no in ['2065', '2081']:
        stats_t, tair5 = read_site.read_sno(site_file, "Air Temperature Average (degC)", si0)
        tair_daily, tair_date = data_process.cal_emi(tair5, y2_empty, doy, hrs=obs[2])
    tair_date-=365
    if pp:
        stats_swe, swe = read_site.read_sno(site_file, "Precipitation Increment (mm)", si0)
    else:
        stats_swe, swe = read_site.read_sno(site_file, "snow", si0, field_no=-1)
    # read snow data
    swe_daily, swe_date = data_process.cal_emi(swe, y2_empty, doy, hrs=0)
    sm5_daily[sm5_daily < -90], t5_daily[t5_daily < -90], \
    swe_daily[swe_daily < -90] = np.nan, np.nan, np.nan
    sm, tsoil, swe = [sm5_date-365, sm5_daily], [t5_date-365, t5_daily], [swe_date-365, swe_daily]

    ons_site, ons_tsoil, day2 = data_process.sm_onset(sm[0], sm[1], tsoil[1])  # onset from tsoil and moisture
    site_result[i_csv] = np.array([ons_tsoil[0], ons_site[0], day2[0], ons_tsoil[1], ons_site[1], float(site_no)])
    gau1_npr, Emax_npr, dsm_npr, dswe_npr, dsoil_npr = [], [], [], [], []
    # center_tb = test_def.main(site_no, [], sm_wind=7, mode='annual', seriestype='tb', tbob='_A_', sig0=7, centers=True)

    for k_width in stds:  # ,7, 8, 9, 10 1, 2.5, 3, 4, 5, 6, 7, 8, 9, 10
        print k_width
        precip = False
        '''
        Edge detection on ASCAT
        '''
        # if str(site_no) == '968' & orders == 2:
        #     k_width = 8
        sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series(site_no, orb_no=obs[0], inc_plot=True, sigma_g=k_width, pp=precip,
                                               order=1)
        # if orders == 2:
        #     print 'second order of guassian'
        #     sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
        #         data_process.ascat_order2(site_no, orb_no=obs[0], inc_plot=False, sigma_g=k_width, pp=precip)# 0 for ascending
        # else:
        #     sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
        #         data_process.ascat_plot_series(site_no, orb_no=obs[0], inc_plot=False, sigma_g=k_width, pp=precip)# 0 for ascending
        sigconv[0]-=365
        x_time, sigma0, date_list, out_ascat, inc45_55 = [], [], [], [], []
        print 'station ID is %s' % site_no

        # some extra process
        # date0, value0 = sm5[0], sm5[1]
        # index2016 = (date0>365)&(date0<730)&(value0>-90)&(np.abs(date0-365-267) >= 1)
        # sm5_daily, sm5_date = value0[index2016], date0[index2016]
        # stats_sm, rain = read_site.read_sno(site_file, "Air Temperature Observed (degC)", site_no)  # percipitation
        # rain_value, rain_date = rain[1][index2016], rain[0][index2016]
        '''
        Edge detection on SMAP
        '''
        tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                         seriestype='tb', tbob=obs[1], sig0=k_width, order=1)  # result tb
        tb_onsetnew = data_process.tb_1st(ons0, gau0)
        plot_funcs.simple_plot(tbv0)
        gau0_tb.append(gau0)
        tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                          tbob=obs[1], sig0=k_width, order=1)  # result npr
        gau1_npr.append(gau1)  # gau1: normalized E(t), peakdate: the date when E(t) reaches max/min
        npr_end = data_process.edge_2nd(ons1, gau1)

        # save the 2nd result
        if orders == 2:
            tb_result[i_csv] = np.array([ons0[0], tb_onsetnew[0], ons0[1], tb_onsetnew[1], float(site_no)])
            ascat_end = data_process.edge_2nd(ons_new, sigconv)
            ascat_result2nd[i_csv] = np.array([ons_new[0], ascat_end[0], ons_new[1], ascat_end[1], float(site_no)])
            npr_end = data_process.edge_2nd(ons1, gau1)
            npr_result2nd[i_csv] = np.array([ons1[0], npr_end[0], ons1[1], npr_end[1], float(site_no)])
        # get the extremum of Et during transition
        if save_h5:
            data_process.save_transition(site_no, k_width, peakdate_sig, sigconv, sm, tsoil, swe, layers='freeze/ascat')
            data_process.save_transition(site_no, k_width, peakdate0, gau0, sm, tsoil, swe, layers='freeze/tb')
            data_process.save_transition(site_no, k_width, peakdate1, gau1, sm, tsoil, swe, layers='freeze/npr')
            data_process.save_transition(site_no, k_width, peakdate0, gau0, sm, tsoil, swe, layers='thaw/tb')
            data_process.save_transition(site_no, k_width, peakdate1, gau1, sm, tsoil, swe, layers='thaw/npr')
            data_process.save_transition(site_no, k_width, peakdate_sig, sigconv, sm, tsoil, swe, layers='thaw/ascat')
            # save all series data: smap, ascat and station
            if k_width == 1:
                narr2 =[[tbv0[0], tbv0[1], tbh0[1]], [npr1[0], npr1[1]],  # smap: tb and npr
                        [sigseries[0], sigseries[1]],  # ascat
                        [tsoil[0], tsoil[1]],  # soil temp
                        [sm[0], sm[1]],  # soil moisture
                        [swe[0], swe[1]]]  # swe
                key2 = ['tb', 'npr', 'ascat', 'tsoil', 'vwc', 'snow']
                for i in range(0, 5):
                    layer2 = "all_2016/"+key2[i]
                    narr2i = np.array(narr2[i])
                    h5_writein('result_07_01/methods/transition_'+site_no+'_v00.h5', "all_2016/"+key2[i], narr2i)

            # save all the convolution series
            h5_writein('result_07_01/methods/transition_'+site_no+'_v00.h5', "conv/npr/width_"+str(k_width), np.array(gau1))
            h5_writein('result_07_01/methods/transition_'+site_no+'_v00.h5', "conv/tb/width_"+str(k_width), np.array(gau0))
            h5_writein('result_07_01/methods/transition_'+site_no+'_v00.h5', "conv/ascat/width_"+str(k_width), np.array(sigconv))

            # for thawing transition
            date_npr_Emax = peakdate1[0][:, 1]  # the date when Conv of NPR reach maximum
            max_win_npr = date_npr_Emax[date_npr_Emax<70]
            max_date_npr = date_npr_Emax[(date_npr_Emax>0)&(date_npr_Emax<150)]
            i0_npr_Et = np.in1d(gau1[0], max_date_npr)
            max_trans = [gau1[0][i0_npr_Et], gau1[1][i0_npr_Et]]  # maximum during transition, only take value >0.5
            i_Et_05 = max_trans[1] > -1
            max_date_npr, Emax = max_trans[0][i_Et_05], max_trans[1][i_Et_05]
            Emax_npr.append([max_date_npr, Emax])  # date and value: maximum greater than 0.5
            max_dsm_npr, max_soil_npr, max_swe_npr = \
                np.zeros(max_date_npr.size), np.zeros(max_date_npr.size), np.zeros(max_date_npr.size)
            onset_01 = np.average(Emax_npr[0][0], weights=Emax_npr[0][1])

            # some in situ data
            i_dsm = 0
            for d0 in max_date_npr:
                i_window_tp0=np.argmin(np.abs(np.fix(sm[0]-d0-np.fix(3*k_width))))
                i_window_tp1=np.argmin(np.abs(np.fix(sm[0]-d0+np.fix(3*k_width))))
                i_window = [np.argmin(np.abs(np.fix(sm[0]-d0+np.fix(3*k_width)))), np.argmin(np.abs(np.fix(sm[0]-d0-np.fix(3*k_width))))]
                # i_window = [np.fix(sm[0]) == d0 + np.fix(3*k_width), np.fix(sm[0]) == d0 - np.fix(3*k_width)]
                dsm = (sm[1][i_window[1]] - sm[1][i_window[0]])
                dsoil = np.nanmean(tsoil[1][i_window[0]: i_window[1]+1])
                # dsm = (tsoil[1][i_window[0]] - tsoil[1][i_window[1]])
                dswe = (swe[1][i_window[1]] - swe[1][i_window[0]])

                max_dsm_npr[i_dsm], max_soil_npr[i_dsm], max_swe_npr[i_dsm] = dsm, dsoil, dswe
                i_dsm += 1
            dsoil_npr.append(max_soil_npr), dsm_npr.append(max_dsm_npr), dswe_npr.append(max_swe_npr)
            # save results during transition
            filedoc = prefix+'methods/'
            h5name1 = 'transition_'+site_no+'_v00.h5'
            narr1 = np.array([max_date_npr, Emax, max_dsm_npr, max_soil_npr, max_swe_npr])  # date, value, change of sm, mean t and change of swe
            h5_writein(filedoc+h5name1, ["thaw", "sig"+str(k_width)], narr1)
        '''
        # calculate the insitu difference of SMAP and ASCAT
        '''
        # sm5_tb = read_site.read_measurement(tb_pass, "Soil Moisture Percent -2in (pct)", site_no)
        # sm5_sig = read_site.read_measurement(sig_pass, "Soil Moisture Percent -2in (pct)", site_no)
        # t5_tb = read_site.read_measurement(tb_pass, "Soil Temperature Observed -2in (degC)", site_no)
        # t5_sig = read_site.read_measurement(sig_pass, "Soil Temperature Observed -2in (degC)", site_no)
        # swe_tb = read_site.read_measurement(tb_pass, 'Snow', site_no)
        # swe_sig = read_site.read_measurement(sig_pass, 'Snow', site_no)
        # sm5_change, sm5_diurnal, pass_diurnal = data_process.cal_obd([sm5_sig[0], sm5_sig[2]], [sm5_tb[0], sm5_tb[2]], sm5_sig[1], sm5_tb[1])
        # add new ascat results:  sigconv, sm, tsoil, swe, sigseries, ons_new

        # onset process
        # onset_new: ascat, tb, npr, site
        ons_new.append(ons0[0]), ons_new.append(ons0[1]), ons_new.append(ons1[0]), ons_new.append(ons1[1])
        ons_new.append(ons_site[0]), ons_new.append(ons_site[1])
        for ons in ons_tsoil:
            ons_new.append(ons)
        onset_file = 'result_07_01/txtfiles/result_txt/smap_ft_compare_%s.csv' % obs[1][1]
        onset_fromfile = np.loadtxt(onset_file, delimiter=',')
        onset_value = onset_fromfile[onset_fromfile[:, 10] == int(site_no), :]
        """
        plotting
        """
        # 1 general plot
        print ons_new
        print gau0[0].size, gau0[2].size
        plt_npr_gaussian_all([tbv0, tbh0, [gau0[0], gau0[2]]],  # row 1, tb
                             [[npr1[0], npr1[1]*100], [gau1[0], gau1[2]*100]],  # row 2, npr
                             [[sigseries[0], sigseries[1]],
                              [sigconv[0], sigconv[2]]],  # row 3 sigma
                             [['Soil moisture (%)', sm[0], sm[1]],  # row4 temp/moisture
                              # swe_date, swe_daily
                              ['Soil temperature (DegC)', tsoil[0], tsoil[1]]],
                             ['SWE (mm)', swe[0], swe[1]], onset_value[0], # row5 swe/percipitation, onset
                             figname=prefix+'all_plot_'+site_no+'_'+str(k_width)+'.png', size=(8, 10), xlims=[1, 366],
                             title=False, site_no=site_no, pp=precip, s_symbol='k.', tair=[tair_date, tair_daily], snow_plot=False)
                             #day_tout=day2, end_ax1=tb_onsetnew, end_ax2=npr_end, end_ax3=ascat_end)
        ons_new.append(int(site_no))
        onset_save.append(ons_new)


    # print onset_save
    # print prefix+'all_plot_'+site_no+'_'+str(k_width)+'.png'
    # os.system('cp result_07_01/*_7.png result_07_01/new_final/tb_interpolated/')
    # txtname = prefix+'all_sonet_new_'+obs[1]+str(k_width)+'.csv'
    # print np.array(onset_save)
    # np.savetxt(txtname, np.array(onset_save), fmt='%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d',
    #             header='ascatt, ascatf, tbt, tbf, nprt, nprf, stationt, stationf, stationid, tsoilt, tsoilf')
    # site_resultname = prefix+'site_onsets.csv'
    # np.savetxt(site_resultname, site_result, fmt='%.3f', header='t_thaw_in, sm_thaw, t_thaw_out, t_freeze_in, sm_freeze', delimiter=',')
    #
    # tb_1st = prefix+'tb_onsets_1st.csv'
    # np.savetxt(tb_1st, tb_result, fmt='%.3f', header='tb_thaw_st, tb_thaw_end, tb_freeze_st, tb_freeze_end, id', delimiter=',')
    # npr_2nd = prefix+'npr_onset_2nd.csv'
    # np.savetxt(npr_2nd, npr_result2nd, fmt='%.3f', header='npr_thaw_st, npr_thaw_end, npr_freeze_st, npr_freeze_end, id', delimiter=',')
    # ascat_2nd = prefix+'ascat_onset_2nd.csv'
    # np.savetxt(ascat_2nd, ascat_result2nd, fmt='%.3f', header='as_thaw_st, as_thaw_end, as_freeze_st, as_freeze_end, id', delimiter=',')


    # plt_npr_gaussian_all([tbv0, tbh0, gau0],  # row 1, tb
    #                      [npr1, gau1],  # row 2, npr
    #                      [[x_time, sigma0_np],
    #                       [x_time_np[ig2][g_size: -g_size], g_sig_valid]],  # row 3 sigma
    #                      [['Soil moisture (%)', sm5_date-365, sm5_daily],  # row4 temp/moisture
    #                       ['Soil temperature (DegC)', t5_date-365, t5_daily]],
    #                      ['SWE (mm)', swe_date-365, swe_daily], onset, figname=prefix+'all_plot_'+site_no+'.png')  # row 5 swe
    #
    # data_process.angle_compare([[x_time, sigma0_np],
    #                       [x_time_np[ig2][g_size: -g_size], g_sig_valid]], 'compare01')
    # txt2 = np.loadtxt(txtname, delimiter=',')
    # as_no = txt2[:, -1] < 0.5  # AS
    # x_time_mid, sig_mid = txt2[:, 0][as_no], txt2[:, 4][as_no]
    # g_size = 8
    # g_sig, ig2 = data_process.gauss_conv(sig_mid, sig=3)  # non-smoothed
    # g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
    #                 /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
    # data_process.angle_compare([[x_time_mid, sig_mid],
    #                       [x_time_mid[ig2][g_size: -g_size], g_sig_valid]], 'compare02')
    # plt_npr_gaussian_ASC(['E(t)', x_time_np[ig2][g_size: -g_size]+365, g_sig_valid],  # npr
    #                          # ['TB_v', t_v, var_npv],
    #                          ['Soil moisture (%)', sm5_date, sm5_daily],  # soil moisture
    #                          ['Soil temperature (DegC)', t5_date, t5_daily],
    #                          ['SWE (mm)', swe_date, swe_daily],
    #                          ['$\sigma^0$', x_time+365, sigma0_np],  # 511
    #                          fa_2=[], vline=onset,  # edge detection, !! t_v was changed
    #                          #ground_data=ons_site,
    #                          fe_2=[],
    #                          figname='onset_based_on_ASCAT'+site_no+'2.png', mode='annual')





