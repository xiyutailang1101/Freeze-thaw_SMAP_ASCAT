__author__ = 'xiyu'
import orb_difference
import plot_funcs
import numpy as np
import site_infos
import Read_radar
import read_site
import re, sys, os
import h5py
import data_process
import matplotlib.pyplot as plt
import csv
import log_write
import time
import peakdetect
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter, YearLocator

def read_h5(peroid):
    for s_no in ['1233', '2081', '2213', '2211', '2210', '2065', '947',
        '949', '950', '960', '962', '1090', '967', '968', '1089', '1175']:
        # , '1233', '2081', '2213', '2211', '2210', '2065', '947',
        # '949', '950', '960', '962', '1090', '967', '968', '1089', '1175'
        Read_radar.radar_read_main('_A_', s_no, peroid, 'vv')
    # sys.exit()


def main(site_no, period, sm_wind=17, mode='all', seriestype='sig'):

    """
    site_t:
        the type of site, if has 2 elements, refer to scan and sno
    """

    # '947', '948', '949', '950',
    ob = '_A_'
    passtime=[18]
    flag_target = []
    p_symbol = ['bo', 'b+', 'b^', 'ro', 'r+', 'r^', 'ko', 'k+', 'k^', 'go', 'g+', 'g^', 'b-o', 'b-+', 'b-^', 'r-o']
    n_symbol = 0
    typez = 2   # 1: air temperature and snow, 2: soil temperature and prp
    # site_t = 1  # 1: scan site
    """
    some global vars
    """
    # indicators
    ffv_all_wi, ffv_all_su, ffnpr_all_wi, ffnpr_all_su = [], [], [], []
    # measurements
    tsoil, tair, sm_5cm = [], [], []
    logger1 = log_write.initlog('log_Dec_Jan')
    loop_site = 0
    logger1.info('******************************this is interface line**************************************')
    #site_dic = {'sno_': ['947', '949', '950', '960', '962', '1090', '967', '968', '1175', '1177'],
    # site_dic = {'sno_': ['947', '1177'],
    #             'scan_': ['2081', '2213', '2210', '2065']}
    site_dic = {'sno_': ['968', '949', '1089', '967', '947', '950', '960', '962', '1090', '1175', '1177'],
                    'scan_': ['2212', '2081', '2213', '2210', '2065', '2211', '1233']}
                    #'scan_': ['2212']}
    # site_dic = {'sno_': ['947', '949', '950', '960', '962', '1090', '967', '968']}
    # site_dic = {'scan_': ['2081', '2213'], 'sno_': ['947', '1177', '968']}
    num_site_list = [len(v) for v in site_dic.values()]
    num_site = sum(num_site_list)
    # for i in site_t:
    #     all_site = site_dic[site_dic.keys()[i]]
    #     for site_no in all_site:
    for key_site in site_dic.keys():
        if site_no in site_dic[key_site]:
            site_type = key_site
            break
        else:
            site_type = 'none'
            continue
    if site_type == 'none':
        print 'dont have %s' % site_no
    # '947', '948', '949', '950', '960', '962', '1090
    site_file = site_type + site_no + '.txt'
    print 'now reading %s' % site_file
    """
    P1: reading station-has-been-found result
    """
    '''
    pay attention to the sigma data
    '''
    # sig, dtr, l = read_site.read_series('20150414', '20150601', site_no, ob)
    # sd, dtrd, ld = read_site.read_series('20150414', '20150601', site_no, ob='_D_')
    tbn, dtr2, l2 = read_site.read_series(period[0], period[1], site_no, ob, data='np', dat='tb_v_aft')
    tb_h, dtr_h, l_h = read_site.read_series(period[0], period[1], site_no, ob, data='np', dat='tb_h_aft')
    """
    P1-1 bi-linear interpolation to determine the RS data of the site
    """

    file_list, d_list = read_site.get_h5_list(period[0], period[1], site_no, ob)
    path = site_infos.get_data_root(site_no, '0901')
    site_path = path
    var_gm = []
    var_n = []
    var_rd_v, var_rd_h = [], []
    for single_file in file_list:  # size of file_list is the same with dtr (a array contains date string)
        path_name = site_path + single_file
        var_rd_v.append(data_process.interp_radius(path_name, site_no, disref=0.5))
        var_rd_h.append(data_process.interp_radius(path_name, site_no, dat='tb_h_aft'))


    """
    work of SMOS algorithm
    """
    var_npv = np.array(var_rd_v)  # remove the negative values
    var_npv[np.where(var_npv < -1)] = np.nan
    var_nph = np.array(var_rd_h)
    var_nph[np.where(var_nph < -1)] = np.nan
    t_v = data_process.get_doy(dtr2)
    t_h = data_process.get_doy(dtr_h)
    ffv = 300.0 - var_npv

    # remove temporal change
    #edge_seriesv = data_process.rm_odd(var_npv)
    #edge_seriesh = data_process.rm_odd(var_nph)
    edge_seriesv = var_npv
    edge_seriesh = var_nph

    ffnpr = (np.array(edge_seriesv) - np.array(edge_seriesh))/(np.array(edge_seriesv) + np.array(edge_seriesh))

    len_origin = t_h.shape[0]
    i_origin = range(0, len_origin, 1)
    i_val = i_origin[(sm_wind-1)/2: t_h.size-(sm_wind+1)/2 + 1]
    valid_dateh = t_h[i_val]  # date --> smooth[valid]
    valid_datev = t_v[i_val]
    ffnpr_s, valid = data_process.n_convolve2(ffnpr, sm_wind)
    ffnpr_s = ffnpr_s[valid]

    # read site data
    y2_empty = np.zeros(tbn.shape[1]) + 1
    stats_cp15_2, tair = read_site.read_sno(site_file, "Air Temperature Observed (degC)", site_no)  # air tmp
    tair_daily, tair_date = data_process.cal_emi(tair, y2_empty, dtr_h, hrs=passtime)
    if all(tair_daily == -99):
        stats_cp15_2, tair = read_site.read_sno(site_file, "Air Temperature Average (degC)", site_no)  # air tmp
        tair_daily, tair_date = data_process.cal_emi(tair, y2_empty, dtr_h, hrs=passtime)

    # # comment part I site read and reference npr
    # stats_cp15_2, swe = read_site.read_sno(site_file, "snow", site_no, field_no=-1)  # snow w e
    # swe_daily, swe_date = data_process.cal_emi(swe, y2_empty, dtr_h, hrs=passtime)
    # stat_prp, prp = read_site.read_sno(site_file, "Precipitation Accumulation (mm)", site_no)  # precip.
    # prp_daily, prp_date = data_process.cal_emi(prp, y2_empty, dtr_h, hrs=passtime)
    # prp_daily[np.where(prp_daily == -99)] = 0
    # stats_cp4, sm_5 = read_site.read_sno(site_file, "Soil Moisture Percent -2in (pct)", site_no)  # moisture
    # stats_cp5, t_5 = read_site.read_sno(site_file, "Soil Temperature Observed -2in (degC)", site_no)
    # t_5_daily, t_5_date = data_process.cal_emi(t_5, y2_empty, dtr_h, hrs=passtime)  # T soil
    # sm_daily, sm_date = data_process.cal_emi(sm_5, y2_empty, dtr_h, hrs=passtime)
    # t_5_val = t_5_daily[i_val]
    # sm_val = sm_daily[i_val]
    # it_wi = np.where(((t_5_val<0) & (sm_val<15))|(t_5_val<-1))
    # ffnpr_wi = ffnpr_s[it_wi]
    # it_su = np.where((t_5_val > 10) & (sm_val < 40))
    # ffnpr_su = ffnpr_s[it_su]
    # # calculate the winter and summer reference
    # ff_su, ff_wi = np.nanmean(ffnpr_su), np.nanmean(ffnpr_wi)
    # ffnpr_all_su.append(ff_su), ffnpr_all_wi.append(ff_wi)
    # tair_daily = read_site.rm_empty_measure(tair_daily)
    # swe_daily = read_site.rm_empty_measure(swe_daily)
    # t_5_daily = read_site.rm_empty_measure(t_5_daily)
    # sm_daily = read_site.rm_empty_measure(sm_daily)
    # prp_daily = read_site.rm_empty_measure(prp_daily)
    # tb_smoothv, valid_v = data_process.n_convolve2(var_npv, sm_wind)
    # tb_smoothh, valid_h = data_process.n_convolve2(var_nph, sm_wind)
    # tair_s, valid_air = data_process.n_convolve2(tair_daily, sm_wind)

    # # comment part II old plot
    # fig = plt.figure(figsize=[8, 8])
    # ax0 = fig.add_subplot(511)  # tb
    # ax0.plot(t_v, var_npv, 'bo', markersize=3)
    # plot_funcs.plt_more(ax0, t_h, var_nph, marksize=3)
    # ax1 = fig.add_subplot(512)  # npr
    # ax1.plot(valid_dateh, ffnpr_s)
    # plot_funcs.plt_more(ax1, valid_dateh[it_wi], ffnpr_wi, symbol='k.')
    # plot_funcs.plt_more(ax1, valid_dateh[it_su], ffnpr_su, symbol='r.')
    # ax1.set_ylim([0.005, 0.050])
    # # fig.text(0.25, 0.75, str(ff_su))
    # # fig.text(0.75, 0.75, str(ff_wi))
    # #
    # ax2 = fig.add_subplot(513)  # sm
    # ax2.plot(sm_date, sm_daily)
    # ax2.set_ylim([0, 100])
    # if typez == 1:
    #     ax3 = fig.add_subplot(514)  # air temp
    #     ax3.plot(tair_date, tair_daily)
    #     ax3.set_ylabel('t_air')
    #     ax4 = fig.add_subplot(515)  # swe
    #     ax4.plot(swe_date, swe_daily)
    #     ax4.set_ylabel('SWE (mm)')
    #     ax4.set_ylim([-20, 250])
    # elif typez == 2:
    #     ax3 = fig.add_subplot(514)  # tmp
    #     ax3.plot(t_5_date, t_5_daily)
    #     ax3.set_ylabel('t_soil')
    #     ax3.set_ylim([-10, 25])
    #     ax4 = fig.add_subplot(515)  # prp
    #     ax4.plot(swe_date, swe_daily)
    #     ax4.set_ylabel('precipitation')
    #     ax4.set_ylim([0, 1000])
    # # axis_list = [ax0, ax1, ax2, ax3, ax4]
    # # for axs in axis_list:
    # #     axs.set_xlim([100, 550])
    # plt.tight_layout()
    # plt.savefig('test_plot_airt'+site_no+ob+'.png', dpi=120)
    # plt.close()  # commented

    # deal the gap
    # gap_e = dtr_h.index('20160902')  # end index of gap
    #
    # ons1, g_npr1, ig1 = data_process.edge_detection(ffnpr[0:gap_e+1], t_h[0: gap_e+1])
    # ons2, g_npr2, ig2 = data_process.edge_detection(ffnpr[gap_e:], t_h[gap_e:], typez='fr')
    # t_h1 = t_h[ig1][g_size: -g_size]
    # t_h2 = t_h[gap_e:][ig2][g_size: -g_size]
    # t_orgin = np.append(t_h1, t_h2)
    # edges = np.append(g_npr1, g_npr2)
    # onset = np.append(ons1, ons2, axis=0)

    # remove some odd points:
    if seriestype=='tb':
        edge_series = var_npv
        t_series = t_v
        # odd_point = data_process.rm_temporal(edge_series)
        # edge_series[odd_point] = np.nan
    else:
        edge_series = ffnpr
        t_series = t_h
    g_size = 8
    for kwidth in [2]:
        g_npr, i_gaussian = data_process.gauss_conv(edge_series, sig=kwidth)  # option: ffnpr-t_h; var_npv-t_h
        if type(g_npr) is int:
            continue
        g_npr_valid_n = (g_npr[g_size: -g_size] - np.nanmin(g_npr[g_size: -g_size]))\
                        /(np.nanmax(g_npr[g_size: -g_size]) - np.nanmin(g_npr[g_size: -g_size]))   # normalized
        max_gnpr, min_gnpr = peakdetect.peakdet(g_npr[g_size: -g_size], 5e-4, t_series[i_gaussian][g_size: -g_size])
        onset = data_process.find_inflect(max_gnpr, min_gnpr, typez='annual', typen=seriestype)  # !!
        # data_process.output_csv(['site', 'onset1', 'onset2', 'onset3', 'onset4', 'sigma'], [site_no, kwidth], onset[:, 1])


    return [t_v-365, var_npv], [t_h-365, var_nph], [t_series-365, ffnpr], \
           [t_series[i_gaussian][g_size: -g_size]-365, g_npr_valid_n], [onset[0], onset[1]]
    #data_process.edge_detection(ffnpr, t_h, typez=all, wind=[5, 7, 11, 15, 17])
    site_info = site_infos.change_site(site_no)
    loc = [site_info[1], site_info[2]]
    logger1.info('\n onset based on npr in %s, %s, %s' % (site_type+site_no, loc[0], loc[1]))
    plt_npr_gaussian(['TB_v', t_v, var_npv],  # npr
                     # ['TB_v', t_v, var_npv],
                     ['Soil moisture (%)', sm_date, sm_daily],  # soil moisture
                     ['Soil temperature (DegC)', t_5_date, t_5_daily],
                     ['SWE (mm)', swe_date, swe_daily],
                     ['TBv', t_v, var_npv],
                     fa_2=[None, t_v[i_gaussian][g_size: -g_size], g_npr_valid_n], vline=onset,  # edge detection, !! t_v was changed
                     # fa_2=[None, t_orgin, edges], vline=onset,
                     fe_2=['TBh', t_h, var_nph],
                     figname='onset_based_on_npr'+site_no+'2.png', mode=mode)
    plt_npr_gaussian_all(['TB_v', t_v, var_npv],  # npr
                     # ['TB_v', t_v, var_npv],
                     ['Soil moisture (%)', sm_date, sm_daily],  # soil moisture
                     ['Soil temperature (DegC)', t_5_date, t_5_daily],
                     ['SWE (mm)', swe_date, swe_daily], ['TBv', t_v, var_npv],
                     fa_2=[None, t_v[i_gaussian][g_size: -g_size], g_npr_valid_n], vline=onset,  # edge detection, !! t_v was changed
                     # fa_2=[None, t_orgin, edges], vline=onset,
                     fe_2=['TBh', t_h, var_nph],
                     figname='onset_based_on_npr'+site_no+'2.png', mode=mode)

    # smoothed
    # gap_e = np.where(valid_dateh == 611)[0][0]
    # ons1, g_npr1, ig1 = data_process.edge_detection(ffnpr_s[0:gap_e+1], valid_dateh[0: gap_e+1], minz=5e-5)
    # ons2, g_npr2, ig2 = data_process.edge_detection(ffnpr_s[gap_e:], valid_dateh[gap_e:], typez='fr', minz=5e-5)
    # t_h1 = valid_dateh[ig1][g_size: -g_size]
    # t_h2 = valid_dateh[gap_e:][ig2][g_size: -g_size]
    # t_smoothed = np.append(t_h1, t_h2)
    # edges = np.append(g_npr1, g_npr2)
    # onset_s = np.append(ons1, ons2, axis=0)
    data_process.edge_detection(ffnpr, t_h, typez=all, wind=[5, 7, 11, 15, 17], figname='test_average'+site_no)
    g_nprs, i_gaussians = data_process.gauss_conv(ffnpr_s)  # ffnpr_s, valid_dateh
    g_nprs_valid_n = (g_nprs[g_size: -g_size] - np.nanmin(g_nprs[g_size: -g_size]))\
                     /(np.nanmax(g_nprs[g_size: -g_size]) - np.nanmin(g_nprs[g_size: -g_size]))
    max_gnprs, min_gnprs = peakdetect.peakdet(g_nprs[g_size: -g_size], 5e-5, valid_dateh[i_gaussians][g_size: -g_size])
    onset_s = data_process.find_inflect(max_gnprs, min_gnprs, typez='annual')
    # data_process.output_csv(['site', 'average day' 'onset1', 'onset2', 'onset3', 'onset4'], [site_no, wind], onset_s[:, 1], fname='csv_smoothed')

    plt_npr_gaussian(['NPR', valid_dateh, ffnpr_s],
                     ['Soil moisture (%)',sm_date, sm_daily],
                     ['Soil temperature (DegC)', t_5_date, t_5_daily],
                     ['SWE (mm)', swe_date, swe_daily], ['TBv', valid_datev, tb_smoothv[valid_v]],
                     fa_2=[None, valid_dateh[i_gaussians][g_size: -g_size], g_nprs_valid_n], vline=onset_s,
                     # fa_2=[None, t_smoothed, edges], vline=onset_s,
                     fe_2=['TBh', valid_dateh, tb_smoothh[valid_h]], figname='onset_based_on_nprs'+site_no+'2.png',
                     mode=mode)


    onset0 = onset_s
    onset0[2, 1] -= 365
    onset0[3, 1] -= 365
    logger1.info('onset smoothed \n'+str(onset0))
    # long term t_soil
    # sys.exit()
    # using the referenced npr
    # continue
    with open('norm_factor.txt', 'rb') as f1:
            f_wi_su = csv.reader(f1)
            npr_ref = np.array([])
            r_num = 0
            for row in f_wi_su:
                tp = float(row[0])  # row No., 0 is winter, 1 is summer
                npr_ref = np.append(npr_ref, tp)
                r_num += 1

    # reference based algorithm
    d_state = np.array(npr_ref[0]) - np.array(npr_ref[1])
    ff_rel = (ffnpr_s - npr_ref[0]) * 1.0/(0-d_state)
    ff_rel2 = (ffnpr - npr_ref[0]) * 1.0/(0-d_state)
    # plot ff_rel series
    if loop_site == 0:
        ax_rel, l_rel = plot_funcs.pltyy(valid_dateh, ff_rel, 'test_relative'+site_no+ob+'2.png', 'relative frost factor')
        # ax_rel2, l_rel2 = plot_funcs.pltyy(t_h, ff_rel2, 'test_relative'+site_no+ob+'nosmth2.png', 'relative frost factor')
        l_all = l_rel
    plot_funcs.pltyy(valid_dateh, ff_rel, 'test_relative'+site_no+ob+'2.png', 'relative frost factor')
    plt.close()
    if loop_site > 0:
        print loop_site
        l_all = plot_funcs.plt_more(ax_rel[0], valid_dateh, ff_rel, fname='test_relative_together00'+site_no+ob+'1', symbol=p_symbol[loop_site], line_list=l_rel)


    if loop_site == num_site - 1:
        plt.legend(l_all, all_site[0: ], bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=len(all_site)/2, mode="expand")
        plt.savefig('test_relative_together_le.png', dpi=120)
    loop_site += 1

    # calculate the 20%-based factors for winter and summer
    sort_wi = np.sort(ffnpr_all_wi)
    sz = sort_wi.size
    # p20_wi = np.nanmean(sort_wi[sz*1/10: sz*3/10 + 1])
    p20_wi = np.nanmin(sort_wi)
    sort_su = np.sort(ffnpr_all_su)
    sz = sort_su.size
    p20_su = np.nanmean(sort_su)
    # p20_su = np.nanmean(sort_su[sz*7/10: sz*9/10 + 1])
    factors = np.array([p20_wi, p20_su])
    # factors = np.append(np.nanmin(ffnpr_all_wi), np.nanmax(ffnpr_all_su))
    np.savetxt('norm_factor.txt', factors, delimiter=',')
    factors_all = np.array([ffnpr_all_wi, ffnpr_all_su])
    np.savetxt('all_reff_'+site_type+'.txt', factors_all, delimiter=',')


def plot_ref(plotname, ref_no):
    """

    :param plotname:
    :param ref_no: 0: winter, 1: summer
    :return:
    """
    ref_sno = read_site.read_ref('all_reff_sno_.txt')
    ref_scan = read_site.read_ref('all_reff_scan_.txt')
    ref_scan.shape = 2, -1
    ref_sno.shape = 2, -1
    ref_all = np.concatenate((ref_sno, ref_scan), axis=1)
    print ref_all.shape
    print ref_all
    tick_site = ['947', '949', '950', '960', '962', '1090', '967', '968',
                 '1089', '1175','2081', '2213', '2211', '2210', '2065']
    axis1, ls1 = plot_funcs.pltyy(range(0, len(tick_site)), ref_all[ref_no[0], :], plotname, 'NPR Reference (%)', label_x='Site No.', clip_on=False)
    ls2, = axis1[0].plot(range(0, len(tick_site)), ref_all[ref_no[1], :], 'ro', clip_on=False)
    plt.locator_params(axis='x', nbins=len(tick_site))
    axis1[0].set_xticklabels(tick_site)
    plt.savefig(plotname+'.png', dpi=120)


def read_alaska(start_date, duration):
    predir = './result_05_01/SMAP_AK/'
    orbit = '_A_'
    site_info = ['alaska', 63.25, -151.5]
    tb_attr_name = ['/cell_tb_v_aft', '/cell_tb_h_aft', '/cell_tb_qual_flag_v_aft', '/cell_tb_qual_flag_h_aft',
                    "/cell_lat", "/cell_lon"]
    TBdir = '/media/Seagate Expansion Drive/Data_Xy/CloudMusic/'
    TB_date_list = sorted(os.listdir(TBdir))
    TB_date = []
    TB_date2, date_ind = [], -1  # use to find yyyy.mm.dd format documents
    # Time list of radar data
    time_list = []
    time_ini = 0
    for time_dot in TB_date_list:
        if time_dot == start_date:
            time_ini = 1
        if time_ini > 0:
            TB_date.append(time_dot.replace(".", ""))
            TB_date2.append(time_dot)
    for time_str in TB_date:
        if date_ind < duration:
            print '%s is reading' % TB_date2[date_ind+1]
        else:
            continue
        date_ind += 1
        # 01 list with radiometer files
        tb_file_list = []  # tb files with different orbits
        tb_ob_list = []  # orbits of tb files
        TB_file_No = TB_date_list.index(TB_date2[date_ind])
        print 'the order of tb_file is :', TB_file_No, 'which is: ', TB_date_list[TB_file_No]
        for tbs in os.listdir(TBdir + TB_date_list[TB_file_No]):
            if tbs.find('.iso') == -1 and tbs.find(orbit) != -1 and tbs.find('.qa') == -1:  # Using ascend orbit data, iso is metadata we don't need, qa neither
                tb_file_list.append(tbs)
                p_underline = re.compile('_')
                tb_ob_list.append(p_underline.split(tbs)[3])
        obset = set(tb_ob_list)
        tb_name_list = []
        for orb_no in obset:  # ob_set is a set of orbit numbers
            list_orb = []
            for tb_file in tb_file_list:  # tb_file_list is a list of all tb files (Line 175)
                if re.search(orb_no, tb_file):
                    list_orb.append(tb_file)
            tb_name_list.append(sorted(list_orb)[-1])
        proj_g = 'Global_Projection'
        proj_n = 'North_Polar_Projection'
        count_tb = 0  # remains 1 if no tb file was found
        d_att = {'_key': 'dict for global tb'}  # create dictionary for store tb data temporally
        for proj in [proj_n]:  # two projection method:
            count_tb = 0
            for namez in tb_attr_name:
                d_att[namez] = np.array([])
            for tb_name in tb_name_list:
                fname = TBdir + TB_date_list[TB_file_No] + '/' + tb_name
                [Site_ind, status1] = Read_radar.return_ind(fname, site_info, 'tbak', prj=proj, thtb=[21.54, 8.58])
                if status1 == -1:  # -1 means not found with specified lat and lon
                    a00 = 1
                else:
                    count_tb = 1
                    print 'the station is inside of this swath: ', tb_name
                    hf = h5py.File(fname, 'r')
                    for att in tb_attr_name:
                        tb_att, status = Read_radar.read_tb(hf, att, Site_ind, prj=proj)
                        d_att[att] = np.append(d_att[att], tb_att)
        # name x: lon, y: lat, z: npr
        v_tb, h_tb = d_att["/cell_tb_v_aft"], d_att["/cell_tb_h_aft"]  # vertical & horizontal
        # vh_pr = (v_tb-h_tb)/(v_tb+h_tb)
        x, y = d_att["/cell_lon"], d_att["/cell_lat"]
        np.save(predir+'txt_xyz'+TB_date_list[TB_file_No], np.transpose(np.array([x, y, v_tb, h_tb])))


def comp_plot(fa, fb, fc, fd, fe, fa_2=[], fb_2=[], fb_3=[], figname= 'test_comp_plot.png'):
    if fa.keys()[0] == 'TB (K)':
        fig = plt.figure(figsize=[8, 8])
        ax0 = fig.add_subplot(511)  # tb
        ax0.plot(fa.values[0], fa.values[1], 'bo', markersize=3)
        if fa_2:
            plot_funcs.plt_more(ax0, fa_2.values[0], fa_2.values[1], marksize=3)

        # ax1.set_ylim([200, 280])
        ax1 = fig.add_subplot(512)  # npr
        ax1.plot(fb[0], fb[1], linewidth=2.0)
        if fb_2:
            plot_funcs.plt_more(ax1, fb_2.values[0], fb_2.values[1], symbol='k.')
        if fb_3:
            plot_funcs.plt_more(ax1, fb_2.values[0], fb_2.values[1], symbol='r.')
        if fb.keys == 'NPR':
            ax1.set_ylim([0.005, 0.050])
            # fig.text(0.25, 0.75, str(ff_su))
            # fig.text(0.75, 0.75, str(ff_wi))
        if fc:
            ax2 = fig.add_subplot(513)  # sm
            ax2.plot(fc.values[0], fc.vlaue[1], linewidth=2.0)
        # ax2.set_ylim([0, 100])
        if fd:
            ax3 = fig.add_subplot(514)  # air temp
            ax3.plot(fd.values[0], fd.values[1], linewidth=2.0)
            ax3.set_ylabel(fd.keys()[0])
        if fe:
            ax4 = fig.add_subplot(515)  # swe
            ax4.plot(fe.values[0], fe.values[1], linewidth=2.0)
            ax4.set_ylabel(fe.keys()[0])
        # ax4.set_ylim([-20, 250])
        axis_list = [ax0, ax1, ax2, ax3, ax4]
        for axs in axis_list:
            axs.set_xlim([100, 550])
        plt.tight_layout()
        plt.savefig(figname, dpi=120)
        plt.close()
    elif fa.keys()[0] == 'NPR':
        axs = []
        fig = plt.figure(figsize=[8, 8])
        ax0 = fig.add_subplot(511)  # tb
        axs.append(ax0)
        plot_funcs.pltyy(fa.values[0], fa.values[1], 'test_comp2', fa.keys,
                         t2=fa_2.values[0], s2=fa_2.values[1], label_y2=fa_2.keys()[0], symbol=['k-o', 'k-+'],
                         handle=[fig, ax0])

        ax1 = fig.add_subplot(512)
        axs.append(ax1)
        ax1.plot(fb[0], fb[1])
        if fc:
            ax2 = fig.add_subplot(513)  # sm
            axs.append(ax2)
            ax2.plot(fc.values[0], fc.vlaue[1])
        for axs in axs:
            axs.set_xlim([100, 550])
        plt.tight_layout()
        plt.savefig(figname, dpi=120)
        plt.close()


def plt_npr_gaussian(fa, fb, fc, fd, fe, fa_2=[], fe_2=[], figname= 'test_comp_plot.png', vline=False, mode='all'):
    axs = []
    fig = plt.figure(figsize=[8, 8])
    if fe:
        # plt.legend(l2, [y_main, ipt_key])
        ax4 = fig.add_subplot(511)  # swe
        axs.append(ax4)
        if mode=='annual':
            fe[1] = fe[1] - 365
            fe_2[1] = fe_2[1] - 365
        l4, = ax4.plot(fe[1], fe[2], 'bo', markersize=2)
        l4_le = plot_funcs.plt_more(ax4, fe_2[1], fe_2[2], marksize=2, line_list=[l4])
        ax4.locator_params(axis='y', nbins=6)
        ax4.set_ylabel('T$_B$ (K)')
        ax4.legend(l4_le, ['T$_{BV}$', 'T$_{BH}$'], loc=2, prop={'size': 10})
        # ax4.text(0.85, 0.85, '(a)')
        fig.text(0.85, 0.21, '(e)')
        fig.text(0.85, 0.37, '(d)')
        fig.text(0.85, 0.53, '(c)')
        fig.text(0.85, 0.705, '(b)')
        fig.text(0.85, 0.87, '(a)')
        if mode=='annual':
            ax4.annotate(str(vline[0]), xy=(vline[0], 260))
            ax4.annotate(str(vline[1]), xy=(vline[1], 260))
    ax0 = fig.add_subplot(512)  # tb
    axs.append(ax0)
    if mode=='annual':
        fa[1] = fa[1] - 365
        fa_2[1] = fa_2[1] - 365
    _, l0 = plot_funcs.pltyy(fa[1], fa[2], 'test_comp2', fa[0],
                     t2=fa_2[1], s2=fa_2[2], symbol=['k', 'g-'], label_y2= 'E(t)',
                     handle=[fig, ax0], nbins2=6)
    ax0.locator_params(axis='y', nbins=6)
    #ax0.set_ylim([0, 0.06])
    ax0.legend(l0, [fa[0], 'E(t)'], loc=2, prop={'size': 8})

    ax1 = fig.add_subplot(513)  # sm
    axs.append(ax1)
    if mode=='annual':
        fb[1] = fb[1] - 365
    ax1.plot(fb[1], fb[2], 'k', linewidth=2.0)
    ax1.set_ylabel('VWC (%s, %%)' % '$m^3/m^3$')
    ax1.set_ylim([0, 100])
    if fc:
        ax2 = fig.add_subplot(514)  # t soil
        axs.append(ax2)
        if mode=='annual':
            fc[1] = fc[1] - 365
        ax2.plot(fc[1], fc[2], 'k', linewidth=2.0)
        if mode=='annual':
            ax2.set_xlim([0, 400])
        ax2.set_ylim([-20, 25])
        ax2.locator_params(axis='y', nbins=6)
        ax2.set_ylabel('T$_{soil}$ ($^\circ$C)')
        ax2.axhline(ls=':', lw=1.5)
    if fd:
        ax3 = fig.add_subplot(515)  # swe
        axs.append(ax3)
        if mode=='annual':
            fd[1] = fd[1] - 365
        ax3.plot(fd[1], fd[2], 'k', linewidth=2.0)
        ax3.set_ylim([0, 200])
        if figname == 'onset_based_on_npr22132.png':
            ax3.set_ylim([0, 200])
        ax3.locator_params(axis='y', nbins=6)
        ax3.set_ylabel('SWE (mm)')
        ax3.set_xlabel('Day of year')
    # for ax in axs:
    #     ax.set_xlim([100, 550])
    if vline is not False:
        ax_count = 0
        for ax in axs:
            ax.set_xlim([0, 350])
            ax_count += 1
            if type(vline) is list:
                ax.axvline(x=vline[0], color='k', ls='--')
                ax.axvline(x=vline[1], color='k', ls='-.')
                continue
            ax.axvline(x=vline[0, 1], color='k', label=repr(vline[0, 1]), ls='--')
            ax.axvline(x=vline[1, 1], color='k', label=repr(vline[1, 1]), ls='-.')
            ax.axvline(x=vline[2, 1], color='k', label=repr(vline[1, 1]), ls='--')
            ax.axvline(x=vline[3, 1], color='k', label=repr(vline[1, 1]), ls='-.')
            # ax.xaxis.set_minor_locator(months)
            # ax.xaxis.set_minor_formatter(monthsFmt)
            # ax.xaxis.set_major_locator(years)
            # ax.xaxis.set_major_formatter(yearsFmt)
            # ax.locator_params(axis='x', nbins=16)
            tick_num = np.array([50, 100, 150, 200, 250, 300, 350, 365, 415, 465, 515, 565, 615, 665, 715], dtype=int)
            ax.xaxis.set_ticks(tick_num)
            labels = [item.get_text() for item in ax.get_xticklabels()]
            n = 0
            for label in labels:
                if tick_num[n] == 50:
                    if ax_count == 5:
                        labels[n] = "50\nYear'15        "
                    else:
                        labels[n] = "50"
                elif tick_num[n] == 350:
                    labels[n] = ' '
                elif tick_num[n] == 365:
                     labels[n] = "365\n              Year'16"
                elif tick_num[n] == 415:
                    labels[n] = repr(tick_num[n]-365)
                elif tick_num[n] > 415:
                    labels[n] = repr(tick_num[n]-365)
                else:
                    labels[n] = repr(tick_num[n])
                n += 1
            # labels[0] = 'Year\n2015'
            # labels[1] = '100'
            # labels[2] = '150'
            # labels[3] = '200'
            # labels[4] = '250'
            # labels[5] = '300'
            # labels[6] = ''
            # labels[7] = 'Year\n2016'
            # labels[8] = '50'
            # labels[9] = '100'
            # labels[10] = '150'
            # labels[11] = '200'

            ax.set_xticklabels(labels)
    # plt.tight_layout()
    plt.savefig(figname, dpi=120)
    plt.close()


def plt_npr_gaussian_ASC(fa, fb, fc, fd, fe, fa_2=[], fe_2=[], figname= 'test_comp_plot.png', vline=False, mode='all'):
    axs = []
    fig = plt.figure(figsize=[8, 8])
    if fe:
        # plt.legend(l2, [y_main, ipt_key])
        ax4 = fig.add_subplot(511)  # swe
        axs.append(ax4)
        if mode=='annual':
            fe[1] = fe[1] - 365
            #fe_2[1] = fe_2[1] - 365
        ax4.plot(fe[1], fe[2], '-o', markersize=1)
        # l4_le = plot_funcs.plt_more(ax4, fe_2[1], fe_2[2], marksize=2, line_list=[l4])
        ax4.locator_params(axis='y', nbins=6)
        ax4.set_ylabel('$\sigma^0$ (dB)')
        #ax4.legend([l4], [fe[0]], loc=2, prop={'size': 10})
        # ax4.text(0.85, 0.85, '(a)')
        fig.text(0.85, 0.21, '(e)')
        fig.text(0.85, 0.37, '(d)')
        fig.text(0.85, 0.53, '(c)')
        fig.text(0.85, 0.705, '(b)')
        fig.text(0.85, 0.87, '(a)')
        if mode=='annual':
            ax4.annotate(str(vline[0]), xy=(vline[0], 260))
            ax4.annotate(str(vline[1]), xy=(vline[1], 260))
    ax0 = fig.add_subplot(512)  # tb
    axs.append(ax0)
    if mode=='annual':
        fa[1] = fa[1] - 365
        #fa_2[1] = fa_2[1] - 365
    _, l0 = plot_funcs.pltyy(fa[1], fa[2], 'test_comp2', fa[0],
                     symbol=['g-'], label_y2= 'E(t)',
                     handle=[fig, ax0], nbins2=6)
    ax0.locator_params(axis='y', nbins=6)
    #ax0.set_ylim([0, 0.06])
    ax0.legend(l0, [fa[0], 'E(t)'], loc=2, prop={'size': 8})

    ax1 = fig.add_subplot(513)  # sm
    axs.append(ax1)
    if mode=='annual':
        fb[1] = fb[1] - 365
    ax1.plot(fb[1], fb[2], 'k', linewidth=2.0)
    ax1.set_ylabel('VWC (%s, %%)' % '$m^3/m^3$')
    ax1.set_ylim([0, 100])
    if fc:
        ax2 = fig.add_subplot(514)  # t soil
        axs.append(ax2)
        if mode=='annual':
            fc[1] = fc[1] - 365
        ax2.plot(fc[1], fc[2], 'k', linewidth=2.0)
        if mode=='annual':
            ax2.set_xlim([0, 400])
        ax2.set_ylim([-20, 25])
        ax2.locator_params(axis='y', nbins=6)
        ax2.set_ylabel('T$_{soil}$ ($^\circ$C)')
        ax2.axhline(ls=':', lw=1.5)
    if fd:
        ax3 = fig.add_subplot(515)  # swe
        axs.append(ax3)
        if mode=='annual':
            fd[1] = fd[1] - 365
        ax3.plot(fd[1], fd[2], 'k', linewidth=2.0)
        ax3.set_ylim([0, 200])
        if figname == 'onset_based_on_npr22132.png':
            ax3.set_ylim([0, 200])
        ax3.locator_params(axis='y', nbins=6)
        ax3.set_ylabel('SWE (mm)')
        ax3.set_xlabel('Day of year')
    # for ax in axs:
    #     ax.set_xlim([100, 550])
    if vline is not False:
        ax_count = 0
        for ax in axs:
            ax.set_xlim([0, 350])
            ax_count += 1
            if type(vline) is list:
                ax.axvline(x=vline[0], color='k', ls='--')
                ax.axvline(x=vline[1], color='k', ls='-.')
                continue
            ax.axvline(x=vline[0, 1], color='k', label=repr(vline[0, 1]), ls='--')
            ax.axvline(x=vline[1, 1], color='k', label=repr(vline[1, 1]), ls='-.')
            ax.axvline(x=vline[2, 1], color='k', label=repr(vline[1, 1]), ls='--')
            ax.axvline(x=vline[3, 1], color='k', label=repr(vline[1, 1]), ls='-.')
            # ax.xaxis.set_minor_locator(months)
            # ax.xaxis.set_minor_formatter(monthsFmt)
            # ax.xaxis.set_major_locator(years)
            # ax.xaxis.set_major_formatter(yearsFmt)
            # ax.locator_params(axis='x', nbins=16)
            tick_num = np.array([50, 100, 150, 200, 250, 300, 350, 365, 415, 465, 515, 565, 615, 665, 715], dtype=int)
            ax.xaxis.set_ticks(tick_num)
            labels = [item.get_text() for item in ax.get_xticklabels()]
            n = 0
            for label in labels:
                if tick_num[n] == 50:
                    if ax_count == 5:
                        labels[n] = "50\nYear'15        "
                    else:
                        labels[n] = "50"
                elif tick_num[n] == 350:
                    labels[n] = ' '
                elif tick_num[n] == 365:
                     labels[n] = "365\n              Year'16"
                elif tick_num[n] == 415:
                    labels[n] = repr(tick_num[n]-365)
                elif tick_num[n] > 415:
                    labels[n] = repr(tick_num[n]-365)
                else:
                    labels[n] = repr(tick_num[n])
                n += 1
            # labels[0] = 'Year\n2015'
            # labels[1] = '100'
            # labels[2] = '150'
            # labels[3] = '200'
            # labels[4] = '250'
            # labels[5] = '300'
            # labels[6] = ''
            # labels[7] = 'Year\n2016'
            # labels[8] = '50'
            # labels[9] = '100'
            # labels[10] = '150'
            # labels[11] = '200'

            ax.set_xticklabels(labels)
    plt.savefig(figname, dpi=120)
    plt.close()


def plt_npr_gaussian_all(tb, npr, sigma, soil, snow, onset, figname='all_plot_test0420'):
    """
    plt_npr_gaussian_all([tbv0, tbh0, gau0],  # row 1, tb
                         [npr1, gau1],  # row 2, npr
                         [[x_time+365, sigma0_np],
                          [x_time_np[ig2][g_size: -g_size]+365, g_sig_valid]],  # row 3 sigma
                         [['Soil moisture (%)', sm5_date, sm5_daily],  # row4 temp/moisture
                          ['Soil temperature (DegC)', t5_date, t5_daily]],
                          ['SWE (mm)', swe_date, swe_daily]) # row 5 swe
    :param tb:
    :param npr:
    :param sigma:
    :param soil:
    :param snow:
    :param onset:
    :return:
    """
    axs = []
    fig = plt.figure(figsize=[8, 8])
    # row 1 tb
    ax1 = fig.add_subplot(511)  # tb
    axs.append(ax1)
    # l1, = ax1.plot(tb[0][0], tb[0][1], 'bo', markersize=2)
    _, l1 = plot_funcs.pltyy(tb[0][0], tb[0][1], 'test_comp2', 'T$_B$ (K)',
                             t2=tb[2][0], s2=tb[2][1], label_y2= 'E(t)',
                             symbol=['k.', 'g-'],
                             handle=[fig, ax1], nbins2=6)  # plot tbv
    l1_le = plot_funcs.plt_more(ax1, tb[1][0], tb[1][1], line_list=[l1])
    ax1.locator_params(axis='y', nbins=6)
    ax1.set_ylabel('T$_B$ (K)')
    ax1.legend([l1_le[0][0], l1_le[1]], ['T$_{BV}$', 'T$_{BH}$'], loc=4, prop={'size': 8})
    #ax4.legend([l4], [fe[0]], loc=2, prop={'size': 10})
    # ax4.text(0.85, 0.85, '(a)')
    fig.text(0.85, 0.21, '(e)')
    fig.text(0.85, 0.37, '(d)')
    fig.text(0.85, 0.53, '(c)')
    fig.text(0.85, 0.705, '(b)')
    fig.text(0.85, 0.87, '(a)')
    #ax1.annotate(str(vline[0]), xy=(vline[0], 260))
    #ax1.annotate(str(vline[1]), xy=(vline[1], 260))
    # row2 npr
    ax2 = fig.add_subplot(512)  # npr
    axs.append(ax2)
    _, l2 = plot_funcs.pltyy(npr[0][0], npr[0][1], 'test_comp2', 'NPR',
                             t2=npr[1][0], s2=npr[1][1], label_y2='E(t)',
                             symbol=['k', 'g-'], handle=[fig, ax2], nbins2=6)
    ax2.locator_params(axis='y', nbins=6)
    #ax0.set_ylim([0, 0.06])
    #ax0.set_ylim([0, 0.06])

    # sigma
    ax3 = fig.add_subplot(513)  # sigma
    axs.append(ax3)
    _, l2 = plot_funcs.pltyy(sigma[0][0], sigma[0][1], 'test_comp2', '$\sigma^0$',
                             t2=sigma[1][0], s2=sigma[1][1], label_y2='E(t)',
                             symbol=['k', 'g-'], handle=[fig, ax3], nbins2=6)
    ax3.locator_params(axis='y', nbins=6)

    # moisture and temperature
    ax4 = fig.add_subplot(514)  # T soil and temperature
    axs.append(ax4)
    ax_tp, l2 = plot_funcs.pltyy(soil[0][1], soil[0][2], 'test_comp2', 'VWC (%s, %%)' % '$m^3/m^3$',
                             t2=soil[1][1], s2=soil[1][2], label_y2='T$_{soil}$ ($^\circ$C)',
                             symbol=['k-', 'b-'], handle=[fig, ax4], nbins2=6)
    ax_tp[1].axhline(ls=':', lw=1.5)
    #ax4.legend(l2, [soil[0][0], soil], loc=2, prop={'size': 10})

    #ax4.locator_params(axis='y', nbins=6)
    # swe
    ax5 = fig.add_subplot(515)  # swe
    axs.append(ax5)
    ax5.plot(snow[1], snow[2], 'k', linewidth=2.0)
    ax5.set_ylim([0, 200])
    ax5.locator_params(axis='y', nbins=6)
    ax5.set_ylabel('SWE (mm)')
    ax5.set_xlabel('Day of year')
    for ax in axs:
        ax.set_xlim([0, 365])
    lz = [':', '--', '-.']
    labelz = ['$\sigma^0$', 'TB', 'NPR']
    for ax in [ax4, ax5]:
        for i in [0, 1, 2]:
            ax.axvline(x=onset[i*2], color='k', ls=lz[i], label=labelz[i])
            ax.axvline(x=onset[i*2+1], color='k', ls=lz[i])
        ax.axvline(x=onset[6], color='r', ls='-', label='in situ')
        ax.axvline(x=onset[7], color='r', ls='-')
    #ax5.legend(bbox_to_anchor=[0., -0.6, 1., -0.6], loc=3, ncol=4, mode="expand", borderaxespad=0.)

    plt.savefig(figname, dpi=120)
    plt.close()
    return 0
    if vline is not False:
        ax_count = 0
        for ax in axs:
            ax.set_xlim([0, 350])
            ax_count += 1
            if type(vline) is list:
                ax.axvline(x=vline[0], color='k', ls='--')
                ax.axvline(x=vline[1], color='k', ls='-.')
                continue
            ax.axvline(x=vline[0, 1], color='k', label=repr(vline[0, 1]), ls='--')
            ax.axvline(x=vline[1, 1], color='k', label=repr(vline[1, 1]), ls='-.')
            ax.axvline(x=vline[2, 1], color='k', label=repr(vline[1, 1]), ls='--')
            ax.axvline(x=vline[3, 1], color='k', label=repr(vline[1, 1]), ls='-.')
            # ax.xaxis.set_minor_locator(months)
            # ax.xaxis.set_minor_formatter(monthsFmt)
            # ax.xaxis.set_major_locator(years)
            # ax.xaxis.set_major_formatter(yearsFmt)
            # ax.locator_params(axis='x', nbins=16)
            tick_num = np.array([50, 100, 150, 200, 250, 300, 350, 365, 415, 465, 515, 565, 615, 665, 715], dtype=int)
            ax.xaxis.set_ticks(tick_num)
            labels = [item.get_text() for item in ax.get_xticklabels()]
            n = 0
            for label in labels:
                if tick_num[n] == 50:
                    if ax_count == 5:
                        labels[n] = "50\nYear'15        "
                    else:
                        labels[n] = "50"
                elif tick_num[n] == 350:
                    labels[n] = ' '
                elif tick_num[n] == 365:
                     labels[n] = "365\n              Year'16"
                elif tick_num[n] == 415:
                    labels[n] = repr(tick_num[n]-365)
                elif tick_num[n] > 415:
                    labels[n] = repr(tick_num[n]-365)
                else:
                    labels[n] = repr(tick_num[n])
                n += 1
            # labels[0] = 'Year\n2015'
            # labels[1] = '100'
            # labels[2] = '150'
            # labels[3] = '200'
            # labels[4] = '250'
            # labels[5] = '300'
            # labels[6] = ''
            # labels[7] = 'Year\n2016'
            # labels[8] = '50'
            # labels[9] = '100'
            # labels[10] = '150'
            # labels[11] = '200'

            ax.set_xticklabels(labels)
    plt.savefig(figname, dpi=120)
    plt.close()


def cal_obd(peroid, site_no):
    tbn, dtr2, l2 = read_site.read_series(peroid[0], peroid[1], site_no, '_A_', data='np', dat='tb_v_aft')





def trans_peroid(ini, end, objv):
    """
    :param ini& end: the initial and end temperature for thawing and freezing
    :return:
    """
    site_dic = site_infos.get_site_dict()
    # site_dic = {'sno_': ['947', '949', '950', '960', '962', '1090', '967', '968']}
    # site_dic = {'scan_': ['2081', '2213'], 'sno_': ['947', '1177', '968']}
    num_site_list = [len(v) for v in site_dic.values()]
    num_site = sum(num_site_list)
    for kw in site_dic.keys():
        for no in site_dic[kw]:  # No. of a single site
            onset_table = np.genfromtxt('onset_test' + no + '.txt', delimiter=',', skip_header=1)
            i_pd = np.where((onset_table[:, 3] < ini) & (onset_table[:, 3] > end))
            i_ons = np.where(onset_table[:, 0] > 1)
            v_pd = np.transpose(onset_table[i_pd[0], :])
            mv_pd, tsoil, tair, ons = v_pd[2], v_pd[3], v_pd[4], onset_table[i_ons[0], 1]
            npr, npr_s = v_pd[6], v_pd[7]
            mv_pd[mv_pd < -50], tsoil[tsoil<-50], tair[tair<-50] = np.nan, np.nan, np.nan
            # plot
            fig = plt.figure(figsize=[8, 3])
            ax = fig.add_subplot(211)
            axiz, _ = plot_funcs.pltyy(v_pd[1], mv_pd, 'onet' + '_test', 'mv', t2=v_pd[1], s2=tsoil, label_y2='t soil',
                             ylim=[0, 80], ylim2=[-5, 10], handle=[fig, ax], symbol=['b.', 'r+'], subtick='yes')
            axiz[1].plot((v_pd[1][0], v_pd[1][-1]), (1, 1), 'k:')
            axiz[1].plot((v_pd[1][0], v_pd[1][-1]), (-1, -1), 'k:')
            ax2 = fig.add_subplot(212)
            ax2.plot(v_pd[1], npr, 'b.')
            plot_funcs.plt_more(ax2, v_pd[1], npr_s, symbol='r.')
            ax2.set_ylim([0, 0.05])
            for vline in ons:
                for ax0 in [ax, ax2]:
                    ax0.set_xlim([200, 750])
                    ax0.axvline(x=vline, color='k', label=repr(vline), ls='--')
            plt.savefig(objv + '_' + no + '.png', dpi=120)
            plt.close
            # with open('onset_test' + no + '.txt') as f1:
            #     reader = csv.reader(f1, delimiter=',')
            #     for row in reader:
            #         if row[0] == '#':
            #             status = 'header' # this is the header
            #         elif row[2] < 1:
    return 0


def ascat_main():
    return 0


def ascat_plot1(x, y, fname='947test', vline='False'):
    '''
    <introduction>
        plot the ascat time series and estimated onsets
    :param x:
    :param y:
    :param fname:
    :param vline:
    :return:
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax1.plot(x[0], y[0])
    #ax1.set_ylim([-20, -7])
    ax1.set_ylabel('sigma0 (dB)')
    ax1.set_xlabel('day of year 2017, descending pass only')
    ax1.set_xlim([0, 350])
    # ax1.set_ylim([-15, -7])
    for vl in vline[0: 2]:
        ax1.annotate(('%d' % vl), xy=(vl, -12))
    for vl in vline[2: ]:
        ax1.annotate(('%d' % vl), xy=(vl, -11))
    ax2 = fig.add_subplot(512)
    ax2.plot(x[1], y[1])
    ax3 = fig.add_subplot(513)
    ax3.plot(np.fix(x[2]), y[2])
    ax4 = fig.add_subplot(514)
    ax4.plot(np.fix(x[3]), y[3])
    ax4.axhline(ls=':', lw=1.5)
    ax5 = fig.add_subplot(515)
    ax5.plot(np.fix(x[4]), y[4])
    if fname == 'ascat_OG2213.png':
        ax5.set_ylim([0, 100])
    ax1.set_ylabel('sigma0 (dB)')
    ax3.set_ylabel('VWC (%s, %%)' % '$m^3/m^3$')
    ax4.set_ylabel('T$_{soil}$ ($^\circ$C)')
    ax5.set_ylabel('SWE (mm)')
    ax2.set_ylabel('Convolutions')

    if vline is not False:
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_xlim([0, 350])
            ax.axvline(x=vline[0], color='k', ls='--')
            ax.axvline(x=vline[1], color='k', ls='-.')
            ax.axvline(x=vline[2], color='r', ls='--')
            ax.axvline(x=vline[3], color='r', ls='-.')
    plt.savefig(fname + '.png', dpi=120)
    return 0


def spatial_plot(x, y, fname='117test'):
    fig = plt.figure(figsize=[8, 10])
    ax = []
    n1 = len(y)  # the number of subplots
    n2 = 0
    for var in y:
        n2 += 1
        ax0 = fig.add_subplot(n1*1e2+10+n2)
        ax0.plot(x, var[0], 'bo', markersize=2)
        plot_funcs.plt_more(ax0, x, var[1], marksize=2)
        ax.append(ax0)
    plt.savefig('spatial_series_'+fname+'.png', dpi=120)
    return 0
