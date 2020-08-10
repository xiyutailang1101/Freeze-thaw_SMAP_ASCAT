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
import basic_xiyu as bxy
from matplotlib import gridspec
import temp_test
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch, Polygon, Arc
from matplotlib import colors, cm

def make_patch(colorP):
    aqua_patch = Patch(edgecolor='none', facecolor=colorP)
    return aqua_patch
def read_h5(peroid):
    for s_no in ['1233', '2081', '2213', '2211', '2210', '2065', '947',
        '949', '950', '960', '962', '1090', '967', '968', '1089', '1175']:
        # , '1233', '2081', '2213', '2211', '2210', '2065', '947',
        # '949', '950', '960', '962', '1090', '967', '968', '1089', '1175'
        Read_radar.radar_read_main('_A_', s_no, peroid, 'vv')
    # sys.exit()


def main(site_no, period, sm_wind=17, mode='all', seriestype='sig', tbob='_A_', sig0=2, norm=False, center=False, order=1,
         value_series=False, thaw_win=[60, 150]):

    """
    period:
        not used in this version (20170911)
    site_t:
        the type of site, if has 2 elements, refer to scan and sno
    """

    # '947', '948', '949', '950',
    ob = tbob
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
    # tbn, dtr2, l2 = read_site.read_series(period[0], period[1], site_no, ob, data='np', dat='cell_tb_v_aft')
    # tb_h, dtr_h, l_h = read_site.read_series(period[0], period[1], site_no, ob, data='np', dat='cell_tb_h_aft')
    """
    P1-1 bi-linear interpolation to determine the RS data of the site
    """
    h5_root = site_infos.get_data_path('_05_01') + 's' + site_no + '/'
    site_path = h5_root

    # read site tb data, new added 12 Aug
    prefix = './result_07_01/'
    tb_fname = prefix+'txtfiles/site_tb/tb_'+site_no+tbob+'2016.txt'
    site_tb = np.loadtxt(tb_fname)
    with open(tb_fname, 'rb') as as0:
        reader = csv.reader(as0)
        for row in reader:
            if '#' in row[0]:
                n_time = row.index('cell_tb_time_seconds_aft')
                n_tbv = row.index('cell_tb_v_aft')
                n_tbh = row.index('cell_tb_h_aft')
                n_flagv = row.index('cell_tb_qual_flag_v_aft')
                n_flagh = row.index('cell_tb_qual_flag_h_aft')
                n_errorv = row.index('cell_tb_error_v_aft')
                n_errorh = row.index('cell_tb_error_h_aft')
                n_lon, n_lat = row.index(' cell_lon'), row.index('cell_lat')
                # n_time = row.index('cell_tb_time_seconds_fore')
                # n_tbv = row.index('cell_tb_v_fore')
                # n_tbh = row.index('cell_tb_h_fore')
                # n_flagv = row.index('cell_tb_qual_flag_v_fore')
                break
    site_passtime = site_tb[:, n_time]  # utc pass time, calculate the local time
    sec_pass = bxy.timetransform(site_passtime, '20000101 11:58:56', '%Y%m%d %X', tzone=True)
    site_date, site_hr = np.modf(sec_pass/(24*3600.0))[1]+1, np.modf(sec_pass/(24*3600.0))[0] * 24
    site_tbv, site_tbh = site_tb[:, n_tbv], site_tb[:, n_tbh]
    site_flagv, site_flagh = site_tb[:, n_flagv], site_tb[:, n_flagh]
    site_errorv, site_errorh = site_tb[:, n_errorv], site_tb[:, n_errorh]
    tb_center_lon, tb_center_lat = site_tb[2, n_lon], site_tb[2, n_lat]
    # utc time transform newly updated 07/21
    idx_nan = np.isnan(site_passtime)
    site_passtime[idx_nan] = 0
    time_tple = bxy.time_getlocaltime(site_passtime)
    site_date0, site_hr0 = time_tple[-2]+(time_tple[0]-2015)*365+\
                           np.max(np.array([(time_tple[0]-2016), np.zeros(time_tple[0].size)]), axis=0), \
                           time_tple[-1]
    site_date0[idx_nan], site_hr0[idx_nan] = -999, -999
    x_test = site_date - site_date
    x_test_no_nan = x_test[x_test>0]
    if center == True:
        return [tb_center_lon, tb_center_lat]
    """
    work of SMOS algorithm
    """

    # remove the invalid value
    # v_day_np = np.array(v_day)
    # var_npv = v_day_np[:, 0]
    var_npv = site_tbv
    # v_flag = np.array(v_day_np[:, 1], dtype=int)
    # invalid0 = (np.bitwise_and(v_flag, 1) == 1) | (var_npv < -1) # the bad quality data
    # invalid0 = var_npv < -1 | (np.bitwise_and(site_flagv.astype(int), 1) == 1)

    invalid0 = var_npv < -1
    var_npv[invalid0] = np.nan
    site_errorv[invalid0] = np.nan

    i_test = np.isnan(var_npv)
    t_nan = site_date[np.isnan(var_npv)]
    print t_nan


    invalid0 = site_errorv > 2*np.nanmean(site_errorv)
    print 'number of high error observations:', invalid0.size
    var_npv[invalid0] = np.nan
    # var_nph = v_day_np[:, 3]
    var_nph = site_tbh
    # h_flag = np.arrayThe file /home/xiyu/PycharmProjects/R3/tb_centers.txt changed on disk.(v_day_np[:, 4], dtype=int)
    # invalid0 = (np.bitwise_and(h_flag, 1) == 1) | (var_nph < -1)
    invalid0 = var_nph < -1
    var_nph[invalid0] = np.nan
    site_errorh[invalid0] = np.nan
    invalid0 = site_errorh > 2*np.nanmean(site_errorh)
    var_nph[invalid0] = np.nan

    #pol_diff = data_process.rm_odd(var_nph, site_Patchdate, 0)

    with open('./result_07_01/txtfiles/site_statistic/mean_error.csv', 'a') as csvfile:
        site_statistic = csv.writer(csvfile)
        site_statistic.writerow(['%s, v:, %.3f, %.3f, h: %.3f, %.3f' % (site_no, np.nanmean(np.abs(site_errorv)), np.nanmax(np.abs(site_errorv)),
                                                                        np.nanmean(np.abs(site_errorh)), np.nanmax(np.abs(site_errorh)))])

    t_v, t_h = site_date, site_date
    ffv = 300.0 - var_npv

    # remove temporal change
    # data_process.rm_odd(var_npv,0,0)
    # data_process.rm_odd(var_nph,0,0)
    edge_seriesv = var_npv
    edge_seriesh = var_nph

    ffnpr = (np.array(edge_seriesv) - np.array(edge_seriesh))/(np.array(edge_seriesv) + np.array(edge_seriesh))
    invalid1 = ffnpr<0
    ffnpr[invalid1] = np.nan
    len_origin = t_h.shape[0]
    i_origin = range(0, len_origin, 1)
    i_val = i_origin[(sm_wind-1)/2: t_h.size-(sm_wind+1)/2 + 1]
    valid_dateh = t_h[i_val]  # date --> smooth[valid]
    valid_datev = t_v[i_val]
    ffnpr_s, valid = data_process.n_convolve2(ffnpr, sm_wind)
    ffnpr_s = ffnpr_s[valid]
    # remove some odd points:
    if seriestype=='tb':
        edge_series = var_npv
        t_series = t_v
        # odd_point = data_process.rm_temporal(edge_series)
        # edge_series[odd_point] = np.nan
    else:
        edge_series = ffnpr
        t_series = t_h
    # g_size = 12
    if seriestype == 'tb':
        peaks_iter = 1e-1
    elif seriestype == 'npr' or seriestype == 'sig':
        peaks_iter = 1e-4
    # new added 2018 Jan 29
    if value_series is not False:
        edge_series = value_series[1]
        t_series = value_series[0]
    for kwidth in [sig0]:
        g_size = 6*kwidth/2
        if order == 1:  # first order of gaussian
            g_npr, i_gaussian = data_process.gauss_conv(edge_series, sig=kwidth, size=2*g_size+1)  # option: ffnpr-t_h; var_npv-t_h
            g_npr_valid_non = g_npr[g_size: -g_size]
            max_gnpr, min_gnpr = peakdetect.peakdet(g_npr_valid_non, peaks_iter, t_series[i_gaussian][g_size: -g_size])
            onset = data_process.find_inflect(max_gnpr, min_gnpr, typez='annual', typen=seriestype, t_win=thaw_win)  # !!
        elif order == 2:
            g_npr, i_gaussian = data_process.gauss2nd_conv(edge_series, sig=kwidth)  # option: ffnpr-t_h; var_npv-t_h
            g_npr_valid_non = g_npr[g_size: -g_size]
            max_gnpr, min_gnpr = peakdetect.peakdet(g_npr_valid_non, peaks_iter, t_series[i_gaussian][g_size: -g_size])
            onset = data_process.find_inflect(max_gnpr, min_gnpr, typez='annual', typen=seriestype)  # !!
        elif order == 3:
            onset, g_npr, i_gaussian, g_npr_valid_non, max_gnpr, min_gnpr, sig = \
                data_process.gauss_cov_snr(edge_series, peaks_iter, t_series, s_type=seriestype)
            g_size = 6*sig/2
        if type(g_npr) is int:
            continue
        g_npr_valid_n = 2*(g_npr[g_size: -g_size] - np.nanmin(g_npr[g_size: -g_size]))\
                        /(np.nanmax(g_npr[g_size: -g_size]) - np.nanmin(g_npr[g_size: -g_size])) - 1  # normalized
        if norm == True:
            g_npr_valid_n = g_npr[g_size: -g_size]  # Non-nomarlized
    site_tb = []
    if center == True:
        return [tb_center_lon, tb_center_lat]
    else:
        return [t_v-365, var_npv], [t_h-365, var_nph], [t_series-365, ffnpr], \
               [t_series[i_gaussian][g_size: -g_size]-365, g_npr_valid_n, g_npr_valid_non], [onset[0], onset[1]], [site_date, np.round(site_hr)], [max_gnpr, min_gnpr]

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


def read_alaska(start_date, end_date):
    predir = './result_05_01/SMAP_AK/'
    orbit = '_A_'
    site_info = ['alaska', 63.25, -151.5]
    tb_attr_name = ['/cell_tb_v_aft', '/cell_tb_h_aft', '/cell_tb_qual_flag_v_aft', '/cell_tb_qual_flag_h_aft',
                    "/cell_lat", "/cell_lon"]
    TBdir = '/media/327A50047A4FC379/SMAP/SPL1CTB.003/'
    TB_date_list = sorted(os.listdir(TBdir))
    TB_date = []
    TB_date2, date_ind = [], -1  # use to find yyyy.mm.dd format documents
    # Time list of radar data
    time_list = []
    time_ini = 0
    ind_tb_date0 = TB_date_list.index(start_date)
    ind_tb_date1 = TB_date_list.index(end_date)
    TB_list_read = TB_date_list[ind_tb_date0: ind_tb_date1+1]
    for time_dot in TB_list_read:
        # if time_dot == start_date:
        #     time_ini = 1
        # if time_ini > 0:
        TB_date.append(time_dot.replace(".", ""))
        TB_date2.append(time_dot)
    for time_str in TB_date:
        print '%s is reading' % TB_date2[date_ind+1]
        date_ind += 1
        # 01 list with radiometer files
        tb_file_list = []  # tb files with different orbits
        tb_ob_list = []  # orbits of tb files
        TB_file_No = TB_date_list.index(TB_date2[date_ind])
        print 'the order of tb_file is :', TB_file_No, 'which is: ', TB_date_list[TB_file_No]
        if TB_date_list[TB_file_No] != TB_date2[date_ind]:
            print 'data of %s is miss-matched' %time_str
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
        np.save(predir+'tp/DES_txt_xyz'+TB_date_list[TB_file_No], np.transpose(np.array([x, y, v_tb, h_tb])))


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


def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale


def plt_npr_gaussian_all(tb, npr, sigma, soil, snow, onset, figname='all_plot_test0420', size=(12, 8), xlims=[0, 365],
                         shade=False, title=False, site_no='947', pp=False, subs=5, s_symbol='k.',
                         day_tout=-1, end_ax1=[0, 0], end_ax2=[0, 0], end_ax3=[0, 0], tair=[], snow_plot=False):
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
    :param snow:
    :param onset:
    :return:
    """
    # ylim for each station
    site_lim = {'947': [-17, -7], '949': [-13, -7], '950': [-13, -7], '960': [-14, -8], '962': [-15, -8], '967': [-12, -8], '968': [-17, -7],
                '1089': [-13, -7], '1090': [-14, -7], '1175': [-15, -8], '1177': [-19, -10],
                '1233': [-17, -6], '2065': [-14, -8], '2081': [-15, -7], '2210': [-16, -8], '2211': [-16, -8], '2212': [-16, -8],
                '2213': [-17, -10]}
    axs = []
    fig = plt.figure(figsize=size)
    gs0 = gridspec.GridSpec(5, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[0])
    ax0 = plt.Subplot(fig, gs00[-1, :])
    # ax1, ax2, ax3, ax4 = plt.subplot(fig, gs0[1]), plt.subplot(fig, gs0[2]), \
    #                      plt.subplot(fig, gs0[3]), plt.subplot(fig, gs0[4])
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=size, sharex=True)  # sharex
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    # #  add a time window bar 2018/05 updated
    # nr, st = 20, 9
    # sub_no = 4*nr+st
    # ax0, ax1, ax2, ax3, ax4 = plt.subplot2grid((sub_no, 1), (0, 0), rowspan=st), \
    #                           plt.subplot2grid((sub_no, 1), (st, 0), rowspan=nr),  \
    #                           plt.subplot2grid((sub_no, 1), (st+nr, 0), rowspan=nr), \
    #                           plt.subplot2grid((sub_no, 1), (st+2*nr, 0), rowspan=nr), \
    #                           plt.subplot2grid((sub_no, 1), (st+3*nr, 0), rowspan=nr)

    # params = {'mathtext.default': 'regular'}
    # plt.rcParams.update(params)
    # row 1 tb
    # ax1 = fig.add_subplot(511)  # tb
    # ax0 setting, for boundary of seasons
    # timings = [60, 150, 260, 350, 366]
    # timing_color = ['aqua', 'red', 'orange', 'blue', 'aqua']
    # timing_color_rgba = plot_funcs.check_rgba(timing_color)
    # timing_color_rgba[3] = [0., .3, 1., 1.]
    # print timing_color_rgba
    # timing_name = ["Frozen", "Thawing", "Thawed", "Freezing", " "]
    # fill_y1 = 1
    # ax0.plot(soil[0][1], soil[0][2]*0)
    # plot_funcs.make_ticklabels_invisible(ax0)  # hide the y_tick
    # ax0.tick_params(axis='x', which='both', bottom='off', top='off')
    # ax0.tick_params(axis='y', which='both', left='off', right='off')
    # text_x0 = 0
    #
    # for i in range(0, len(timings)):
    #     ax0.fill_between(np.arange(text_x0, timings[i]), fill_y1, color=timing_color_rgba[i])
    #     text_x = 0.5*(timings[i]+text_x0)
    #     print text_x
    #     ax0.text(text_x, 0.5, timing_name[i], va="center", ha="center")  # 1.3 up
    #     text_x0 = timings[i]+1
    #     if i < len(timings)-1:

    #         # add vertical line and label
    #         ax0.axvline(timings[i])
    #         ax0.text(timings[i], 1.3, timings[i], va="center", ha="center")


    print np.nanmax(soil[0][1]), np.nanmin(soil[0][1])
    ax0.set_xlim(xlims)
    axs.append(ax1)
    # l1, = ax1.plot(tb[0][0], tb[0][1], 'bo', markersize=2)
    _, ax1_2, l1 = plot_funcs.pltyy(tb[0][0], tb[0][1], 'test_comp2', 'T$_b$ (K)',
                             t2=tb[2][0], s2=tb[2][1], label_y2= '$E_{Tbv}$\n(K/day)',
                             symbol=['k.', 'g-'],
                             handle=[fig, ax1], nbins2=6)  # plot tbv
    l1_le = plot_funcs.plt_more(ax1, tb[1][0], tb[1][1], line_list=[l1])
    # ax1.locator_params(axis='y', nbins=4)
    # ax1_2.axhline(y=0)
    ax1.set_ylabel('T$_b$ (K)')
    # ax1.legend([l1_le[0][0], l1_le[1]], ['T$_{BV}$', 'T$_{BH}$'], loc=3, prop={'size': 6})
    if title is not False:
        plt.title(title)
    #ax4.legend([l4], [fe[0]], loc=2, prop={'size': 10})
    # ax4.text(0.85, 0.85, '(a)')
    # fig.text(0.85, 0.21, '(e)')
    # fig.text(0.85, 0.37, '(d)')
    # fig.text(0.85, 0.53, '(c)')
    # fig.text(0.85, 0.705, '(b)')
    # fig.text(0.85, 0.87, '(a)')
    #ax1.annotate(str(vline[0]), xy=(vline[0], 260))
    #ax1.annotate(str(vline[1]), xy=(vline[1], 260))
    # row2 npr
    # ax2 = fig.add_subplot(512)  # npr
    axs.append(ax2)
    _, ax2_2, l2 = plot_funcs.pltyy(npr[0][0], npr[0][1], 'test_comp2', 'NPR ($10^{-2}$)',
                             t2=npr[1][0], s2=npr[1][1], label_y2='$E_{NPR}$\n($10^{-2}$/day)',
                             symbol=[s_symbol, 'g-'], handle=[fig, ax2], nbins2=6)
    # ax2.locator_params(axis='y', nbins=5)
    #ax0.set_ylim([0, 0.06])
    #ax0.set_ylim([0, 0.06])

    # sigma
    # ax3 = fig.add_subplot(513)  # sigma
    axs.append(ax3)
    _, ax3_2, l2 = plot_funcs.pltyy(sigma[0][0], sigma[0][1], 'test_comp2', '$\sigma^0_{45} (dB)$',
                             t2=sigma[1][0], s2=sigma[1][1], label_y2='$E_{\sigma^0_{45}}$\n(dB/day)',
                             symbol=[s_symbol, 'g-'], handle=[fig, ax3], nbins2=6)
    # ax3.set_ylim(site_lim[site_no])
    # ax3.locator_params(axis='y', nbins=4)

    # moisture and temperature
    # ax4 = fig.add_subplot(514)  # T soil and temperature
    axs.append(ax4)
    if snow_plot is False:
        _, ax4_2, l2 = plot_funcs.pltyy(soil[0][1], soil[0][2], 'test_comp2', 'VWC (%)',
                                 t2=soil[1][1], s2=soil[1][2], label_y2='T$_{soil}$ ($^\circ$C)',
                                 symbol=['k-', 'b-'], handle=[fig, ax4], nbins2=6)
        for ax_2 in [ax4_2]:
            ax_2.axhline(ls='--', lw=1.5)
    else:
        ax4.plot(snow[1], snow[2], 'k', linewidth=2.0)
        ax4_2 = ax4.twinx()
        if len(tair) > 0:
            tair[1][tair[1] < -60] = np.nan
            ax4_2.plot(tair[0], tair[1], 'k:')
            ax4_2.set_ylim([-30, 30])
            ax4_2.axhline(ls='--', lw=1.5)
            ax4_2.yaxis.set_major_locator(MaxNLocator(5))
            ax4_2.set_ylabel('T$_{air}$ ($^o$C)')


    ax2s = [ax1_2, ax2_2, ax3_2, ax4_2]
    ax_ins = [ax4]
    # swe
    # ax5 = fig.add_subplot(515)  # swe
    # axs.append(ax5)
    # ax_ins.append(ax5)
    # ax5.plot(snow[1], snow[2], 'k', linewidth=2.0)
    # add air temperature
    # if len(tair) > 0:
    #     ax5_2 = ax5.twinx()
    #     ax2s.append(ax5_2)
    #     tair[1][tair[1] < -60] = np.nan
    #     ax5_2.plot(tair[0], tair[1], 'k:')
    #     ax5_2.set_ylim([-30, 30])
    #     ax5_2.axhline(ls='--', lw=1.5)
    #     ax5_2.yaxis.set_major_locator(MaxNLocator(5))
    #     ax5_2.set_ylabel('T$_{air}$ ($^o$C)')
    # if not pp:
    #     if site_no in ['947', '949', '950', '967', '1089']:
    #         ax5.set_ylabel('SWE (mm)')
    #         ax5.set_ylim([0, 200])
    #     else:
    #         ax5.set_ylabel('SD (cm)')
    #         ax5.set_ylim([0, 100])
    #     if site_no in ['950', '1089']:
    #         ax5.set_ylim([0, 500])
    # else:
    #     ax5.set_ylabel('precipitation (mm)')
    # ax4.set_xlabel('Day of year 2016')

    # add vertical line
    lz = ['--', '--', '--']
    labelz = ['$\sigma^0$', 'TB', 'NPR']
    if onset.size> 4:  # freeze and thaw
        i2 = -1
        # for ax in [ax4, ax5]:
        #     # for i in [0, 1, 2]:
        #     for i in [0, 1, 2]:
        #         ax.axvline(x=onset[i*2], color='k', ls=lz[i], label=labelz[i])
        #         ax.axvline(x=onset[i*2+1], color='k', ls=lz[i])
        for ax in [ax3, ax1, ax2]:
            # ax.axvline(x=onset[-2], color='r', ls='-', label='in situ')
            # ax.axvline(x=onset[-1], color='r', ls='-')
            i2 += 1
            ax.axvline(x=onset[i2*2], color='k', ls=lz[i2], label=labelz[i2])
            ax.axvline(x=onset[i2*2+1], color='k', ls=lz[i2])
    elif onset.size <=4:
        for ax in ax_ins:
            for i in [0]:
                ax.axvline(x=onset[i], color='k', ls=lz[i], label=labelz[i])
                ax.axvline(x=onset[i+1], color='k', ls=lz[i+1])
                ax.axvline(x=onset[i+2], color='k', ls=lz[i+2])

    l2d_sm = ax4.axvline(x=onset[6], color='r', ls='--')
    ax4.axvline(x=onset[7], color='r', ls='--')

    # special vline
    if day_tout > 0:
        l2d, = ax4.axvline(x=day_tout, color='r', ls=':')
        ax1.axvline(x=end_ax1[0], color='b', ls='--')
        ax1.axvline(x=end_ax1[1], color='b', ls='--')
        ax2.axvline(x=end_ax2[0], color='b', ls='-')
        ax2.axvline(x=end_ax2[1], color='b', ls='-')
        ax3.axvline(x=end_ax3[0], color='b', ls=':')
        ax3.axvline(x=end_ax3[1], color='b', ls=':')

    # plot settings
    # ticks setting
    for ax in ax_ins:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axs:
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        yticks[-1].label1.set_visible(False)
    for ax in ax2s:
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label2.set_visible(False)
        yticks[-1].label2.set_visible(False)
    # label location
    text4 = ['a', 'b', 'c', 'd', 'e']
    i4 = -1
    for i, ax in enumerate(axs):
        ax.yaxis.set_major_locator(MaxNLocator(4))
        i4 += 1
        ax.get_yaxis().set_label_coords(-0.09, 0.5)
        ax.text(0.02, 0.95, text4[i], transform=ax.transAxes, va='top', fontsize=16)
        # ax.annotate(text4[i4], xy=get_axis_limits(ax), fontweight='bold')
    for ax in ax2s:
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.get_yaxis().set_label_coords(1.10, 0.5)  # position of labels

    # ylims
    ax3_2.set_ylim([-3, 2])
    ax1.set_ylim([210, 280])
    if site_no == '1233':
        ax1.set_ylim([180, 275])
        ax3_2.set_ylim([-3, 3])
    # ax1_2.set_ylim([-9, 9])
    if site_no == '1177':
        st = 0
    else:
        ax2.set_ylim([0, 6])
        ax2_2.set_ylim([-2, 2])

    # x_label
    for i3 in range(0, 4):
        axs[i3].set_xlabel('')

    if shade is False:
        shade_window = 'no shade'
    else:
        for ax in axs:
            for shade0 in shade:
                ax.axvspan(shade0[0], shade0[1], color=(0.8, 0.8, 0.8), alpha=0.5, lw=0)
    if xlims:
        for ax in axs:
            ax.set_xlim(xlims)

    # legend setting
    leg1 = ax1.legend([l1_le[0][0], l1_le[1]], ['T$_{bv}$', 'T$_{bh}$'],
               loc=3, ncol=1, prop={'size': 12}, numpoints=1)
    # for leg in [leg1]:
    #     leg.get_frame().set_linewidth(0.0)
    # layout setting
    ax4.set_xlabel('Day of year 2016')
    plt.tight_layout()

    # if site_no == '1233':
    #     ax1.set_visible(False)
    #     ax1_2.set_visible(False)
    #     ax3.set_visible(False)
    #     ax3_2.set_visible(False)

    fig.subplots_adjust(hspace=0.05)

    # other setting like the title


    # ax_name = ['tb', 'npr', 'sig', 'VWC', 'SWE', 'tbG', 'nprG', 'sigG', 'tsoil']
    # ax_i = 0
    # yticks = ax2.yaxis.get_major_ticks()
    # yticks[0].label1.set_visible(False)
    # yticks[-2].label1.set_visible(False)
    # for ax in [ax1, ax2, ax3, ax4, ax5]:
    #     yticks = ax.yaxis.get_major_ticks()
    #     yticks[0].label1.set_visible(False)
    #     yticks[-1].label1.set_visible(False)
    # for ax in [ax1_2, ax2_2, ax3_2, ax4_2]:
    #     yticks = ax.yaxis.get_major_ticks()
    #     yticks[0].label2.set_visible(False)
    #     yticks[-1].label2.set_visible(False)
    plt.rcParams.update({'font.size': 16})
    print figname
    plt.savefig(figname, dpi=300)
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
    plt.savefig(figname, dpi=300)
    plt.close()


def plot_snow_effect(tbv, tbh, obd_v, obd_h, swe, dswe=False, air_change=False, fname='', sno='947', vlines=False):
    fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    # ax0 = fig.add_subplot(3, 1, 1)
    l0, = ax0.plot(tbv[0], tbv[1]*100, 'ko', markersize=3)
    # l00 = plot_funcs.plt_more(ax0, tbh[0], tbh[1], line_list=[l0])
    # ax0.legend(l00, ['V-pol', 'H-pol'], loc=2, prop={'size': 10})
    # ax1 = fig.add_subplot(3, 1, 2)
    l1, = ax1.plot(obd_v[0], obd_v[1], 'ko', markersize=3)
    l11 = plot_funcs.plt_more(ax1, obd_h[0], obd_h[1], line_list=[l1], symbol='ro')
    ax1.axhline(ls=':', lw=1.5)
    ax1.legend(l11, ['V-pol', 'H-pol'], loc=2, prop={'size': 10}, numpoints=1)

    # ax2 = fig.add_subplot(3, 1, 3)
    ax2.plot(swe[0], swe[1], 'k-')
    ax22 = ax2.twinx()
    if air_change is False:
        ax22_color = 'k'
        dswe[1][(dswe[1] >-1) & (dswe[1] < 1)] = np.nan
        ax22.bar(dswe[0], dswe[1], 5, color='grey')
        ax22.set_ylim([-60, 60])
        ax22label = '10-day SWE change\n(mm)'
    else:
        ax22_color = 'b'
        ax22label = '$T_{air} (^oC)$'
        ax22.axhline(ls='--', lw=1.5)
        ax22.plot(air_change[0], air_change[1], ax22_color+'o', markersize=3)
        # air_new = np.zeros(swe[0].size) - 99
        # np.intersect1d(swe[0], air_change[0])
        cm0 = plot_funcs.make_cm()
        color0 = plot_funcs.make_rgba(cm0, air_change[1])
        color1 = plot_funcs.get_colors(air_change[1], cm0, -30, 30)
        # color2 = np.random.rand(swe[0].size, 3)
        color2 = tuple(map(tuple, color1))
        normalize = colors.Normalize(vmin=air_change[1][air_change[1]>-10].min(), vmax=air_change[1][air_change[1]>10].min())
        cmap = plt.get_cmap('coolwarm')
        test00=0
        # for i in range(swe[0].size-1):
        #     ax2.fill_between([swe[0][i], swe[0][i+1]], [swe[1][i], swe[1][i+1]], color=cmap(normalize(air_change[1][i])), cmap=plt.get_cmap('rainbow'))  # c=z_value, cmap=plt.get_cmap('rainbow')
        # ax2.fill_between(swe[0], 0, swe[1], color=cmap(normalize(swe[1])), cmap=plt.get_cmap('rainbow'))  # c=z_value, cmap=plt.get_cmap('rainbow')
        # ax22.contourf(air_change[0], air_change[1], air_change[1], 20)
    plt.xlim([0, 365])
    ylabels = ['$NPR_{pm}$ (10$^{-2}$)', '$\Delta T_{b, diurnal}$ (K)', 'SWE (mm)', ax22label]
    i = 0
    # locate label or shade the T_air region
    onset_file = 'result_07_01/txtfiles/result_txt/smap_ft_compare_%s.csv' % 'A'
    onset_fromfile = np.loadtxt(onset_file, delimiter=',')
    onset_value = onset_fromfile[onset_fromfile[:, 10] == int(sno), :]

    shade0 = [onset_value[0][16], swe[0][(swe[1]>-1)&(swe[1]<5)][0]]
    # swe[0][(swe[1]>0)(swe[1]<5)][0]
    print onset_value.size, 'the t_air0 time is ', shade0[0]
    ax1.set_ylim([-10, 40])
    i=0
    for ax in [ax0, ax1, ax2]:
        ax.axvspan(shade0[0], shade0[1], color=(0.8, 0.8, 0.8), alpha=0.5, lw=0)
        no_use = 0
    if vlines is not False:
        for vline0 in vlines:
            ax1.axvline(x=vline0, color='k', ls='--')
    # ax1.axvline(x=80, color='k', ls='-')
    # ax1.axvline(x=115, color='k', ls='-')

    # added 2018/07/31, calculating the average delta Tb in the shade area.
    mean_delta_tbv = np.mean(obd_v[1][(obd_v[0]>shade0[0]) & (obd_v[0]<shade0[1])])
    mean_delta_tbh = np.mean(obd_h[1][(obd_h[0]>shade0[0]) & (obd_h[0]<shade0[1])])
    with open('diurnal_test.txt', 'a') as txt0:
        txt0.write('%s, %.2f, %.2f \n' % (sno, mean_delta_tbv, mean_delta_tbh))

    text3, i3 = ['a', 'b', 'c'], -1
    for ax in [ax0, ax1, ax2]:
        i3 += 1
        ax.get_yaxis().set_label_coords(-0.07, 0.5)
        ax.text(0.95, 0.95, text3[i3], transform=ax.transAxes, va='top', fontsize=16)
    for ax in [ax0, ax1, ax2]:
        ax.set_ylabel(ylabels[i])
        i+=1
    ax22.set_ylabel(ax22label, color=ax22_color)
    for tn in ax22.get_yticklabels():
        tn.set_color(ax22_color)
    ax2.set_xlabel('Day of year 2016')
    # layout setting
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    ax22.yaxis.set_major_locator(MaxNLocator(5))
    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    if len(fname) > 0:
        plt.savefig(fname, dpi=300)
    else:
        plt.savefig('snow_eff_test.png', dpi=300)
    return 0





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


def plot_ft_compare(pixel_id, ft10, ft10_dict, period, sno='947', compare=False, orb_plot='_A_', sec_dict=[],
                    npr_ref=False, ldcover=False, subplot_a=1):
            # check if is a gap
    # initial
    sno_dict1 = {'947': ['24.6', '25.07'], '949': ['17.36', '18.32'], '950': ['18.93', '19.77'], '968': ['9.04'], '1090': ['11.45'],
                 '962': ['15.01', '20.79'], '2212': ['27.66', '8.83'],
                 '2210': ['18.77', '21.43'], '967': ['24.13', '15.65'], '2081': ['12.06', '25.91'],
                 '1175': ['12.57', '26.86'], '1177': ['21.29'], '1233': ['25.38', '26.04'], '2065': ['8.24'],
                 '2213': ['23.9', '24.84']}  # '2213': ['26.29', '26.6']
    # check the NN
    keysf = np.array([float(k) for k in ft10_dict.keys()])
    key_nn = ft10_dict.keys()[np.argmin(keysf)]
    if sno == '962':
        key_nn = '20.79'
    for key_num in pixel_id:
        key0 = str(key_num)
        if sum(ft10_dict[key0] == 254) > 100:
            del ft10_dict[key0]
            continue
    fig, axs = plt.subplots(pixel_id.size, sharex=True)
    if type(axs) is not np.ndarray:
        axs = [axs]
    symbol = ['ro', 'go', 'bo', 'ko']
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    x = np.arange(1, ft10.shape[0]+1)
    for i, key0 in enumerate(ft10_dict.keys()):
        # convs the FT product
        sz = 5
        filter = np.zeros(sz)
        wei = 1.0/sz
        filter[0: 2] = wei
        filter[2: ] = (1-2*wei)/(sz-2)
        ft_conv = np.convolve(ft10_dict[key0], filter, 'valid')
        t_th = 50
        v0 = np.where(ft_conv[t_th:]<0.01)   # complete thawing
        v01 = np.where(ft_conv[t_th:]<0.5)  # early thawing
        t_fr = 240
        v1 = np.where(ft_conv[t_fr:]>0.8)  # complete freezing
        v11 = np.where(ft_conv[t_fr:]>0.5)  # early freezing
        vline0, vline01, vline1, vline11 = v0[0][0] + t_th + filter.size/2, v01[0][0] + t_th + filter.size/2, \
                                           v1[0][0]+t_fr + filter.size/2, v11[0][0]+t_fr + filter.size/2
        if key0 == key_nn:
            vline0r, vline01r, vline1r, vline11r = vline0, vline01, vline1, vline11
        ll = axs[i].plot(x, ft10_dict[key0], symbol[i], markersize=3)
        for vv in [vline0, vline1]:
            axs[i].axvline(x=vv, color=symbol[i][0], ls='--')
        for vv in [vline01, vline11]:
            axs[i].axvline(x=vv, color=symbol[i][0], ls='-')
        axs[i].set_ylim([-2, 2])
        axs[i].set_xlim([period[1], period[2]])
        axs[i].legend(ll, [key0], prop={'size': 10}, numpoints=1)
        axs[i].yaxis.set_major_locator(MaxNLocator(5))
        # ax1.legend(l11, ['V-pol', 'H-pol'], loc=2, prop={'size': 10}, numpoints=1)
    figname = './result_07_01/%s_smap_ft_comapre_%s_%s' % (orb_plot[1], period[0], sno)  #new_final/smap_compare/
    plt.savefig(figname)
    plt.close()

    # plot together with NPR, NPR-onset, Series of Temperature and VWC
        # calculate the NPR-onset
    sno_sp = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    sno_dict = {'947': ['24.6', '25.07'], '949': ['17.36', '18.32'], '950': ['18.93', '19.77']}
    # sno_sp = ['947', '968']
    if (sno in sno_sp) & compare is True:
        if orb_plot == '_A_':
            in_situ_time = 18
        else:
            in_situ_time = 8

        pass_time = bxy.time_getlocaltime(sec_dict[key_nn][0:20])
        in_situ_time =np.round(np.mean(pass_time[-1]))
        print 'site no %s, with pass hour %d' % (sno, in_situ_time)
        # npr time series
        tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = \
            main(sno, [], sm_wind=7, mode='annual', tbob=orb_plot, sig0=7, order=1)  # result npr
        si0 = sno
        y2_empty = []
        doy = np.arange(1, 366) + 365
        site_type = site_infos.get_type(si0)
        site_file = './copy0519/txt/'+site_type + si0 + '.txt'
        stats_sm, sm5 = read_site.read_sno(site_file, "Soil Moisture Percent -2in (pct)", si0)  # air tmp
        sm5_daily, sm5_date = data_process.cal_emi(sm5, y2_empty, doy, hrs=int(in_situ_time))
        stats_t, t5 = read_site.read_sno(site_file, "Soil Temperature Observed -2in (degC)", si0)
        t5_daily, t5_date = data_process.cal_emi(t5, y2_empty, doy, hrs=int(in_situ_time))
        stats_swe, swe = read_site.read_sno(site_file, "snow", si0, field_no=-1)
        swe_daily, swe_date = data_process.cal_emi(swe, y2_empty, doy, hrs=0)
        stats_t, tair5 = read_site.read_sno(site_file, "Air Temperature Observed (degC)", si0)
        tair_daily, tair_date = data_process.cal_emi(tair5, y2_empty, doy, hrs=int(in_situ_time))
        if sno in ['2065', '2081']:
            stats_t, tair5 = read_site.read_sno(site_file, "Air Temperature Average (degC)", si0)
            tair_daily, tair_date = data_process.cal_emi(tair5, y2_empty, doy, hrs=int(in_situ_time))
        sm5_daily[sm5_daily < -90], t5_daily[t5_daily < -90], \
        swe_daily[swe_daily < -90], tair_daily[tair_daily < -90] = np.nan, np.nan, np.nan, np.nan
        sm, tsoil, swe, tair = \
            [sm5_date-365, sm5_daily], [t5_date-365, t5_daily], [swe_date-365, swe_daily], [tair_date-365, tair_daily]

        # start ploting
        # fig1 = plt.figure()
        # ax0 = fig1.add_subplot(111)
        ax0 = plt.subplot2grid((3, 5), (0, 0), colspan=4, rowspan=3)
        ax0.set_ylim([0, 150])
        smap_ft = 100 + (ft10_dict[key_nn]*-1+1)*20  # plot smap ft
        smap_ft[smap_ft>130] = np.nan
        l_smap_ft = ax0.plot(x, smap_ft, 'ks', markersize=2)
        l_sm = ax0.plot(sm[0], sm[1], 'k-')
        ax01 = ax0.twinx()
        l_temp = ax01.plot(tsoil[0], tsoil[1], 'b-')
        l_tair = ax01.plot(tair[0], tair[1], 'bo', markersize=3)
        ax01.axhline(ls='--', lw=1.5)
        ax01.set_ylim(-10, 60)
        ax02 = ax0.twinx()
        ax02.get_yaxis().set_visible(False)
        npr_norm = 20+(30-20)*(npr1[1]-np.nanmin(npr1[1]))/(np.nanmax(npr1[1])-np.nanmin(npr1[1]))
        l_npr = ax01.plot(npr1[0], npr_norm, 'r-')
        if npr_ref is not False:
            npr_fr = (npr_ref[0][key_nn])*1e-2
            npr_th = (npr_ref[1][key_nn])*1e-2
            npr_smap_nn = npr_ref[2][key_nn]
            npr_smap_nn[npr_smap_nn==-1] = np.nan
            npr_smap=npr_smap_nn*1e-2
            N_min, N_max = np.nanmin(npr1[1]), np.nanmax(npr1[1])
            npr_smap_normed = 20+(30-20)*(npr_smap-N_min)/(N_max-N_min)
            npr_smap_date = np.arange(1, 366)
            l_npr_smap = ax01.plot(npr_smap_date, npr_smap_normed, 'k-')
            npr_ref_normed = [20+(30-20)*(npri-N_min)/(N_max-N_min)
                              for npri in [npr_fr, npr_th, (npr_th-npr_fr)*0.5+npr_fr]]
            ax01.axhline(y=npr_ref_normed[0], ls=':')
            ax01.axhline(y=npr_ref_normed[1], ls=':')
            ax01.axhline(y=npr_ref_normed[2], ls='-')
        for ax00 in [ax0, ax01, ax02]:
            ax00.set_xlim([0, 365])

        # set x, y lables
        plt.draw()
        ax0.set_ylabel('VWC (%)                            F/T state')
        ax01.set_ylabel('$T_{    } (^0C)$                                     ', color='b')
        ax0label = [item.get_text() for item in ax0.get_yticklabels(which='major')]
        for i, label0 in enumerate(ax0label):
            if label0 == '120':
                ax0label[i] = 'Thawed'
            elif label0 == '100':
                ax0label[i] = 'Frozen'
            elif label0 == '140':
                ax0label[i] = ' '
        ax0.set_yticklabels(ax0label)  # set Y1 label
        ax1tick_label = [item.get_text() for item in ax01.get_yticklabels(which='major')]
        ax1tick = [item for item in ax01.get_yticklabels(which='major')]
        for i, label0 in enumerate(ax1tick_label):
            if label0 in ['30', '40', '50', '60']:
                ax1tick_label[i] = ' '
            else:
                ax1tick[i].set_color('b')
        ax01.set_yticklabels(ax1tick_label)  # set Y2 label

        # add onset vertical line
        v_file ='./result_07_01/txtfiles/result_txt/smap_ft_compare_%s.csv' % orb_plot[1]  # vline from txt
        onset_array = np.loadtxt(v_file, delimiter=',')
        onset_row = onset_array[onset_array[:, 10] == int(sno), :][0]
        ths, frs = \
            [onset_row[4], onset_row[6], onset_row[11]], [onset_row[5], onset_row[7], onset_row[13]]  # npr, site, smap
        v_symbol = [':', '-', '--']  # npr, site, smap
        vline_l2d = []
        for onset2 in [ths, frs]:
            for i, v in enumerate(onset2):
                l2d = ax0.axvline(x=v, color='k', ls=v_symbol[i])
                vline_l2d.append(l2d)
        # add legend
        ax0.legend([l_temp[0],l_tair[0], l_npr[0], l_sm[0], l_smap_ft[0], vline_l2d[0], vline_l2d[1], vline_l2d[2]],
                   ['$T_{soil}$', '$T_{air}$', 'NPR', 'VWC', 'SMAP F/T', 'onset (NPR)', 'onset (In situ)', 'onset (SMAP)'],
                   bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0., prop={'size': 12})
        # save plotting
        fname = 'result_07_01/temp_result/%s_plot_ft_compare_%s' % (orb_plot[1], sno)
        plt.savefig(fname)
        plt.close()

        # new plotting
        if npr_ref is not False:
            # npr_fr = (npr_ref[0][key_nn])*1e-2
            # npr_th = (npr_ref[1][key_nn])*1e-2
            # npr_smap = npr_ref[2][key_nn]*1e2
            # npr_smap[npr_smap==-1] = np.nan
            # npr_smap*=1e-2
            # N_min, N_max = np.nanmin(npr1[1]), np.nanmax(npr1[1])
            # npr_smap_normed = 20+(30-20)*(npr_smap-N_min)/(N_max-N_min)
            # npr_smap_date = np.arange(1, 366)
            # l_npr_smap = ax01.plot(npr_smap_date, npr_smap_normed, 'k-')
            # npr_ref_normed = [20+(30-20)*(npri-N_min)/(N_max-N_min)
            #                   for npri in [npr_fr, npr_th, (npr_th-npr_fr)*0.5+npr_fr]]

            fig0 = plt.figure()
            ax_ft = fig0.add_subplot(2, 1, subplot_a)
            colors = ft10_dict[key_nn]
            colors2 = ['orange' if ift == 0 else 'aqua' if ift==1 else 'white' for ift in colors]
            colors[colors==-1] = np.nan
            doyz = x
            npr_mid = (npr_th-npr_fr)*0.5+npr_fr  # the threshold value of npr
            npr_compare = np.interp(np.arange(1, 366), x[~np.isnan(npr_smap)], npr_smap[~np.isnan(npr_smap)])
            tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = \
            main(sno, [], sm_wind=7, mode='annual', tbob=orb_plot, sig0=7, order=1)  # result npr
            # smap_ft + our NPR
            # ax_ft.bar(x, np.zeros(x.size)+npr_compare*100, color=colors2, width=1, edgecolor="none")
            # ax_ft.plot(npr1[0], npr1[1]*100, 'r-', markersize=3)
            # smap_ft + smap NPR
            ax_ft.bar(x, npr_smap*1e2, color=colors2, width=1, edgecolor="none")
            ax_ft.plot(x, npr_compare*1e2, 'k-', markersize=3)
            # ax_ft.axhline(y=npr_th*100, ls=':')
            # ax_ft.axhline(y=npr_fr*100, ls=':')
            ax_ft.axhline(y=npr_mid*100, ls='--')

            # ax_ft.plot(npr1[0], npr1[1]*100 , 'k-')
            ax_ft.set_ylim([0, 6])
            ax_ft.set_xlim([0, 366])
            figname2 = 'new_smap_ft_compare_%s' % sno
            plt.savefig(figname2)
            if subplot_a > 1:
                plt.close()
                return 0, 0
            else:
                return ax_ft

        # plotting for landcover type
        if ldcover is not False:
            fig1 = plt.figure()  #
            # num = len(ft10_dict.keys())
            site_nos = ['1175', '1177', '1233', '2065', '2213']

            num = max(2, len(sno_dict1[sno]))
            for r0, key0 in enumerate(sno_dict1[sno]):  # enumerate(ft10_dict.keys()):
                ax_ft = fig1.add_subplot(num, 1, r0+1)
                colors = ft10_dict[key0]
                colors2 = ['orange' if ift == 0 else 'aqua' if ift==1 else 'white' for ift in colors]
                colors[colors==-1] = np.nan
                doyz = x
                npr_fr = (npr_ref[0][key0])
                npr_th = (npr_ref[1][key0])
                npr_mid = (npr_th-npr_fr)*0.5+npr_fr  # the threshold value of npr
                npr_smap0_tp = npr_ref[2][key0]
                npr_smap0_tp[npr_smap0_tp==-1] = np.nan
                npr_smap0=npr_smap0_tp*1e-2
                npr_compare = np.interp(np.arange(1, 366), x[~np.isnan(npr_smap0)], npr_smap0[~np.isnan(npr_smap0)])
                tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = \
                main(sno, [], sm_wind=7, mode='annual', tbob=orb_plot, sig0=7, order=1, value_series=[x, npr_compare])  # result npr
                # smap_ft + our NPR
                # ax_ft.bar(x, np.zeros(x.size)+npr_compare*100, color=colors2, width=1, edgecolor="none")
                # ax_ft.plot(npr1[0], npr1[1]*100, 'r-', markersize=3)
                # smap_ft + smap NPR
                ax_ft.bar(x, npr_smap0*1e2, color=colors2, width=1, edgecolor="none")
                ax_ft.plot(x, npr_compare*1e2, 'k-', markersize=3)
                ax_ft.axhline(y=npr_th, ls=':')
                ax_ft.axhline(y=npr_fr, ls=':')
                ax_ft.axhline(y=npr_mid, ls='--')
                # add onset vline
                ax_ft.axvline(x=ons1[0], color='k', ls='--')
                ax_ft.axvline(x=ons1[1], color='k', ls='--')
                # ax_ft.plot(npr1[0], npr1[1]*100 , 'k-')
                text_tp = float(key0)
                key_text = 'Distance to station = %.2f km' % (text_tp)
                ax_ft.text(0.02, 0.95, key_text, transform=ax_ft.transAxes, va='top', fontsize=16)
                ax_ft.set_ylim([0, 5])
                ax_ft.set_xlim([0, 366])
                ax_ft.yaxis.set_major_locator(MaxNLocator(5))
            figname2 = 'result_07_01/temp_result/all_pixel_SMAP_compare_%s' % sno
            plt.rcParams.update({'font.size': 16})
            plt.tight_layout()
            plt.savefig(figname2)
            plt.close()
    return [vline01r, vline0r, vline11r, vline1r], float(key_nn)*100


def npr_smap_compare(pixel_id, ft10, ft10_dict, period, sno='947', compare=False, orb_plot='_A_', sec_dict=[],
                    npr_ref=False, ldcover=False, subplot_a=1):
    # npr_fr = (npr_ref[0][key_nn])*1e-2
    # npr_th = (npr_ref[1][key_nn])*1e-2
    # npr_smap = npr_ref[2][key_nn]*1e2
    # npr_smap[npr_smap==-1] = np.nan
    # npr_smap*=1e-2
    # N_min, N_max = np.nanmin(npr1[1]), np.nanmax(npr1[1])
    # npr_smap_normed = 20+(30-20)*(npr_smap-N_min)/(N_max-N_min)
    # npr_smap_date = np.arange(1, 366)
    # l_npr_smap = ax01.plot(npr_smap_date, npr_smap_normed, 'k-')
    # npr_ref_normed = [20+(30-20)*(npri-N_min)/(N_max-N_min)
    #                   for npri in [npr_fr, npr_th, (npr_th-npr_fr)*0.5+npr_fr]]
    keysf = np.array([float(k) for k in ft10_dict.keys()])
    key_nn = ft10_dict.keys()[np.argmin(keysf)]
    x = np.arange(1, ft10.shape[0]+1)
    npr_fr = (npr_ref[0][key_nn])*1e-2
    npr_th = (npr_ref[1][key_nn])*1e-2
    npr_smap = npr_ref[2][key_nn]
    npr_smap[npr_smap==-1] = np.nan
    npr_smap*=1e-2
    # plotting
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    ax_ft = plt.subplot(2, 1, subplot_a)
    # check output
    print 'plot %s in row %d' % (sno, subplot_a)

    colors = ft10_dict[key_nn]
    colors2 = ['orange' if ift == 0 else 'aqua' if ift==1 else 'white' for ift in colors]
    colors[colors==-1] = np.nan
    doyz = x
    npr_mid = (npr_th-npr_fr)*0.5+npr_fr  # the threshold value of npr
    npr_compare = np.interp(np.arange(1, 366), x[~np.isnan(npr_smap)], npr_smap[~np.isnan(npr_smap)])
    tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = \
    main(sno, [], sm_wind=7, mode='annual', tbob=orb_plot, sig0=7, order=1)  # result npr
    ax_ft.bar(x, npr_smap*1e2, color=colors2, width=1, edgecolor="none")
    ax_ft.plot(x, npr_compare*1e2, 'k-', markersize=3)
    # ax_ft.axhline(y=npr_th*100, ls=':')
    # ax_ft.axhline(y=npr_fr*100, ls=':')
    l2d_npr_threshold = ax_ft.axhline(y=npr_mid*100, color='k', ls=':', lw='3')  # NPR threshold
    ax_ft.axvline(x=ons1[0], color='k', ls='--')  # in situ onset
    l2d_npr_onset = ax_ft.axvline(x=ons1[1], color='k', ls='--')
    v_file ='./result_07_01/txtfiles/result_txt/smap_ft_compare_%s.csv' % orb_plot[1]  # vline from txt
    onset_array = np.loadtxt(v_file, delimiter=',')
    onset_row = onset_array[onset_array[:, 10] == int(sno), :][0]
    ths, frs = \
        [onset_row[4], onset_row[6], onset_row[11]], [onset_row[5], onset_row[7], onset_row[13]]  # npr, site, smap


    # ax_ft.plot(npr1[0], npr1[1]*100 , 'k-')
    ax_ft.set_ylim([0, 6])
    ax_ft.set_xlim([0, 366])  # ax1.set_ylabel('T$_B$ (K)')

    # label or text box
    plot_id = ['a', 'b']
    ax_ft.text(0.05, 0.95, plot_id[subplot_a-1], transform=ax_ft.transAxes, va='top', ha='right', fontsize=16)
    site_name = site_infos.change_site(sno, names=sno)
    text_site = '%s(ID: %s)' % (site_name, sno)
    ax_ft.text(0.95, 0.95, text_site, transform=ax_ft.transAxes, va='top', ha='right', fontsize=16, bbox=dict(facecolor='white', alpha=1))  # text
    ax_ft.set_ylabel('NPR ($10^{-2}$)')
    plt.rcParams.update({'font.size': 16})  # set the font size
    # set the legend
    if subplot_a > 1:  # x label for the bottom figure
        ax_ft.set_xlabel('Day of year 2016')
        figname2 = 'new_smap_ft_compare_%s' % sno
        # plt.tight_layout()
        # plt.savefig(figname2)
        # plt.close()
    else:
        aqua_patch = Patch(edgecolor='none', facecolor='aqua')
        orange_patch = Patch(edgecolor='none', facecolor='orange')
        plt.legend((aqua_patch, orange_patch, l2d_npr_onset, l2d_npr_threshold),
                   ('Frozen (SMAP L3_FT_P)', 'Thawed (SMAP L3_FT_P)', 'Onset (Edge detection)', 'NPR threshold (SMAP L3_FT_P)'),
                   bbox_to_anchor=[0., 1.02, 1., .102], loc=3, ncol=2, mode='expand', borderaxespad=0., prop={'size': 12})
    if subplot_a > 1:
        figname2 = 'new_smap_ft_compare_%s' % sno
        plt.savefig(figname2, dpi=300)
        plt.close()
        return 0, 0
    else:
        return ax_ft, 0


def plot_patch_test():
    fig0 = plt.figure()
    ax = fig0.add_subplot(111)
    aqua_patch = Patch(edgecolor='none', facecolor='aqua')
    dash_arc = Arc((0.1, 0.1), width=.01, height=.1, ls='--')
    plt.legend((aqua_patch, dash_arc), ('Frozen', 'Thawed'),
                   bbox_to_anchor=[0., 1.02, 1., .102], loc=3, ncol=2, mode='expand', borderaxespad=0., prop={'size': 14})
    figname2 = 'new_smap_ft_compare_%s' % ('legend')
    # plt.tight_layout()
    plt.savefig(figname2)


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
            g_npr, i_gaussian = data_process.gauss_conv(edge_series, sig=s, n=w, sig2=s2)
        else:
            g_npr, i_gaussian = data_process.gauss_conv(edge_series, sig=s, n=w)  # option: ffnpr-t_h; var_npv-t_h
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


def edge_detect_iter(t_series, edge_series, s, order=1, seriestype='tb', is_sort=True, w=4, long_short=False):
    """
    :param t_series: from overpass second
    :param edge_series: ft indicators
    :param s: sigma of gaussian filter
    :param order: 1st detrivative of gaussian
    :param seriestype: the name of indicator.
    :param w: window in unit of std
    :return:
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
            p = 0
            g_npr, i_gaussian = data_process.gauss_conv(edge_series, sig=s, n=w, sig2=s/2)
        else:
            g_npr, i_gaussian = data_process.gauss_conv(edge_series, sig=s, n=w)  # option: ffnpr-t_h; var_npv-t_h
        if g_npr.size < 2:
            return np.array([[-999, -999, -999]]), np.array([[-999, -999, -999]]), \
                   np.zeros([2, t_series.size])[g_size: -g_size] - 999
        conv_valid = g_npr[g_size: -g_size]  # valid interval: g_size: -g_size
        max_gnpr, min_gnpr = peakdetect.peakdet(conv_valid, peaks_iter, t_series[i_gaussian][g_size: -g_size])
        # calculate the winter mean convolution as well as the snr
        t_valid = t_series[i_gaussian][g_size: -g_size]
        if t_valid.size < 1:
            p = 0
            t_valid = np.array([np.min(t_series[i_gaussian])])
        # i_winter = (t_valid > 1+365) & (t_valid < 60+365)
        # conv_winter = conv_valid[i_winter]
        # conv_noise = np.nanmean(np.abs(conv_winter))
        # snr = max_gnpr[:, -1]/conv_noise
        # max_npr_valid = max_gnpr[np.abs(snr) > snr_threshold]
        # snr = np.abs(min_gnpr[:, -1])/conv_noise
        # min_npr_valid = min_gnpr[np.abs(snr) > snr_threshold]
        max_npr_valid = max_gnpr
        min_npr_valid = min_gnpr
        if max_npr_valid.size < 1:
            max_npr_valid = np.array([[-1., t_series[0], -1.]])
        if min_npr_valid.size < 1:
            min_npr_valid = np.array([[-1., t_series[0], -1.]])
        return max_npr_valid, min_npr_valid, np.array([t_valid, conv_valid])


def edge_detect_v2(t_series, edge_series, s, order=1, peaks_iter=1e-1):
    """
    :param t_series: time variable of edge_series
    :param edge_series: time series, e.g., brightness temperature
    :param s: sigma of gaussian filter
    :param order: 1st detrivative of gaussian
    :return:
    """
    if order == 1:  # first order of gaussian
        g_size = 6*s/2
        g_conv, i_gaussian = gauss_conv(edge_series, sig=s)
        conv_valid = g_conv[g_size: -g_size]  # valid interval: g_size: -g_size
        maximums, minimums = peakdetect.peakdet(conv_valid, peaks_iter, t_series[i_gaussian][g_size: -g_size])
        t_valid = t_series[i_gaussian][g_size: -g_size]
        return maximums, minimums, np.array([t_valid, conv_valid])


def gauss_conv(series, sig=1, fill_v=-999):
    """
    :param series:
    :param sig:
    :param fill_v: filled value for unvalid data
    :return:
        ig: the indice of valid data
    """
    series[series == fill_v] = np.nan
    size = 6*sig+1
    ig = ~np.isnan(series)
    x = np.linspace(-size/2+1, size/2, size)
    filterz = ((-x)/sig**2)*np.exp(-x**2/(2*sig**2))
    if series[ig].size < 1:
        return -1, -1
    else:
        f1 = np.convolve(series[ig], filterz, 'same')
    return f1, ig





def get_peak(series, iter, t_x):
    """

    :param series: the time series of a given indicator
    :param iter: peaks greater than iter are returned
    :param t_x: the x axis for the time series (i.e., date)
    :return:
    """
    max_series, min_series = peakdetect.peakdet(series, iter, t_x)
    return max_series, min_series