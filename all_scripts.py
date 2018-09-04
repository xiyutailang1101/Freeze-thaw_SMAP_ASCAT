import spt_quick
import os, re
import data_process
import numpy as np
import csv
import test_def
from sys import exit as quit0
from datetime import datetime
from datetime import timedelta
from plot_funcs import pltyy
import plot_funcs
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from plot_funcs import plot_filter, plot_filter_series
import basic_xiyu as bxy
import glob
import sys
from shutil import copyfile
import read_site
import site_infos
import pytz
import Read_radar
import subprocess as sb
import h5py
from matplotlib import colors, cm, colorbar


def draw_pie_landcover():
    prefix = 'result_07_01/txtfiles/site_landcover/'
    save_direct = 'result_07_01/txtfiles/site_landcover_tp/'
    all_pixel_class = prefix+'class_all_SMAP_pixels.txt'
    site_nos = ['1177', '1233', '947', '2065', '967', '2213', '949', '950', '960', '962', '968', '1090', '1175', '2081', '2210', '1089',  '2212', '2211']
    #'1177',  '1233', '947', '2065', '967', '2213', '949', '950', '960', '962', '968',
    for site_no in site_nos:
        count0 = 0
        is_data, is_site = 0, 0
        line_num = 0
        # initials
        n = 0
        arr1 = np.zeros([100, 2])-1
        with open(all_pixel_class, 'rb') as class_all:
            for line in class_all:
                line_num += 1
                if 'site_'+site_no in line:
                    is_site = 1
                    print 'Line:%d This is the line for file_info' % count0
                    line_sp = line.split('.')
                    file_info = line_sp[0].split(' ')[-1]
                    pixel_dis = file_info.split('_')[-1]  # the distance from pixel center to site
                    continue
                if is_site > 0:
                    if 'Histogram' in line:
                        is_site = 0
                        is_data = 1
                        continue
                if is_data > 0:
                    x = re.split(r'\t', line)
                    arr1[n, 0], arr1[n, 1] = float(x[1]), float(x[4])
                    n+=1
                    if float(x[-1]) == 100: # the last line of data
                        # plot pies
                        i_0 = arr1[:, 0] == 11  # water
                        p_0 = np.sum(arr1[:, 1][i_0])
                        i_0 = arr1[:, 0] == 12  # snow
                        p_1 = np.sum(arr1[:, 1][i_0])
                        i_0 = arr1[:, 0] == 31  # barren
                        p_2 = np.sum(arr1[:, 1][i_0])
                        i_0 = (arr1[:, 0] > 40) & (arr1[:, 0] < 50)  # forest
                        p_3 = np.sum(arr1[:, 1][i_0])
                        i_0 = (arr1[:, 0] > 50) & (arr1[:, 0] < 60)  # shrub
                        p_4 = np.sum(arr1[:, 1][i_0])
                        i_0 = (arr1[:, 0] > 70) & (arr1[:, 0] < 90) # grass
                        p_5 = np.sum(arr1[:, 1][i_0])
                        i_0 = (arr1[:, 0] > 89) & (arr1[:, 0] < 92)  # wetland
                        p_6 = np.sum(arr1[:, 1][i_0])
                        i_0 = (arr1[:, 0] > 91)
                        p_7 = np.sum(arr1[:, 1][i_0])
                        pp = np.array([p_1, p_4, p_0, p_6, p_7, p_3, p_5, p_2])
                        i_0 = np.where(pp>5)
                        label0s = {'11': 'water', '12': 'perennial snow', '31': 'Barren Land', '41': 'Forest', '51': 'Shrub', '71': 'Grass', '90': 'wetland'}
                        label1s = ['Water', 'Perennial Snow', 'Barren Land', 'Forest', 'Shrub',  'Grass', 'wetland']
                        label_color = {'Water': 'aqua', 'Perennial Snow': 'snow', 'Barren Land': 'brown', 'Forest': 'forestgreen',
                                       'Shrub': 'olive',  'Grass': 'palegreen', 'wetland': 'steelblue', 'wetland2': 'cyan'}
                        labels = [label_color.keys()[i] for i in i_0[0]]  # labels
                        sizes = [pp[i] for i in i_0[0]]  # sizes
                        cc = [label_color[i] for i in labels]
                        labels.append('Other')
                        sizes.append(100-np.sum(sizes))
                        cc.append('gray')
                        fig1, ax1 = plt.subplots()
                        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=cc)
                        ax1.axis('equal')
                        fname  = 'pie_landcover_%s_%s.png' % (site_no, pixel_dis)
                        plt.savefig(save_direct+fname)
                        plt.close()
                        # initial again
                        n = 0
                        arr1 = np.zeros([100, 2])-1
                        is_data = 0
    quit0()
        # initials
        # arr1 = np.zeros([85, 2])-1
        # fname = prefix+site_no+'.txt'
        # i = 0
        # n = 0
        # # read data from txt
        # with open(fname, 'rb') as as0:
        #     for line in as0:
        #         i+=1
        #         if i > 7:
        #             print site_no, i
        #             x = re.split(r'\t', line)
        #             arr1[n, 0], arr1[n, 1] = float(x[1]), float(x[4])
        #             n+=1
        #             if n == 84:
        #                 pause = 0
        #         else:
        #             continue
        # # plot pies
        # i_0 = arr1[:, 0] == 11  # water
        # p_0 = np.sum(arr1[:, 1][i_0])
        # i_0 = arr1[:, 0] == 12  # snow
        # p_1 = np.sum(arr1[:, 1][i_0])
        # i_0 = arr1[:, 0] == 31  # barren
        # p_2 = np.sum(arr1[:, 1][i_0])
        # i_0 = (arr1[:, 0] > 40) & (arr1[:, 0] < 50)  # forest
        # p_3 = np.sum(arr1[:, 1][i_0])
        # i_0 = (arr1[:, 0] > 50) & (arr1[:, 0] < 60)  # shrub
        # p_4 = np.sum(arr1[:, 1][i_0])
        # i_0 = (arr1[:, 0] > 70) & (arr1[:, 0] < 90) # grass
        # p_5 = np.sum(arr1[:, 1][i_0])
        # i_0 = (arr1[:, 0] > 89)  # wetland
        # p_6 = np.sum(arr1[:, 1][i_0])
        # pp = np.array([p_1, p_4, p_0, p_6, p_3, p_5, p_2])
        # i_0 = np.where(pp>5)
        # label0s = {'11': 'water', '12': 'perennial snow', '31': 'Barren Land', '41': 'Forest', '51': 'Shrub', '71': 'Grass', '90': 'wetland'}
        # label1s = ['Water', 'Perennial Snow', 'Barren Land', 'Forest', 'Shrub',  'Grass', 'wetland']
        # label_color = {'Water': 'aqua', 'Perennial Snow': 'snow', 'Barren Land': 'brown', 'Forest': 'forestgreen',
        #                'Shrub': 'olive',  'Grass': 'palegreen', 'wetland': 'steelblue'}
        # labels = [label_color.keys()[i] for i in i_0[0]]  # labels
        # sizes = [pp[i] for i in i_0[0]]  # sizes
        # cc = [label_color[i] for i in labels]
        # labels.append('Other')
        # sizes.append(100-np.sum(sizes))
        # cc.append('gray')
        # fig1, ax1 = plt.subplots()
        # ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=cc)
        # ax1.axis('equal')
        # plt.savefig(prefix+'pie_landcover_'+site_no+'.png')


def ascat_within_tb(disref=[19, 9], subpixel=False):
    """
    disref: 0: for all tb_pixel, 1: for subpixel
    """
    prefix = './result_07_01/'
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1089', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968', '1090', '1175', '1177'],
                        'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
    daily_pass_folder = 'result_07_01/txtfiles/ascat_daily_pass/'
    all_subcenter = np.loadtxt('result_07_01/txtfiles/sub_tb/subc_all.txt', delimiter=',').T # the subcenters
    lat_all_tb, lon_all_tb, col_all_tb, row_all_tb, site_num = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    lat_all_ascat, lon_all_ascat = np.array([]), np.array([])
    lat_sub_ascat, lon_sub_ascat = np.array([]), np.array([])
    bb = 0  # index for boundary
    boundPoint = np.loadtxt('result_07_01/txtfiles/boundPoint.txt', delimiter=',')
    for site_no in site_nos:
        # get the subcenter of tb pixel
        i_tb = all_subcenter[2] == float(site_no)
        site_subcenter = all_subcenter[:, i_tb]
        sfolder0 = daily_pass_folder+'s'+site_no
        if not os.path.exists(sfolder0):
            os.makedirs(sfolder0)
        k_width = 7
        tbob = '_A_'
        tb_fname = prefix+'txtfiles/site_tb/tb_'+site_no+tbob+'2016.txt'
        with open(tb_fname, 'rb') as as0:
            reader = csv.reader(as0)
            for row in reader:
                if '#' in row[0]:
                    n_lon, n_lat = row.index(' cell_lon'), row.index('cell_lat')
                    n_row, n_col = row.index('cell_row'), row.index('cell_column')
                    break
            tb_mx = np.loadtxt(tb_fname)
            lon_36n, lat_36n = tb_mx[0, n_lon], tb_mx[0, n_lat]
            lat_all_tb = np.append(lat_all_tb, lat_36n)
            lon_all_tb = np.append(lon_all_tb, lon_36n)
            col_36n, row_36n = tb_mx[0, n_col].astype(int), tb_mx[0, n_row].astype(int)
            col_all_tb = np.append(col_all_tb, col_36n)
            row_all_tb = np.append(row_all_tb, row_36n)
            site_num = np.append(site_num, int(site_no))
            print col_36n, row_36n, lon_36n, lat_36n
            ease_lat_un = np.fromfile('/home/xiyu/Data/easegrid2/gridloc.EASE2_N36km/EASE2_N36km.lats.500x500x1.double', dtype=float).reshape(500, 500)
            ease_lon_un = np.fromfile('/home/xiyu/Data/easegrid2/gridloc.EASE2_N36km/EASE2_N36km.lons.500x500x1.double', dtype=float).reshape(500, 500)
            print ease_lon_un[row_36n, col_36n], ease_lat_un[row_36n, col_36n]
            txt_path = '/home/xiyu/PycharmProjects/R3/result_05_01/site_ascat/' + 's' + site_no + '/'
        period = ['ascat_20151101_'+site_no+'.npy', 'ascat_20151130_'+site_no+'.npy']
        filelist0 = sorted(os.listdir(txt_path))
        ind0 = [filelist0.index(period[0]), filelist0.index(period[1])]
        filelist_test = filelist0[ind0[0]: ind0[1]+1]  # test ascat file
        doy_test = []  # test day of year
        lon_valid0, lat_valid0 = np.array([]), np.array([])
        p_underline = re.compile('_')
        for file0 in filelist_test:
            file0_date = p_underline.split(file0)[1]
            doy_test.append(data_process.get_doy(file0_date))
            ascat_fname0 = txt_path + file0
            ascat_m0 = np.load(ascat_fname0)
            if ascat_m0.size < 1:
                continue
            lat_ascat, lon_ascat = ascat_m0[:, 0], ascat_m0[:, 1]

            # subpixel test
            if subpixel is not False:
                for sub in range(0, 9):
                    sub_lon, sub_lat = site_subcenter[0, sub], site_subcenter[1, sub]
                    dis_m0 = bxy.cal_dis(sub_lat, sub_lon, ascat_m0[:, 0], ascat_m0[:, 1])
                    i_sub = dis_m0 < disref[1]
                    if sum(i_sub)>0:
                        lon_sub_ascat = np.append(lon_sub_ascat, lon_ascat[i_sub])
                        lat_sub_ascat = np.append(lat_sub_ascat, lat_ascat[i_sub])

            dis_m0 = bxy.cal_dis(lat_36n, lon_36n, ascat_m0[:, 0], ascat_m0[:, 1])
            i25 = dis_m0 < disref[0]
            if sum(i25)>0:
                np.savetxt(sfolder0+'/'+file0_date+'.txt', np.append(lon_ascat[i25], lat_ascat[i25]).reshape(2, -1).T, fmt='%.4f', header='lon, lat')
                lon_valid0 = np.append(lon_valid0, lon_ascat[i25])
                lat_valid0 = np.append(lat_valid0, lat_ascat[i25])
            else:
                print 'no valid data in date %s' % file0_date
        lat_all_ascat = np.concatenate((lat_all_ascat, lat_valid0))
        lon_all_ascat = np.concatenate((lon_all_ascat, lon_valid0))
        dis_m1 = np.unique(bxy.cal_dis(lat_36n, lon_36n, lat_valid0, lon_valid0))
        a = 0
        # test the ascat within sub_pixels (0~8, clock-wise)
        for file0 in filelist_test:
            a = 0
        continue
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        i_max = np.argmax(dis_m1)
        ax0.scatter(lon_36n, lat_36n, c='r', marker='*')
        ax0.scatter(lon_valid0, lat_valid0, c='k', marker='o')
        ax0.scatter(lon_valid0[i_max], lat_valid0[i_max], c='r', marker='^')
        bound0 = boundPoint[bb]
        for i0 in range(0, 7, 2):
            ax0.scatter(bound0[i0+1], bound0[i0], marker='x', s=20, c='b')
        radius1 = np.sqrt((lon_36n-lon_valid0[i_max])**2+(lat_36n-lat_valid0[i_max])**2)
        print radius1, dis_m1[i_max]
        circle1 = plt.Circle((lon_36n, lat_36n), radius1)
        circle1.set_facecolor("none")
        ax0.add_artist(circle1)
        ax0.ticklabel_format(useOffset=False)
        ax0.set_xlim([lon_36n-1.5, lon_36n+1.5])
        ax0.set_ylim([lat_36n-1.5, lat_36n+1.5])
        plt.title(site_no)
        plt.savefig('test_ascat_within'+site_no+'.png')
    np.savetxt('tb_centers.txt',  np.concatenate((lon_all_tb, lat_all_tb, col_all_tb, row_all_tb, site_num)).reshape(5, -1).T, fmt='%.4f,%.4f,%d,%d,%d', delimiter=',')
    np.savetxt('ascat_within.txt', np.array([lon_all_ascat, lat_all_ascat]).reshape(2, -1).T, fmt='%.6f', delimiter=',', header='lon,lat')
    np.savetxt('ascat_within_sub.txt', np.array([lon_sub_ascat, lat_sub_ascat]).reshape(2, -1).T, fmt='%.6f', delimiter=',', header='lon,lat')


def test_grid():
    '''
    test the ease grid
    :return:
    '''
    lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy'), \
                                np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
    onset0 = np.random.rand(80, 90)
    data_process.pass_zone_plot(lons_grid, lats_grid, onset0, './result_05_01/onset_result/', fname='onset_1_smap_npr', prj='laea',
                                z_max=1, z_min=0, odd_points=[-144.54913393, 65.05981213])


def new_process(site_nos):
    # site_array = np.array([int(sno) for sno in site_nos])
    melt_date = np.zeros(len(site_nos))-1
    for i0, site_no in enumerate(site_nos):
        obv, obh, m1, m1_change = data_process.plot_obd(site_no, p='vh', isplot=False)
        st_id = np.where(m1_change[0] == 20)[0]
        a_lims = range(st_id, m1_change[0].size, 10)
        date_10 = m1_change[0][a_lims]
        change_10 = np.zeros(date_10.size)-99.0
        n=0
        for a in a_lims:
            change_10[n] = np.nansum(m1_change[1][a: a+10])
            n+=1
        # tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0
        tbv0, tbh0, npr0, gau0, ons0, _, _ = test_def.main(site_no, ['20160101', '20161225'], sm_wind=7, mode='annual', seriestype='tb', tbob='_A_', sig0=5)
        npr0[1][npr0[1]<-1] = np.nan
        m_time = (obv[0] > 80) & (obv[0] < 120)
        p = (np.where(obv[1][m_time]>0)[0].size+np.where(obh[1][m_time]>0)[0].size)*1.0/(obv[1][m_time].size+obh[1][m_time].size)
        # read more in situ
        doy = np.arange(1, 366) + 365
        si0 = site_no
        site_type = site_infos.get_type(si0)
        site_file = './copy0519/txt/'+site_type + si0 + '.txt'
        stats_t, tair5 = read_site.read_sno(site_file, "Air Temperature Observed (degC)", si0)
        tair_daily, tair_date = data_process.cal_emi(tair5, [], doy, hrs=18)
        if si0 in ['2065', '2081']:
            stats_t, tair5 = read_site.read_sno(site_file, "Air Temperature Average (degC)", si0)
            tair_daily, tair_date = data_process.cal_emi(tair5, [], doy, hrs=18)
        print type(site_no), site_no
        test_def.plot_snow_effect(npr0, npr0, obv, obh, m1, air_change=[tair_date-365, tair_daily],
                                  fname='./result_07_01/obd'+site_no+'npr_snow.png', sno=site_no)  # [date_10, change_10]
        date_melt = np.nanmin(m1[0][m1[1]<5])
        melt_date[i0] = date_melt
    # save_txt = np.concatenate((site_array, melt_date), axis=0).reshape(2, -1)
    # np.savetxt('result_07_01/test_melt_date.txt', save_txt.T, fmt='%d', delimiter=',')


def test_method(ft, txt=False, ft2='freeze'):
    # plot_filter()
    txtname = './result_07_01/methods/%s_ratio.csv' % ft
    if txt is False:
        site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177', '2210', '1089', '1233', '2212', '2211']
        twR = []
        twR_tb = []
        twR_ascat = []
        indic_npr, indic_tb, indic_ascat = '%s/npr' % ft, '%s/tb' % ft, '%s/ascat' % ft
        w_filter = np.arange(1, 10.4, 0.4)
        s = w_filter
        for site_no in site_nos:
            trans2winter_ratio, a_list = data_process.test_winter_trans(site_no, indic=indic_npr, w=w_filter)
            trans2winter_ratio_tb, b_list = data_process.test_winter_trans(site_no, indic=indic_tb, w=w_filter)
            trans2winter_ratio_ascat, c_list = data_process.test_winter_trans(site_no, indic=indic_ascat, w=w_filter)
            twR.append(trans2winter_ratio)
            twR_tb.append(trans2winter_ratio_tb)
            twR_ascat.append(trans2winter_ratio_ascat)
            # plot_filter_series(site_no, indic='tb')
            # plot_filter_series(site_no, indic='ascat', scale=2)
            # plot_filter_series(site_no, indic='npr')
        stat = 0
        snr_npr, snr_tb, snr_ascat = np.array(twR).T, np.array(twR_tb).T, np.array(twR_ascat).T
        mean_ascat = bxy.trim_mean(snr_ascat)
        mean_npr, mean_tb = bxy.trim_mean(snr_npr), bxy.trim_mean(snr_tb)

        print snr_npr.shape, mean_npr.shape
        # np.savetxt('thaw_ratio_npr.csv', np.append(snr_npr, mean_npr, axis=0), delimiter=',', fmt='%.4f')
        # np.savetxt('thaw_ratio_ascat.csv', np.append(snr_ascat, mean_ascat, axis=0), delimiter=',', fmt='%.4f')
        np.savetxt(txtname, np.array([mean_npr, mean_tb, mean_ascat]), delimiter=', ', fmt='%.4f')
    else:
        snrs = np.loadtxt(txtname, delimiter=',')
        mean_npr, mean_tb, mean_ascat = snrs[0], snrs[1], snrs[2]
        s = np.arange(1, 10.4, 0.4)
        txtname2 = './result_07_01/methods/%s_ratio.csv' % ft2
        snrs2 = np.loadtxt(txtname2, delimiter=',')
        mean_npr2, mean_tb2, mean_ascat2 = snrs2[0], snrs2[1], snrs2[2]
        mean_npr, mean_tb, mean_ascat =0.5*(mean_npr+mean_npr2), 0.5*(mean_tb+mean_tb2), 0.5*(mean_ascat+mean_ascat2)
    figs = plt.figure()
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    plt.plot(s, mean_npr, 'k-', label='$NPR$', linewidth=2.0)
    plt.plot(s, mean_tb, 'k--', label='$T_{bv}$', linewidth=2.0)
    plt.plot(s, mean_ascat, 'k:', label='$\sigma_{45}^0$', linewidth=2.0)
    plt.ylabel('$SNR$')
    plt.xlabel('s (days)')
    fig_fname = './result_07_01/methods/testmethod_%s_fig' % ft
    plt.xlim([1, 10])
    plt.ylim([0, 12])
    plt.rcParams.update({'font.size': 16})
    plt.legend(loc=0, prop={'size': 14})
    plt.savefig(fig_fname, dpi=300)
    plt.close()


def call_data_process():
    site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177', '2210', '1089', '1233', '2212', '2211']
    for site in site_nos:
        for pol in ['vh', 'sig']:
            print site, pol
            data_process.plot_obd(site, p=pol)

    plt.savefig('yaxis_test.png')

    # normalized
    data_process.ascat_onset_map(['AS', 'DES'],
                                 odd_point=[[-160.3533, 59.4904], [-152.05521, 61.62864], [-153.38525605, 60.80840073],
                                        [-153.48717832, 62.25746204], [-148.24980309, 61.72569201], [-148.41475430, 61.36947935],
                                        [-145.91968862, 64.71503588], [-153.50964993, 70.70806989], [-163.28627617, 68.65911367],
                                        [-162.24943920, 69.13594698], [-156.34578665, 67.67279153], [-157.73233549, 68.15037569],
                                        [-157.93985569, 69.16732504], [-151.20377421, 60.73384959], [-150.67175906, 60.04628798],
                                        [-156.28640511, 70.27669083], [-147.02034702, 68.58702006], [-147.55250323, 69.12119637],
                                        [-143.77940611, 69.13717664], [-143.60888602, 69.79829215], [-153.06530376, 70.50435315]])
    for m in ['area_6', 'area_8']:
        print 'test series of pixel in %s' % m
        for ob in ['DES']:
            for mo in ['_norm_']:
                print 'mode is %s, orbit is %s' % (mo, ob)
                data_process.ascat_result_test(m, mode=mo, key=ob, odd_rc=(89, 190))

    # smap
    # data_process.smap_alaska_onset(mode='npr')
    lons_grid = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
    lats_grid = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
    onset0 = np.load('./result_05_01/onset_result/smap_onset_1_2016_npr_AS.npy')
    data_process.pass_zone_plot(lons_grid, lats_grid, onset0, './result_05_01/onset_result/', fname='onset_1_smap_npr',
                                   z_max=360, z_min=250, odd_points=[-144.54913393, 65.05981213])
    data_process.ascat_onset_map(['AS', 'DES'], product='npr')

    #  69, -155.82855579, 67.40914083, 46, 61
    #  338, -144.54913393, 65.05981213, 47, 45
    for m in ['area_100']:
        print 'test series of pixel in %s' % m
        for ob in ['AS']:
            for mo in ['tb', 'npr']:
                data_process.smap_result_test(m, orbit=ob, odd_rc=(47, 45), mode=mo, ft='1')
    data_process.ascat_plot_series()
    data_process.smap_alaska_onset(std=4)
    data_process.smap_alaska_onset(mode='npr', std=4)

    data_process.ascat_onset_map(['AS', 'DES'], product='smap', odd_point=[-156.05106272, 70.47259330], mask=True)
    for m in ['area_8']:
        # 346, -159.98123961, 67.72333190, 45, 66
        # 319, -158.42869281, 67.48967273, 45, 64
        # 127, -159.82054134, 69.79921252, 51, 68
        # 126, -155.55604522, 68.44360398, 49, 62
        # 129, -157.05106272, 70.47259330, 54, 66
        print 'test series of pixel in %s' % m
        for ob in ['AS']:
            for mo in ['npr']:
                data_process.smap_result_test(m, orbit=ob, odd_rc=(54, 66), mode=mo, ft='0')


    # ob difference

    data_process.plot_obd('950', p='sig')

    site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177', '2210', '1089', '1233', '2212', '2211']
    for site in site_nos:
        for pol in ['vh', 'sig']:
            print site, pol
            data_process.plot_obd(site, p=pol)

    # edge detection method test
    site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177', '2210', '1089', '1233', '2212', '2211']
    twR = []
    for site_no in site_nos:
        print site_no
        trans2winter_ratio = data_process.test_winter_trans(site_no, indic='thaw/ascat', trans_date=[80, 80])
        twR.append(trans2winter_ratio)
        plot_filter_series(site_no, indic='tb')
        plot_filter_series(site_no, indic='ascat', scale=2.5)
        plot_filter_series(site_no, indic='npr')
    stat = 0
    np.savetxt('thaw_ratio_ascat.csv', np.array(twR).T, delimiter=',', fmt='%.4f')


def ascat_area_script():
    file_list = os.listdir('/home/xiyu/PycharmProjects/R3/result_05_01/ASCAT_AK')
    date_list = []
    p_underline = re.compile('_')
    for file in file_list:
        date_list.append(p_underline.split(file)[1])
    date_list = sorted(date_list)
    for da in date_list:
    # for da in ['20160301']:
        print 'Processing Alaska regional ASCAT data at %s' % da
        spt_quick.ascat_area_plot2(da)
        spt_quick.ascat_area_plot2(da, orbit_no=1)
    spt_quick.build_mask()

    # smap daily tb in Alaska
    file_list = os.listdir('/home/xiyu/PycharmProjects/R3/result_05_01/SMAP_AK')
    date_list = []
    p_underline = re.compile('_')
    for file in file_list:
        date_list.append(p_underline.split(file)[1][3: 13])
    date_list = sorted(date_list)
    order = 4
    for da in date_list[4: ]:
        print order
        order+=1
    # for da in ['20160301']:
        print 'Processing Alaska regional SMAP data at %s' % da
        spt_quick.smap_area_plot(da)
        #spt_quick.ascat_area_plot2(da, orbit_no=1)

    # smap daily tb in Alaska, New
    file_list = glob.glob('/home/xiyu/PycharmProjects/R3/result_05_01/SMAP_AK/smap_ak_as/*.npy')
    date_list = []
    p_underline = re.compile('_')
    for file in file_list:
        date_list.append(p_underline.split(file)[-1][3: 13])
    date_list = sorted(date_list)
    order = [date_list.index('2015.10.01'), date_list.index('2017.03.01')]
    for da in date_list[order[0]: order[1]+1]:
        print da
    # for da in ['20160301']:
        print 'Processing Alaska regional SMAP data at %s' % da
        spt_quick.smap_area_plot(da)

    # smap mask, calculate onset
    spt_quick.smap_mask()
    data_process.smap_alaska_onset()


def ascat_map():
    """
    20170922
    """
    std0 = 2.5
    odd_target = [-143.41559065, 60.45068596, 110, 164]
    region_code = 'area_5f'
    data_process.ascat_alaska_onset(norm=True, std=std0, version='new', target00=[odd_target[2], odd_target[3]])
    data_process.ascat_onset_map(['AS'], std=std0, version='new', odd_point=[odd_target[0], odd_target[1]])
    for m in [region_code]:
            print 'test series of pixel in %s' % m
            for ob in ['AS']:
                for mo in ['_norm_']:
                    print 'mode is %s, orbit is %s' % (mo, ob)
                    data_process.ascat_result_test(m, ft='1', mode=mo, key=ob, odd_rc=(odd_target[2], odd_target[3]), std=std0)
    for std0 in [5, 8]:
    # data_process.smap_alaska_onset(std=4)
    # data_process.smap_alaska_onset(mode='npr', std=4)
    #
    # data_process.ascat_onset_map(['AS'], product='npr', odd_point=[-156.05106272, 70.47259330])
    # data_process.ascat_onset_map(['AS'], product='tb', odd_point=[-156.05106272, 70.47259330])
    # sys.exit()-152.19497406, 60.74955388, 91, 199
        data_process.ascat_alaska_onset(norm=True, std=std0, version='new', target00=[odd_target[2], odd_target[3]])
        data_process.ascat_onset_map(['AS'], std=std0, version='new', odd_point=[odd_target[0], odd_target[1]])
        for m in [region_code]:
            print 'test series of pixel in %s' % m
            for ob in ['AS']:
                for mo in ['_norm_']:
                    print 'mode is %s, orbit is %s' % (mo, ob)
                    data_process.ascat_result_test(m, mode=mo, key=ob, odd_rc=(odd_target[2], odd_target[3]), std=std0)


def Alaska_ascat_and_smap():
    std0 = 7
    odd_target = [-149.33543542, 68.69714936, 53, 56]
    region_code = 'area_2'
    # data_process.smap_alaska_onset(std=std0, version='new')
    # data_process.smap_alaska_onset(mode='npr', std=std0, version='new')
    # lons_grid = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
    # lats_grid = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
    # onset0 = np.load('./result_05_01/onset_result/all_year_observation/smap_onset_0_2016_npr_AS_w'+str(std0)+'.npy')
    # data_process.pass_zone_plot(lons_grid, lats_grid, onset0, './result_05_01/onset_result/all_year_observation/', fname='test_new_npr',
    #                                z_max=180, z_min=50, odd_points=[odd_target[0], odd_target[1]])

    data_process.ascat_onset_map(['AS'], product='npr', odd_point=[odd_target[0], odd_target[1]], mask=False, version='new', std=std0)
    data_process.ascat_onset_map(['AS'], product='tb', odd_point=[odd_target[0], odd_target[1]], mask=False, version='new', std=std0)

    smap_date = data_process.get_doy('20151001')-365
    for m in [region_code]:
        print 'test series of pixel in %s' % m
        for ob in ['AS']:
            for mo in ['tb', 'npr']:
                data_process.smap_result_test(m, orbit=ob, odd_rc=odd_target, mode=mo, ft='0',
                                              version='new', ini_doy=smap_date, std=std0)
    # odd_target = [-143.41559065, 60.45068596, 110, 164]
    region_code = 'area_5'
    data_process.ascat_alaska_onset(norm=True, std=std0, version='new')
    data_process.ascat_onset_map(['AS'], std=std0, version='new', odd_point=[odd_target[0], odd_target[1]])
    for m in [region_code]:
            print 'test series of pixel in %s' % m
            for ob in ['AS']:
                for mo in ['_norm_']:
                    print 'mode is %s, orbit is %s' % (mo, ob)
                    data_process.ascat_result_test(m, ft='0', mode=mo, key=ob, odd_rc=odd_target, std=std0, version='new')


def ascat_sub_tb(ascat_series, sub_no, in_situ=False):
    prefix = './result_07_01/'
    save_path = './result_07_01/new_final/tb_subcenter/'
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1089', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    site_nos = ['1177']
    sha = {'947': [[90, 115], [60, 120]], '968': [[120, 145], [90, 150]], '1089': [100, 120]}
    site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968','1090', '1175', '1177'],
                        'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
    # ASCAT process
    n_pixel = []
    onset_save = []
    gau0_tb = []
    save_h5 = False
    # dsm_npr: for moisture change in that day
    for site_no in site_nos:
        # site read
        pp = False
        doy = np.arange(1, 366) + 365
        si0 = site_no
        site_type = site_infos.get_type(si0)
        site_file = './result_07_01/txtfiles/site_measure/'+site_type + si0 + '.txt'
        y2_empty = 0
        stats_sm, sm5 = read_site.read_sno(site_file, "Soil Moisture Percent -2in (pct)", si0)  # air tmp
        sm5_daily, sm5_date = data_process.cal_emi(sm5, y2_empty, doy, hrs=21)
        stats_t, t5 = read_site.read_sno(site_file, "Soil Temperature Observed -2in (degC)", si0)
        t5_daily, t5_date = data_process.cal_emi(t5, y2_empty, doy, hrs=21)
        if pp:
            stats_swe, swe = read_site.read_sno(site_file, "Precipitation Increment (mm)", si0)
        else:
            stats_swe, swe = read_site.read_sno(site_file, "snow", si0, field_no=-1)
        swe_daily, swe_date = data_process.cal_emi(swe, y2_empty, doy, hrs=0)
        sm5_daily[sm5_daily < -90], t5_daily[t5_daily < -90], \
        swe_daily[swe_daily < -90] = np.nan, np.nan, np.nan
        sm, tsoil, swe = [sm5_date-365, sm5_daily], [t5_date-365, t5_daily], [swe_date-365, swe_daily]
        ons_site, ons_tsoil,day2 = data_process.sm_onset(sm[0], sm[1], tsoil[1])
        gau1_npr, Emax_npr, dsm_npr, dswe_npr, dsoil_npr = [], [], [], [], []
        # center_tb = test_def.main(site_no, [], sm_wind=7, mode='annual', seriestype='tb', tbob='_A_', sig0=7, centers=True)

        for k_width in [7]:  # ,7, 8, 9, 10 1, 2.5, 3, 4, 5, 6, 7, 8, 9, 10
            print k_width
            precip = False
            sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g=k_width, pp=precip,
                                               txt_path=ascat_series)# 0 for ascending
            sigconv[0]-=365
            x_time, sigma0, date_list, out_ascat, inc45_55 = [], [], [], [], []
            print 'station ID is %s' % site_no
            # some extra process
            # date0, value0 = sm5[0], sm5[1]
            # index2016 = (date0>365)&(date0<730)&(value0>-90)&(np.abs(date0-365-267) >= 1)
            # sm5_daily, sm5_date = value0[index2016], date0[index2016]
            # stats_sm, rain = read_site.read_sno(site_file, "Air Temperature Observed (degC)", site_no)  # percipitation
            # rain_value, rain_date = rain[1][index2016], rain[0][index2016]
            tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual', seriestype='tb', tbob='_A_', sig0=k_width)  # result tb
            gau0_tb.append(gau0)
            tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = test_def.main(site_no, [], sm_wind=7, mode='annual', tbob='_A_', sig0=k_width)  # result npr
            gau1_npr.append(gau1)  # gau1: normalized E(t), peakdate: the date when E(t) reaches max/min
            ons_new.append(ons0[0]), ons_new.append(ons0[1]), ons_new.append(ons1[0]), ons_new.append(ons1[1])
            ons_new.append(ons_site[0]), ons_new.append(ons_site[1])
            # plot in situ
            if in_situ is not False:
                soil = [['Soil moisture (%)', sm[0], sm[1]], ['Soil temperature (DegC)', tsoil[0], tsoil[1]]]
                snow = ['SWE (mm)', swe[0], swe[1]]
                fig, (ax_soil, ax_swe) = plt.subplots(2, sharex=True, figsize=[8, 4])
                _, ax4_2, l2 = pltyy(soil[0][1], soil[0][2], 'test_comp2', 'VWC (%)',
                                         t2=soil[1][1], s2=soil[1][2], label_y2='T$_{soil}$ ($^\circ$C)',
                                         symbol=['k-', 'b-'], handle=[fig, ax_soil], nbins2=6, label_x='')
                for ax_2 in [ax4_2]: # ax1_2, ax2_2, ax3_2,
                    ax_2.axhline(ls=':', lw=1.5)
                ax_swe.plot(snow[1], snow[2], 'k', linewidth=2.0)
            if site_no in ['947', '949', '950', '967', '1089']:
                ax_swe.set_ylabel('SWE (mm)')
                ax_swe.set_ylim([0, 200])
            else:
                ax_swe.set_ylabel('SD (cm)')
                ax_swe.set_ylim([0, 100])
            if site_no in ['950', '1089']:
                ax_swe.set_ylim([0, 400])
            for ax in [ax_soil, ax_swe]:
                ax.axvspan(100, 150, color=(0.8, 0.8, 0.8), alpha=0.5, lw=0)
            plt.savefig(save_path+'in_situ_'+site_no)
            # plot ascat only
            site_lim = {'947': [-14, -7], '949': [-13, -7], '950': [-13, -7], '960': [-14, -8], '962': [-15, -8], '967': [-12, -8], '968': [-17, -8],
                '1089': [-15, -7], '1090': [-14, -7], '1175': [-15, -8], '1177': [-19, -10],
                '1233': [-17, -9], '2065': [-14, -8], '2081': [-15, -7], '2210': [-16, -8], '2211': [-16, -8], '2212': [-16, -8],
                '2213': [-17, -10]}
            sigma = [[sigseries[0], sigseries[1]], [sigconv[0], sigconv[1]]],  # row 3 sigma)
            fig = plt.figure(figsize=[6, 3])
            ax_ascat = fig.add_subplot(111)
            ax_ascat.plot(sigseries[0], sigseries[1], 'k-^')
            ax_ascat.set_xlim([100, 150])
            # _, ax3_2, l2 = plot_funcs.pltyy(sigma[0][0], sigma[0][1], 'test_comp2', '$\sigma^0$',
            #                  t2=sigma[1][0], s2=sigma[1][1], label_y2='E$(\\tau)_{\sigma^0}$',
            #                  symbol=[s_symbol, 'g-'], handle=[fig, ax3], nbins2=6)
            ax_ascat.set_ylim(site_lim[site_no])
            ax_ascat.tick_params(axis='both', which='major', labelsize=18)
            ax_ascat.locator_params(axis='y', nbins=6)
            # ax_ascat.set_ylabel('$\sigma^0$ (dB)')
            plt.savefig(save_path+'ascat'+site_no+'_p'+sub_no)
            continue
            test_def.plt_npr_gaussian_all([tbv0, tbh0, gau0],  # row 1, tb
                                 [npr1, gau1],  # row 2, npr
                                 [[sigseries[0], sigseries[1]],
                                  [sigconv[0], sigconv[1]]],  # row 3 sigma
                                 [['Soil moisture (%)', sm[0], sm[1]],  # row4 temp/moisture
                                  # swe_date, swe_daily
                                  ['Soil temperature (DegC)', tsoil[0], tsoil[1]]],
                                 ['SWE (mm)', swe[0], swe[1]], ons_new, # row5 swe/percipitation, onset
                                 figname=prefix+'all_plot_'+site_no+'_'+str(k_width)+'.png', size=(8, 6), xlims=[0, 365],
                                 title=site_no, site_no=site_no, pp=precip, s_symbol='k.')
            ons_new.append(int(site_no))
            onset_save.append(ons_new)
    return 0


def ascat_sub9(site_nos, sub_dir):
    """

    :param site_nos:
    :param sub_dir:
    :return:
    """
    for site_no_subs in site_nos:
        ascat_record = []
        ascat_onset = []
        for site_no in site_no_subs:
            sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series(str(site_no), orb_no=0, inc_plot=False, sigma_g=7, pp=False,
                                               txt_path=sub_dir)# 0 for ascending
            ascat_record.append(sigseries)
            ascat_onset.append(ons_new)
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex='col', sharey='row')
        axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        i = -1
        for axi in axs:
            # print ascat_record[i][0][2], ascat_record[i][1][2]
            i += 1
            axi.plot(ascat_record[i][0], ascat_record[i][1], 'k-^', markersize=3)
            axi.axvline(x=ascat_onset[i][0], color='k', ls='-.')
            axi.axvline(x=ascat_onset[i][1], color='k', ls='-.')
            axi.set_xlim([0, 365])
            # ylimits = site_infos.ascat_site_lim(site_no)
            # axi.set_ylim([ylimits[0], ylimits[1]])
            axi.tick_params(axis='y', which='minor')
            axi.tick_params(axis='x', which='minor')
            axi.locator_params(axis='y', nbins=6)
            axi.locator_params(axis='x', nbins=4)
        png_name = 'test_sub_%s.png' % site_no
        plt.savefig(png_name, dpi=150)
        plt.close()
        ## the angular dependencies
        # f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex='col', sharey='row')
        # axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        # i = -1
        # for axi in axs:
        #     i += 1
        #     axi.plot(ascat_record[i][2], ascat_record[i][1], '^', markersize=3)
        #     ylimits = site_infos.ascat_site_lim(site_no)
        #     axi.set_ylim([ylimits[0], ylimits[1]])
        #     axi.locator_params(axis='y', nbins=6)
        #     axi.locator_params(axis='x', nbins=4)
        # png_name = 'test_angular_sub_%s.png' % site_no
        # plt.savefig(png_name, dpi=150)
        # plt.close()
    return 0


def disscus_sm_variation(sno='1233'):
    save_path = './result_07_01/new_final/'
    # site_nos = ['947', '949', '950', '960', '962', '967', '968', '1089', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    site_nos = [sno]
    gau0_tb = []
    for site_no in site_nos:
        # site read
        pp = False
        doy = np.arange(1, 366) + 365
        si0 = site_no
        site_type = site_infos.get_type(si0)
        site_file = './result_07_01/txtfiles/site_measure/'+site_type + si0 + '.txt'  # ascat
        y2_empty = 0
        stats_sm, sm5 = read_site.read_sno(site_file, "Soil Moisture Percent -2in (pct)", si0)  # air tmp
        sm5_daily, sm5_date = data_process.cal_emi(sm5, y2_empty, doy, hrs=21)
        stats_t, t5 = read_site.read_sno(site_file, "Soil Temperature Observed -2in (degC)", si0)
        t5_daily, t5_date = data_process.cal_emi(t5, y2_empty, doy, hrs=21)
        stats_swe, swe = read_site.read_sno(site_file, "snow", si0, field_no=-1)
        swe_daily, swe_date = data_process.cal_emi(swe, y2_empty, doy, hrs=0)
        sm5_daily[sm5_daily < -90], t5_daily[t5_daily < -90], \
        swe_daily[swe_daily < -90] = np.nan, np.nan, np.nan
        sm, tsoil, swe = [sm5_date-365, sm5_daily], [t5_date-365, t5_daily], [swe_date-365, swe_daily]
        ons_site, ons_tsoil, day2 = data_process.sm_onset(sm[0], sm[1], tsoil[1])
        gau1_npr, Emax_npr, dsm_npr, dswe_npr, dsoil_npr = [], [], [], [], []
        plt_col = -1
        for k_width in [7]:  # ,7, 8, 9, 10 1, 2.5, 3, 4, 5, 6, 7, 8, 9, 10
            plt_col += 1
            print k_width
            precip = False
            sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g=k_width, pp=precip)# 0 for ascending
            sigconv[0]-=365
            conv_freeze = peakdate_sig[1]
            conv_250_350 = conv_freeze[(conv_freeze[:, 1] > 250) & (conv_freeze[:, 1] < 350)]
            i1 = bxy.gt_nstd(conv_250_350, peakdate_sig[1][:, -1], 2)
            vline4 = [conv_250_350[i1, 1]]
            x_time, sigma0, date_list, out_ascat, inc45_55 = [], [], [], [], []
            print 'station ID is %s' % site_no
            # some extra process
            # date0, value0 = sm5[0], sm5[1]
            # index2016 = (date0>365)&(date0<730)&(value0>-90)&(np.abs(date0-365-267) >= 1)
            # sm5_daily, sm5_date = value0[index2016], date0[index2016]
            # stats_sm, rain = read_site.read_sno(site_file, "Air Temperature Observed (degC)", site_no)  # percipitation
            # rain_value, rain_date = rain[1][index2016], rain[0][index2016]
            tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual', seriestype='tb', tbob='_A_', sig0=k_width)  # result tb
            gau0_tb.append(gau0)
            conv_freeze = peakdate0[0]
            conv_250_350 = conv_freeze[(conv_freeze[:, 1] > 250) & (conv_freeze[:, 1] < 350)]
            i1 = bxy.gt_nstd(conv_250_350, peakdate0[0][:, -1], 0)
            vline2 = conv_250_350[i1, 1]

            tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = test_def.main(site_no, [], sm_wind=7, mode='annual', tbob='_A_', sig0=k_width)  # result npr
            gau1_npr.append(gau1)  # gau1: normalized E(t), peakdate: the date when E(t) reaches max/min
            conv_freeze = peakdate1[1]
            conv_250_350 = conv_freeze[(conv_freeze[:, 1] > 250) & (conv_freeze[:, 1] < 350)]
            i1 = bxy.gt_nstd(conv_250_350, peakdate1[1][:, -1], 0)
            vline3 = conv_250_350[i1, 1]

            ons_new.append(ons0[0]), ons_new.append(ons0[1]), ons_new.append(ons1[0]), ons_new.append(ons1[1])
            ons_new.append(ons_site[0]), ons_new.append(ons_site[1])
            # plotting:
            vline1 = np.array([292, 314])
            plt.figure(figsize=[10, 7.5])
            v_symbol = ['-', '--']
            l_symbol = ['k-', 'g-']
            params = {'mathtext.default': 'regular'}
            plt.rcParams.update(params)
            # if plt_col == 0:
            #     ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
            #     soil = [['Soil moisture (%)', sm[0], sm[1]], ['Soil temperature (DegC)', tsoil[0], tsoil[1]]]
            #     _, ax4_2, l2 = pltyy(soil[0][1], soil[0][2], 'test_comp2', 'VWC (%)',
            #                              t2=soil[1][1], s2=soil[1][2], label_y2='T$_{soil}$ ($^\circ$C)',
            #                              symbol=['k-', 'b-'], handle=[0, ax1], nbins2=6, label_x='')
            #     ax1.set_xlim([250, 365])
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            soil = [['Soil moisture (%)', sm[0], sm[1]], ['Soil temperature (DegC)', tsoil[0], tsoil[1]]]
            _, ax1_2, l2 = pltyy(soil[0][1], soil[0][2], 'test_comp2', 'VWC (%)',
                                     t2=soil[1][1], s2=soil[1][2], label_y2='T$_{soil}$ ($^\circ$C)',
                                     symbol=['k-', 'b-'], handle=[0, ax1], nbins2=6, label_x='')
            ax1.set_xlim([250, 365])

            ax2 = plt.subplot2grid((2, 2), (0, 1))  # tb
            tb = [tbv0, tbh0, [gau0[0], gau0[2]]]
            _, ax2_2, l2 = pltyy(tb[0][0], tb[0][1], 'test_comp2', 'T$_{bv}$ (K)',
                             t2=tb[2][0], s2=tb[2][1], label_y2='$E_{Tbv}$ (K/day)',
                             symbol=l_symbol,
                             handle=[0, ax2], nbins2=6, label_x='')

            ax3 = plt.subplot2grid((2, 2), (1, 0))  # npr
            npr = [[npr1[0], npr1[1]*100], [gau1[0], gau1[2]*100]]
            _, ax3_2, l3 = pltyy(npr[0][0], npr[0][1], 'test_comp2', 'NPR ($10^{-2}$)',
                         t2=npr[1][0], s2=npr[1][1], label_y2='$E_{NPR} $ ($10^{-2}/day$)',
                         symbol=l_symbol, handle=[0, ax3], nbins2=6, label_x='')
            ax3.locator_params(axis='y', nbins=5)

            ax4 = plt.subplot2grid((2, 2), (1, 1))  # ascat
            sigma = [[sigseries[0], sigseries[1]],
                              [sigconv[0], sigconv[2]]]
            _, ax4_2, l4 = pltyy(sigma[0][0], sigma[0][1], 'test_comp2', '$\sigma_{45}^0$ (dB)',
                                 t2=sigma[1][0], s2=sigma[1][1], label_y2='$E_{\sigma_{45}^0}$ (dB/day)',
                                 symbol=l_symbol, handle=[0, ax4], nbins2=6, label_x='')
            ax4.locator_params(axis='y', nbins=4)
            i0 = 0
            for axi in [ax2, ax3, ax4]:
                i0 += 1
                axi.set_xlim([250, 365])
                if i0 < 3:
                    # axi.get_xaxis().set_visible(False)
                    status = 1
            axes = [ax1, ax2, ax3, ax4]
            vlines = [vline1, vline2, vline3, vline4]
            vlines = [[0, 0], ons0, ons1, [0, ons_new[1]]]

            text_ur = ['a', 'b', 'c', 'd']
            for j, ax0 in enumerate(axes):
                print j
                ax0.get_yaxis().set_label_coords(-0.18, 0.5)
                ax0.set_xlabel('DOY 2016')
                for i, x0 in enumerate(vlines[j]):
                    ax0.text(0.90, 0.95, text_ur[j], transform=ax0.transAxes, va='top', size=16)
                    ax0.axvline(x=x0, color='k', ls=v_symbol[i])
            for ax02 in [ax3_2, ax4_2]:
                ax02.get_yaxis().set_label_coords(1.2, 0.5)
            for ax01 in [ax1, ax1_2, ax2, ax2_2, ax3, ax3_2, ax4_2]:
                ax01.yaxis.set_major_locator(MaxNLocator(7))
            # read onest from onset file
            onset_value = site_infos.site_onset(sno)
            insitu_frz = onset_value[0][7]
            ax1.axvline(x=insitu_frz, color='r', ls=v_symbol[1])

            plt.rcParams.update({'font.size': 16})
            # add labels. updated 0514/2018
            insitu_frz2 = 313

            x_length = 365 - 250 + 1
            v_text = [v[1] for v in vlines[1:]]
            print v_text
            # ax2.text(0.568965517241, 1.05, '315', transform=ax2.transAxes, va='top', size=16)
            for ax00, v00 in zip([ax2, ax3, ax4], v_text):
                x_text = (v00*1.0-250+1)/x_length
                y_text = 1.08
                print x_text, y_text
                ax00.text(x_text, y_text, 'DOY '+str(int(v00)), transform=ax00.transAxes, va='top', ha='center', size=16)
            ax1.text((insitu_frz*1.0-250)/x_length, y_text, 'DOY '+str(int(insitu_frz)), transform=ax1.transAxes, va='top', ha='center', size=16)
            # ax1.text((insitu_frz2*1.0-250)/x_length, y_text, str(int(insitu_frz2)), transform=ax1.transAxes, va='top', ha='center', size=16)
        plt.tight_layout()
        plt.savefig('test00', dpi=300)
        plt.close()
    return 0


def discuss_combining(site_nos = ['2213']):
    # site_nos = ['2213']
    text4 = ['a', 'b', 'c']
    axs=[]
    fig0 = plt.figure(0, figsize=[6, 5.4])
    l_symbol = ['k-', 'b-']
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    # if plt_col == 0:
    #     ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    #     soil = [['Soil moisture (%)', sm[0], sm[1]], ['Soil temperature (DegC)', tsoil[0], tsoil[1]]]
    #     _, ax4_2, l2 = pltyy(soil[0][1], soil[0][2], 'test_comp2', 'VWC (%)',
    #                              t2=soil[1][1], s2=soil[1][2], label_y2='T$_{soil}$ ($^\circ$C)',
    #                              symbol=['k-', 'b-'], handle=[0, ax1], nbins2=6, label_x='')
    #     ax1.set_xlim([250, 365])
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    k_width = 7

    for i, site_no in enumerate(site_nos):
        i+=1
        sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g=k_width) # 0 for ascending

        tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = test_def.main(site_no, [], sm_wind=7, mode='annual', tbob='_A_', sig0=k_width)  # result npr
        ax0 = plt.subplot2grid((4, 1), (i, 0))
        axs.append(ax0)
        npr = [[npr1[0], npr1[1]*100], [gau1[0], gau1[2]*100]]
        _, ax0_2, l2 = pltyy(npr[0][0], npr[0][1], 'test_comp2', 'NPR ($10^{-2}$)',
                     t2=sigseries[0], s2=sigseries[1], label_y2='$\sigma_{45}^0$ (dB)',
                     symbol=l_symbol, handle=[0, ax0], nbins2=6, label_x='')
        # plot snow
        xlimit = [50, 150]
        swe_value, swe_date = read_site.read_measurements(site_no, 'snow', np.arange(366, 366+365), hr=0)
        swe_date-=365
        ax_sn = ax0.twinx()
        ax_sn.spines["right"].set_position(("axes", 1.35))
        plot_funcs.make_patch_spines_invisible(ax_sn)
        ax_sn.spines["right"].set_visible(True)

        # add in situ
        air_temp, air_t_date = read_site.read_measurements(site_no, "Air Temperature Observed (degC)",np.arange(366, 366+365), hr=18)
        if site_no in ['2213', '1090']:
            # add tbv and tbd
            ax_vh = plt.subplot2grid((4, 1), (i+1, 0))
            # obd_v, obd_h, m1, m1_change = data_process.plot_obd(site_no, p='vh', isplot=False)
            obd_v, obd_h, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                         seriestype='tb', sig0=k_width, order=1)  # result tb
            l1, = ax_vh.plot(obd_v[0], obd_v[1], 'ko', markersize=3)
            # ax_vh.plot(np.arange(60, 150), np.arange(60, 150)*0+1, 'ko', markersize=3)
            l11 = plot_funcs.plt_more(ax_vh, obd_h[0], obd_h[1], line_list=[l1], symbol='ro')
            ax_vh.axhline(ls=':', lw=1.5)
            ax_vh.legend(l11, ['V-pol', 'H-pol'], bbox_to_anchor=(1., 1), loc=2, prop={'size': 10}, numpoints=1)
            ax_vh.set_xlim(xlimit)
            ax_vh.set_ylim([190, 280])
            ax_vh.set_ylabel("$T_{b}$ (K)")
            ax_vh.yaxis.set_major_locator(MaxNLocator(4))
            ax_vh.text(0.92, 0.2, 'b', transform=ax_vh.transAxes, va='top')
            xticks = ax_vh.axes.get_xticklabels()
            for x_tick in xticks:
                x_tick.set_visible(False)
            # plot_funcs.make_patch_spines_invisible(ax_vh)

            # add in situ 07/2018
            soil_t, soil_t_date = read_site.read_measurements(site_no, "Soil Temperature Observed -2in (degC)",
                                                              np.arange(366, 366+365), hr=18)
            soil_sm, soil_sm_date = read_site.read_measurements(site_no, "Soil Moisture Percent -2in (pct)",
                                                                np.arange(366, 366+365), hr=18)
            soil_t_date-=365
            soil_sm_date-=365
            ax_3rd = plt.subplot2grid((4, 1), (i+1+1, 0))
            axs.append(ax_3rd)
            _, ax_3rd2, l2 = pltyy(soil_sm_date, soil_sm, 'test_comp2', 'VWC (%)',
                                 t2=soil_t_date, s2=soil_t, label_y2='T$_{soil}$ ($^\circ$C)',
                                 symbol=['k-', 'b-'], handle=[0, ax_3rd], nbins2=6)
            ax_3rd.set_xlim(xlimit)
            ax_3rd.set_ylim([0, 60])
            ax_3rd2.set_ylim([-30, 10])
            ax_3rd2.axhline(y=0, ls='--')
            ax_3rd.text(0.92, 0.2, 'c', transform=ax_3rd.transAxes, va='top')

        # fill colors 07/2018
        normalize = colors.Normalize(vmin=air_temp[air_temp>-15].min(), vmax=air_temp[air_temp>15].min())
        cmap = plt.get_cmap('coolwarm')
        swe_target = (swe_date>0) & (swe_date<150)
        k = 1
        for j in range(swe_date[swe_target].size/k-1):
            ax_sn.fill_between([swe_date[swe_target][j], swe_date[swe_target][j+k]], [swe_value[swe_target][j], swe_value[swe_target][j+k]],
                               color=cmap(normalize(air_temp[j])))
        # ax_sn.fill_between(swe_date, 0, swe_value, facecolor='grey', alpha=0.6)

        # plot snow
        l3, = ax_sn.plot(swe_date, swe_value, 'k:', label="snow depth")
        ax_sn.set_ylim([0, 150])
        ax_sn.yaxis.set_major_locator(MaxNLocator(4))
        ax0.set_xlim(xlimit)
        ax0.set_ylim([-9, 8])
        ax0.yaxis.set_major_locator(MaxNLocator(4))
        ax0_2.set_ylim([-22, -4])
        yticks = ax0.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        yticks2 = ax0_2.yaxis.get_major_ticks()
        yticks2[-1].label2.set_visible(False)
        ax0.text(0.92, 0.2, 'a', transform=ax0.transAxes, va='top')
        # special setting
        # x_lines = [80, 87, 104, 112]
        # for x00 in x_lines:
        #     ax0.axvline(x=x00)

        if i < 2:
            xticks = ax0.axes.get_xticklabels()
            for xt in xticks:
                xt.set_visible(False)
            ax_sn.set_ylabel("Snow depth (cm)")
            # ax_sn.get_yaxis().set_label_coords(0, 0)
        if i<1:
            grey_patch = test_def.make_patch("grey")
            # leg0 = ax0.legend([l2[0], l2[1], l3], ['NPR', '$\sigma_{45}^0$', 'Snow depth cm'], bbox_to_anchor=(0., 1.02, 1., 1.02), ncol=3, loc=3, prop={'size': 10}, frameon=False)
            # leg0.get_frame().set_linewidth(0.0)

    # for ax_labelloc in [ax0, ax_vh, ax_3rd]:
    #     ax_labelloc.get_yaxis().set_label_coords(-0.2, 0.5)
    cax = fig0.add_axes([0.12, 0.8, 0.5, 0.05])
    cb2 = colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, ticks=[-10, -5, 0, 5, 10, 15], orientation='horizontal')
    cb2.set_label('Air temperature ($^\circ$C)', labelpad=-50)
    # cb2.ax.axis.set_label_position('top')

    axs[-1].set_xlabel('Day of year 2016')
    plt.tight_layout()
    fig0.subplots_adjust(hspace=0.2)
    plt.savefig('result_08_01/test03.png', dpi=300)


def result_scatter():
    prefix = './result_07_01/txtfiles/result_txt/'
    site_onset = np.loadtxt(prefix+'site_onsets.csv', delimiter=',')
    tb_onset = np.loadtxt(prefix+'tb_onsets_1st.csv', delimiter=',')
    npr_onset2nd = np.loadtxt(prefix+'npr_onset_2nd.csv', delimiter=',')
    ascat_onset2nd = np.loadtxt(prefix+'ascat_onset_2nd.csv', delimiter=',')
    scatter_labeled(site_onset[:, 2], tb_onset[:, 1], tb_onset[:, -1].astype(int), fname=prefix+'tb_thaw_end_x_temp_out') # tb_end x t_thaw_out
    scatter_labeled(site_onset[:, 4], tb_onset[:, 2], tb_onset[:, -1].astype(int),
                    xylim=[250, 365], fname=prefix+'tb_freeze_st_x_smfreeze_st')
    fname_list = ['npr_thaw_st_x_ascat_thaw_st', 'npr_thaw_end_x_ascat_thaw_end', 'npr_freeze_st_x_ascat_freeze_st', 'npr_freeze_end_x_ascat_freeze_end']
    lims = [[80, 150], [80, 150], [250, 365], [250, 365]]
    for i0, txt0 in enumerate(fname_list):
        scatter_labeled(npr_onset2nd[:, i0], ascat_onset2nd[:, i0], ascat_onset2nd[:, -1].astype(int),
                        xylim=lims[i0], fname=prefix+txt0)
    scatter_labeled(site_onset[:, 1], npr_onset2nd[:, 1], npr_onset2nd[:, -1].astype(int),fname=prefix+'npr_thaw_end_x_sm_thaw')
    scatter_labeled(site_onset[:, 4], npr_onset2nd[:, 3], npr_onset2nd[:, -1].astype(int),
                    xylim=[250, 365], fname=prefix+'npr_freeze_end_x_sm_freeze')
    return 0


def scatter_labeled(x, y, label, xylim=[80, 150], fname='test01'):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(label):
        ax.annotate(txt, (x[i], y[i]))
    lx = np.arange(0, 410, 10)
    ly = np.arange(0, 410, 10)
    ax.plot(lx, ly, 'k--')
    ax.set_xlim([xylim[0], xylim[1]])
    ax.set_ylim([xylim[0], xylim[1]])
    plt.savefig(fname)
    return 0


def smap_ft_result(site_no, orb=1):
    prefix1 = '/media/327A50047A4FC379/SMAP/n5eil01u.ecs.nsidc.org/SMAP/SPL3FTP.001'
    txtname = './result_07_01/txtfiles/site_tb/multipixels/tb_multipixels_%s.txt' % site_no
    pixel_center = np.loadtxt(txtname, delimiter=',')
    uplayer = 'Freeze_Thaw_Retrieval_Data/'
    h5_test_file = 'SMAP_L3_SM_P_20150428_R13080_001.h5'
    # att_list0 = data_process.get_h5_atts(h5_test_file, uplayer)

    h5_atts = ['freeze_thaw', 'freeze_thaw_time_seconds', 'freeze_reference', 'freeze_thaw_uncertainty',
               'latitude', 'longitude', 'normalized_polarization_ratio',
               'reference_image_threshold', 'thaw_reference']
    att_list = [uplayer+str0 for str0 in h5_atts]
    # h5 files
    periods1 = np.arange(1, 366)
    result_npy = np.zeros([periods1.size, pixel_center.shape[0], len(h5_atts)+1])-1
    date0 = datetime(2016, 1, 1)
    date_str = []
    for doy in periods1:
        date_str.append(date0.strftime("%Y.%m.%d"))
        date0+=timedelta(1)
    date_list_local = os.listdir('/media/327A50047A4FC379/SMAP/n5eil01u.ecs.nsidc.org/SMAP/SPL3FTP.001')
    for i, d0 in enumerate(date_str):
        if d0 in date_list_local:
            path = '/media/327A50047A4FC379/SMAP/n5eil01u.ecs.nsidc.org/SMAP/SPL3FTP.001/%s/*.h5' % d0
            h5path = glob.glob(path)
            print path, '\n', h5path[0]
            a0, id0, status0 = data_process.read_h5_latlon(h5path[0], pixel_center,
                                                           att_list,
                                                           orb=orb)
            result_npy[i, :, :] = a0
        else:
            print 'no data measured on %s' % d0
    fname = './result_07_01/new_final/smap_compare/SMAP_FT_%s_%s' % (site_no, str(orb))
    np.save(fname, result_npy)
    fname_hearder = './result_07_01/new_final/smap_compare/smap_FT.meta'
    np.savetxt(fname_hearder, h5_atts, delimiter=',', fmt='%s')
    return 0


def smap_ft_compare(site_no, period=['thaw', 50, 150], orb=1, subplot_id=1):  # orb 1: PM (Asc pass)
    # initial
    prefix0 = './result_07_01/new_final/smap_compare/'
    ft10_dict_value = {}
    ft10_dict_time = {}
    ft10_dict_fr_ref, ft10_dict_th_ref, ft10_dict_npr = {}, {}, {}

    fname0 = prefix0+'SMAP_FT_%s_0.npy' % site_no
    fname1 = prefix0+'SMAP_FT_%s_1.npy' % site_no
    if orb == 1:
        orb_plot_all = '_A_'
        ft10 = np.load(fname1)  # ASC pm pass
    else:
        orb_plot_all = '_D_'
        ft10 = np.load(fname0)  # DES am pass
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

    if period[0] == 'all':
        all_compare = True
    else:
        all_compare = False
    # onset_smap_ft, dist = test_def.plot_ft_compare(pixel_id, ft10, ft10_dict_value, period,
    #                                                sno=site_no, compare=all_compare,
    #                                                orb_plot=orb_plot_all, sec_dict=ft10_dict_time,
    #                                                npr_ref=[ft10_dict_fr_ref, ft10_dict_th_ref, ft10_dict_npr],
    #                                                ldcover=False, subplot_a=subplot_id)
    onset_smap_ft, status = test_def.npr_smap_compare(pixel_id, ft10, ft10_dict_value, period,
                                                   sno=site_no, compare=all_compare,
                                                   orb_plot=orb_plot_all, sec_dict=ft10_dict_time,
                                                   npr_ref=[ft10_dict_fr_ref, ft10_dict_th_ref, ft10_dict_npr],
                                                   ldcover=False, subplot_a=subplot_id)  # compare 947 and 968 NPR and SMAP FT product
    print 'the plotted rows are: ', subplot_id

    return onset_smap_ft, status


def ft_product(site_nos, orb0=0):  # 0 is am pass
    orb_list = ['D', 'A']
    onset_tp = np.zeros([len(site_nos), 5])
    for i, site_no in enumerate(site_nos):
        smap_ft_name = './result_07_01/new_final/smap_compare/SMAP_FT_%s_0.npy' % site_no
        print site_no
        if os.path.exists(smap_ft_name) is False:
            smap_ft_result(site_no, orb=0)
            smap_ft_result(site_no, orb=1)
        smap_ft_compare(site_no, orb=orb0)
        smap_ft_compare(site_no, period=['freeze', 235, 350], orb=orb0)
        onset_01, dist = smap_ft_compare(site_no, period=['all', 0, 365], orb=orb0)
        onset_tp[i, 0:-1] = onset_01
        onset_tp[i, -1] = dist
    if orb0 == 0:
        onset_file = './result_07_01/all_sonet_D_7.csv'
    else:
        onset_file = './result_07_01/all_sonet_A_7.csv'
    onset_00 = np.loadtxt(onset_file, delimiter=',')
    onset_saving = np.zeros([onset_00.shape[0], onset_00.shape[1]+5])
    print onset_tp.shape
    onset_saving[:, [-5, -4, -3, -2, -1]] = onset_tp
    onset_saving[:, 0: -5] = onset_00
    savename = 'smap_ft_compare_%s.csv' % orb_list[orb0]
    np.savetxt(savename, onset_saving, delimiter=',', fmt='%d',
               header='ascatt, ascatf, tbt, tbf, nprt, nprf, stationt, stationf, tsoilt, tsoilf, stationid, smap_t0, smap_t1, smap_f0, smapf1')
    # disscus_sm_variation()
    # discuss_combining()
    # new_process()
    # test_method('thaw', txt=True)
    # test_method('freeze', txt=True)


def orbit_compare(figure_type):
    obs2 = ['_D_', '_A_']
    onset2 = []  # onse2[0] for descending results
    for obs in obs2:
        csvname = './result_07_01/all_sonet%s7.csv' % obs
        onset = np.loadtxt(csvname, delimiter=',')
        onset[onset[:, 10]==1089, :] = np.nan
        onset2.append(onset)
    # start plotting
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    data_symbol = ['ro', 'go', 'bo']
    l_list = []
    fig0 = plt.figure()  # thaw
    ons_thaw = []
    if figure_type == 'thaw':
        colnum = [2, 4, 6]
    elif figure_type == 'freeze':
        colnum = [1, 3, 5]
    [ons_thaw.append([onset2[0][:, n], onset2[1][:, n]]) for n in colnum]
    ax0 = fig0.add_subplot(1, 1, 1)
    for i, sym in enumerate(data_symbol):
        x0, y0 = ons_thaw[i][0], ons_thaw[i][1]
        xerr, yerr = (x0-y0)/2, (y0-x0)/2
        # l0 = ax0.plot(ons_thaw[i][0], ons_thaw[i][1], data_symbol[i])  # x: Des, y: Asc
        l0 = ax0.errorbar(x0, y0, yerr=xerr, fmt='o')
        l_list.append(l0)
    # figure layout setting
    ax0.plot(np.arange(1, 400), np.arange(1, 400), 'k-')
    ax0.legend(l_list, ['ASCAT', '$T_{BV}$', 'NPR'], loc=0, numpoints=1)
    ax0.set_ylabel('Ascending onsets')
    ax0.set_xlabel('Desending onsets')
    if figure_type == 'thaw':
        ax0.set_xlim([50, 150])
        ax0.set_ylim([50, 150])
    elif figure_type == 'freeze':
        ax0.set_xlim([250, 350])
        ax0.set_ylim([250, 350])
    figname = './result_07_01/temp_result/orbit_compare_%s' % figure_type
    plt.savefig(figname)
    return 0


def amsr2_plot(site_no, orb='A', pol='V'):
    # plot Tsoil, T6.9, T18.7, T36.5
    txtname = 'amsr2_series_%s_%s.txt' % (site_no, orb)
    save_value = np.loadtxt(txtname)
    with open(txtname, 'rb') as as0:
        reader = csv.reader(as0)
        for row in reader:
            if '#' in row[0]:
                header = ''.join(row)
                head_list = header.split(';')
                if pol=='V':
                    # for v-pol
                    i6, i18, i36 = 3, 4, 5
                    label0 = row[1]+' '+row[2][0]
                    label1 = row[3]+' '+row[4][0]
                    label2 = row[5]+' '+row[6][0]
                    label4 = '2.4GHz V'
                # for h-pol
                else:
                    i6, i18, i36 = 7, 8, 9
                    label0 = row[7]+' '+row[8][0]
                    label1 = row[9]+' '+row[10][0]
                    label2 = row[11]+' '+row[12][0]
                    label4 = '2.4GHz H'
    # plotting the time series
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(2, 1, 1)
    xy0 = save_value[save_value[:, -1]>0, :]
    # remove odd value
    non_odd_row3, non_odd_row4, non_odd_row5 = \
        (xy0[:, i6]>1e4) & (xy0[:, i6]<3e4), (xy0[:, i18]>1e4) & (xy0[:, i18]<3e4), (xy0[:, i36]>1e4) & (xy0[:, i36]<3e4)
    non_odd_row = non_odd_row3 & non_odd_row4 & non_odd_row5
    xy = xy0[non_odd_row, :]
    doy_measured = np.arange(366, 366+365)
    t_daily, t_doy = read_site.read_measurements(site_no, "Soil Temperature Observed -2in (degC)", doy_measured)
    swe_daily, swe_doy = read_site.read_measurements(site_no, "snow", doy_measured)
    sm_daily, sm_doy = read_site.read_measurements(site_no, "Soil Moisture Percent -2in (pct)", doy_measured)
    t_doy-=365
    swe_doy-=365
    sm_doy-=365
    l0, = ax0.plot(xy[:, 0], xy[:, i6]*0.01, label=label0)
    l1, = ax0.plot(xy[:, 0], xy[:, i18]*0.01, label=label1)
    l2, = ax0.plot(xy[:, 0], xy[:, i36]*0.01, label=label2)
    l3, = ax0.plot(t_doy, t_daily+273.1, label='-5 cm')
    # add L-band SMAP
    obs = '_%s_' % (orb)
    tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                         seriestype='tb', tbob=obs, sig0=7, order=1)
    if pol == 'H':
        l4, = ax0.plot(tbh0[0], tbh0[1], label=label4)
    else:
        l4, = ax0.plot(tbv0[0], tbv0[1], label=label4)
    plt.legend(handles=[l0, l1, l2, l3, l4], loc=0, prop={'size': 10})
    ax1 = fig0.add_subplot(2, 1, 2)
    l4_1, = ax1.plot(sm_doy, sm_daily, 'k--', label='vwc')
    ax1_2 = ax1.twinx()
    l4_2, = ax1_2.plot(swe_doy[swe_daily>0], swe_daily[swe_daily>0], 'k-', label='swe/sd')
    plt.legend(handles=[l4_1, l4_2])
    for ax in [ax0, ax1]:
        ax.set_xlim([0, 365])
    # add the onset line
    v_file ='./result_07_01/txtfiles/result_txt/smap_ft_compare_%s.csv' % orb  # vline from txt
    onset_array = np.loadtxt(v_file, delimiter=',')
    onset_row = onset_array[onset_array[:, 10] == int(site_no), :][0]
    ths, frs = \
        [onset_row[4], onset_row[6], onset_row[11]], [onset_row[5], onset_row[7], onset_row[13]]  # npr, site, smap
    ax0.axvline(x=ths[0], color='k', ls='--')
    ax0.axvline(x=frs[0], color='k', ls='--')
    ax0.axhline(y=273.15, color='k', ls=':')
    fname = 'amsr2_series_%s_%s_%s' % (site_no, orb, pol)
    plt.savefig(fname)
    plt.close()
    # scatter the 18.7 and 36.5, different color for before or after the thawing onset based on v_file
    fig1 = plt.figure()
    ax2 = fig1.add_subplot(1, 1, 1)
    frozens_ind = (xy[:, 0]<ths[0]) | (xy[:, 0]>frs[0])
    ax2.scatter(xy[frozens_ind, 4]*0.01, xy[frozens_ind, 5]*0.01, facecolor='none', edgecolor='b', label='frozen')
    ax2.scatter(xy[~frozens_ind, 4]*0.01, xy[~frozens_ind, 5]*0.01, facecolor='none', edgecolor='g', label='unfrozen')
    fname = 'amsr2_scatter_%s_%s' % (site_no, orb)
    ax2.set_xlim([180, 300]), ax2.set_xlabel('18.7 GHz V')
    ax2.set_ylim([180, 300]), ax2.set_ylabel('36.5 GHz V')
    x, y = np.arange(180, 300), np.arange(180, 300)
    ax2.plot(x, y, 'k-')
    ax2.plot(x, y-5, 'k--')
    ax2.plot(x, y+5, 'k--')
    plt.legend(loc=2)
    plt.savefig(fname)
    plt.close()
    return 0


def ft_product_check(siteno, atts=[], orb='1'):
    if type(orb) is not str:
        raise ValueError("The orbit code should be string type")
    fname = './result_07_01/new_final/smap_compare/SMAP_FT_%s_%s.npy' % (siteno, orb)
    x00 = np.load('./result_07_01/new_final/smap_compare/SMAP_FT_947_1.npy')
    shp_x00 = x00.shape
    print 'site %s \'s data has shape: ' % (siteno), x00.shape
    with open('smap_FT.meta') as meta0:
        content = meta0.readlines()
        metas = [x.strip() for x in content]
    save_siteno = np.zeros([len(atts), shp_x00[1]])
    for i, attr in enumerate(atts):
        if attr in metas:
            idx = metas.index(attr)
            print 'the %s of %s is: ' % (attr, siteno), x00[10, :, idx]
            save_siteno[i] = x00[10, :, idx]
    txt_save_name = 'smap_pixel_%s.txt' % siteno
    np.savetxt(txt_save_name, save_siteno.T, fmt='%.5f', delimiter=',', header='latitude, longitude')


def plot_snow_depth(orb='A'):
    '''
    Draw two examples to show the snow information when thawing is detected
    :return:
    '''
    # read the onset infomation
    onset_file = 'result_07_01/txtfiles/result_txt/smap_ft_compare_%s.csv' % orb
    onset_fromfile = np.loadtxt(onset_file, delimiter=',')
    # read snow data for stations
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175',
                '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    start_date = 366
    doy_measured = np.arange(start_date, start_date+365)
    snow_date, snow_value = [], []
    snow_date10, snow_change10 = [], []
    onset_list = []
    # onset 10 day change
    snow_melt10 = np.zeros([2, len(site_nos)])-1
    tair_thawing = np.zeros([2, len(site_nos)])-99
    # ascat_melt10 = np.zeros(len(site_nos))-1
    for i2, site_no in enumerate(site_nos):
        obv, obh, m1, m1_change = data_process.plot_obd(site_no, p='vh', isplot=False)
        air_name = "Air Temperature Observed (degC)"
        if site_no in ['2065', '2081']:
            air_name = "Air Temperature Average (degC)"
        tair, tair_d = read_site.read_measurements(site_no, air_name, np.arange(366, 366+365), hr=18)
        tair[tair<-90] = np.nan
        # extra checking
        if site_no == '960':
            fig1 = plt.figure(figsize=[8, 2])
            ax000 = fig1.add_subplot(111)
            ax000.plot(tair_d-365, tair, 'k:')
            ax000.set_xlim([0, 365])
            ax000.axvline(x=107)
            plt.savefig('plot_snow_depth_check_tair.png')

        tair_d-=365
        # [sm_date365, sm_des], [sm_date365[1:], sm_change]
        swe_daily, swe_doy = m1[1], m1[0]
        swe_daily[swe_daily < -90] = np.nan
        snow_date.append(swe_doy)
        snow_value.append(swe_daily)
        st_id = np.where(np.abs(m1_change[0] - 10)<0.5)[0]
        doy_interval = 5
        a_lims = range(st_id, m1_change[0].size, doy_interval)
        date_10 = m1_change[0][a_lims]
        change_10 = np.zeros(date_10.size)-99.0
        n=0
        for a in a_lims:
            change_10[n] = np.nansum(m1_change[1][a: a+doy_interval])
            n+=1
        snow_date10.append(date_10)
        snow_change10.append(change_10)
        onset_value = onset_fromfile[onset_fromfile[:, 10] == int(site_no), :]
        onset_list.append(onset_value)  # add to the onsets list, all onsets, 0: ascat, 4: npr
        timing2 = 60
        snow_reference = swe_daily[np.where((swe_doy - timing2<1)&(swe_doy - timing2>=0))]
        print snow_reference
        print 'snw at doy 60 is %.1f' % snow_reference
        for i3, timing0 in enumerate([onset_value[0][4], onset_value[0][0]]):  # i3: 0 npr, 1 ascat
            window_10 = np.where((swe_doy - timing0<10)&(swe_doy - timing0>0))[0]
            melt_10 = swe_daily[window_10[0]] - swe_daily[window_10[-1]]
            melt_10 = swe_daily[window_10[0]]
            snow_melt10[i3][i2] = melt_10
            air_timing0 = np.where((tair_d - timing0<=2)&(tair_d - timing0>=-2))[0]
            tair_temp = np.mean(tair[air_timing0])
            tair_thawing[i3][i2]=tair_temp

        # draw single plot for each station
        fig0 = plt.figure(figsize=[4, 3])
        ax00 = fig0.add_subplot(111)
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        ax00.plot(swe_doy, swe_daily, 'k-')
        ax01 = ax00.twinx()
        change_plot = change_10
        change_plot[(change_plot >-1) & (change_plot < 1)] = np.nan
        ax01.bar(date_10+0.5, change_plot, color='grey')
        ax00.axvline(x=onset_value[0][0], color='k', ls='--')  # ascat thawing
        ax00.axvline(x=onset_value[0][4], color='k', ls='-')  # npr thawing
        ax00.set_xlim([60, 150])
        ax00.set_ylim([0, 120])
        plt.savefig('plot_snow_depth_'+site_no+'.png')
    for i4, sno in enumerate(site_nos):
        print '10-melt at %s: NPR: DOY%.f, %.1f (%.2f degC); ASCAT: DOY%.f, %.1f(%.2f degC)' \
              % (sno, onset_list[i4][0][4], snow_melt10[0][i4], tair_thawing[0][i4], onset_list[i4][0][0], snow_melt10[1][i4], tair_thawing[1][i4])
    return 0
    # draw the plot
    fig, (ax0, ax1) = plt.subplots(2, sharex=True)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    for i0, ax in enumerate([ax0, ax1]):
        ax.plot(snow_date[i0], snow_value[i0], 'k-')
        ax22 = ax.twinx()
        dswe=[snow_date10[i0], snow_change10[i0]]
        dswe[1][(dswe[1] >-1) & (dswe[1] < 1)] = np.nan
        ax22.bar(dswe[0]+0.5, dswe[1], color='grey')
        ax22.set_ylim([-60, 60])
        ax22label = '10-day SWE change\n(mm)'
        # add vline based on NPR and ASCAT for thawing
        print onset_list[i0]
        ax.axvline(x=onset_list[i0][0][0], color='k', ls='--')
        ax.axvline(x=onset_list[i0][0][4], color='k', ls='-')
        ax.set_xlim([60, 150])
        ax.set_ylim([0, 120])
    plt.savefig('plot_snow_depth.png')
    return 0


def thawing_snow_depth(orb='A'):
    # read the onset infomation
    onset_file = 'result_07_01/txtfiles/result_txt/smap_ft_compare_%s.csv' % orb
    onset_fromfile = np.loadtxt(onset_file, delimiter=',')
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']  # all stations
    site_nos = ['947', '949', '950', '960', '967', '968', '1090', '1175', '1177', '2065', '2081', '2210', '2211', '2212', '2213']  # scan and sno
    # site_nos = ['960', '968', '1090', '1175', '1177', '2065', '2081', '2210', '2211', '2212', '2213']  # scan only
    scan_sites = ['960', '962', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    scan_sites = ['960', '968', '1090', '1175', '1177', '2065', '2081', '2210', '2211', '2212', '2213']
    swe_sites = ['947', '949', '950', '967']
    swe_thawing, i0 = np.zeros([2, len(swe_sites)]) - 1, 0  # 0: npr, 1: ascat
    sd_thawing, i1 = np.zeros([2, len(site_nos)]) - 1, 0
    for site_no in site_nos:
        swe_daily, swe_date = read_site.read_measurements(site_no, 'snow', np.arange(366, 366+365))
        onset_value = onset_fromfile[onset_fromfile[:, 10] == int(site_no), :]  # 04 npr thawing, 00 ascat thawing
        swe_npr = swe_daily[np.where(np.abs(swe_date-onset_value[0][4]-365) < 1)]
        swe_ascat = swe_daily[np.where(np.abs(swe_date-onset_value[0][0]-365) < 1)]
        if site_no in swe_sites:
            sd_thawing[0][i0] = swe_npr/2
            sd_thawing[1][i0] = swe_ascat/2
            i0 += 1
        else:
            sd_thawing[0][i0] = swe_npr
            sd_thawing[1][i0] = swe_ascat
            i0 += 1
    for i2, site_no in enumerate(site_nos):
        print '%s, NPR, %d, ASCAT, %d' % (site_no, sd_thawing[0][i2], sd_thawing[1][i2])
    print 'all NPR-ASCAT: %.1f $\pm$ %.1f' % (np.mean(sd_thawing[0]-sd_thawing[1]), np.std(sd_thawing[0]-sd_thawing[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # p1, = ax.plot(swe_thawing[0], swe_thawing[1], 'ro')
    p2, = ax.plot(sd_thawing[0], sd_thawing[1], 'bo')
    # leg1 = ax.legend([p1, p2], ['SWE', 'Snow Depth'],
    #            loc=0, ncol=1, prop={'size': 12}, numpoints=1)
    l2d_ascat = ax.axhline(y=np.mean(sd_thawing[1]), ls='--')
    l2d_npr = ax.axvline(x=np.mean(sd_thawing[0]), ls='--')
    ax.set_xlabel('NPR_snow')
    ax.set_ylabel('ASCAT_snow')
    ax.set_xlim([-10, 100])
    ax.set_ylim([-10, 100])
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    plt.savefig('thawing_snow_dpth.png')
    plt.close()

    return 0


def check_distance():
    site_no = '1090'
    npy_name = 'result_05_01/ascat_point/ascat_%s_2016.npy' % site_no
    ascat_series = np.load('result_05_01/ascat_point/ascat_s1090_2016.npy')
    s_info = site_infos.change_site(site_no)
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    dis = bxy.cal_dis(s_info[1], s_info[2], ascat_series[:, -3], ascat_series[:, -2])
    x0 = np.arange(1, dis.size+1)
    ax0.plot(x0, dis)
    plt.savefig('check_distance')
    print ascat_series[0]


def table_stations():
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081',
                    '2210', '2211', '2212', '2213']
    for site_no in site_nos:
        s_info = site_infos.change_site(site_no)
        site_name = site_infos.change_site(site_no, names=True)
        # with open('table_site_loc.csv', 'a') as csvfile:
        #     site_statistic = csv.writer(csvfile)
        #     site_statistic.writerow(['%s, v:, %.3f, %.3f, h: %.3f, %.3f' % (site_no, np.nanmean(np.abs(site_errorv)), np.nanmax(np.abs(site_errorv)),
        #                                                                 np.nanmean(np.abs(site_errorh)), np.nanmax(np.abs(site_errorh)))])
        print s_info, site_name
        with open('table_site_loc.txt', 'a') as t_file:
            t_file.write('%s & %s & %.2f & %.2f \n' % (s_info[0], site_name, s_info[1], s_info[2]))


def ascat_fp_ak(npyname, dtr='test'):
    npy_orig = np.load(npyname)
    indx0, indx1, indx2 = range(0, 14), range(14, 18), range(21, 27)
    savetxt0 = npy_orig[:, indx0+indx1+indx2+[-1]]
    savetxt1 = savetxt0[savetxt0[:, -1]<1]
    utctime0 = savetxt1[:, 14]
    utctime1 = bxy.time_getlocaltime(utctime0, ref_time=[2000, 1, 1, 0])  # 2000-01-01 00:00:00
    savetxt1[:, 14] = utctime1[4]
    heards = 'latitude, longitude, \
             sigma0_trip0, sigma0_trip1, sigma0_trip2, f_usable0, f_usable1, f_usable2, \
             inc_angle_trip0, inc_angle_trip1, inc_angle_trip2, \
             f_land0, f_land1, f_land2, \
             utc_line_nodes, abs_line_number, \
             sat_track_azi, swath_indicator, azi_angle_trip0, azi_angle_trip1, azi_angle_trip2,  num_val_trip0, \
             num_val_trip1, num_val_trip2'
    txtname = 'ascat_ft_%s.txt' % dtr
    np.savetxt(txtname, savetxt1, delimiter=',', fmt='%.5f', header=heards)


def check_ascat_sub(fname, fname2):
    """
    :param fname:
    :param fname2: corner coordinate
    :return:
    """
    value0 = np.load(fname)
    value_corner = np.load(fname2)
    value01 = value0[0, :, :]
    value_corner01 = value_corner[0, :, :]
    time_utc = value0[0, 1:10, 14]
    time_array = bxy.time_getlocaltime(time_utc, ref_time=[2000, 1, 1, 0])
    print value0.shape
    # print value1[1]
    print time_array
    print value0[0, 1:5, 45]
    print value0[0, 1:5, 0]
    print value0[0, 1:5, 1]
    print value0[0, 1:5, 2]
    hr0 = site_infos.ascat_heads('ascat0')
    value01 = value01[value01[:, -1] > -999]
    value_corner01 = value_corner01[value_corner01[:, -1]>-999]

    np.savetxt('ascat_20160409_0509_947_000.txt', value01, header=hr0, delimiter=',', fmt='%.6f')
    np.savetxt('ascat_20160409_0509_947_000_corner.txt', value_corner01, delimiter=',', fmt='%.6f')

    print all(value01[:, -1] == value_corner01[:, -1])


    # f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex='col', sharey='row')
    # axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    # i = -1
    # for axi in axs:
    #     # print ascat_record[i][0][2], ascat_record[i][1][2]
    #     i += 1
    #     # get values
    #     ascat_record0 = value0[i, :, :]
    #     ascat_record_as = ascat_record0[ascat_record0[:, -1]==0]  # ascending data
    #     time_list = bxy.time_getlocaltime(ascat_record_as[:, 14], ref_time=[2000, 1, 1, 0])
    #     ascat_sub_time = time_list[3]+time_list[4]/24.0  # time of the measurements
    #     ascat_value_m = ascat_record_as[:, 3]
    #     ascat_value_inc = ascat_record_as[:, 9]
    #     sig_mn = ascat_value_m - (ascat_value_inc-45)*-0.11
    #     # get plotted
    #     axi.plot(ascat_sub_time, sig_mn, 'k-^', markersize=3)
    #     # axi.axvline(x=ascat_onset[i][0], color='k', ls='-.')
    #     # axi.axvline(x=ascat_onset[i][1], color='k', ls='-.')
    #     # axi.set_xlim([0, 365])
    #     # ylimits = site_infos.ascat_site_lim(site_no)
    #     # axi.set_ylim([ylimits[0], ylimits[1]])
    #     axi.tick_params(axis='y', which='minor')
    #     axi.locator_params(axis='y', nbins=6)
    #     axi.locator_params(axis='x', nbins=4)
    #     axi.tick_params(axis='x', which='minor', bottom='off')
    #     axi.set_ylim([-14, -8])
    # png_name = 'test_sub_%s.png' % '1090'
    # plt.savefig(png_name, dpi=150)
    # plt.close()


def get_corner(fname):
    value = np.loadtxt(fname, delimiter=',')
    newloc = data_process.ascat_corner_rotate([value[0, 1], value[0, 0]], value[0, 16]-180)
    np.savetxt('test_get_corner.txt', newloc, fmt='%.5f', delimiter=',', header='longitude, latitude')


def gdal_clips(sno, ipt='snowf', shp_folder='./result_08_01/point/ascat/shp/ascat_shp'):
    # gdalwarp -cutline ascat_1126200.shp -crop_to_cutline ims2016127_1km_GIS_v1.3.tif ims126_3.ti
    shp_path = '%s/ascat_%s*.shp' % (shp_folder, sno)
    results_path = './tp'
    results_path = 'result_08_01/point/ascat/pixels/%s' % ipt
    if ipt == 'snowf':
        inputif_path = '/home/xiyu/Data/Snow/2016_1km'
    else:
        inputif_path = '/home/xiyu/Data/nlcd'
    shpname0 = '%s/ascat_%s*.shp' % (shp_path, sno)
    shpname_list = glob.glob(shp_path)
    for shpnamei in shpname_list:
        fname0 = shpnamei.split('/')[-1]
        doy0 = fname0.split('_')[-1][1:4]
        shpid = fname0.split('_')[-1][0:7]
        if ipt == 'snowf':
            ipt_tif = '%s/ims2016%s_1km_GIS_v1.3.tif' % (inputif_path, doy0)
            output_tif = '%s/ascat_snowf_%s_%s.tif' % (results_path, sno, shpid)
        else:
            ipt_tif = '%s/1km03_uncombine_tf.tif' % inputif_path
            output_tif = '%s/ascat_lc_%s_%s.tif' % (results_path, sno, shpid)
        print shpnamei, ipt_tif, output_tif
        bcomand = "gdalwarp -cutline %s -crop_to_cutline %s %s" % (shpnamei, ipt_tif, output_tif)
        os.system(bcomand)


def read_clips(fname_ipt, fname_opt, type='snowf'):
    # translate to ascii
    comand_2_asc = ["gdal_translate", "-of", "AAIGrid", fname_ipt, fname_opt]
    sb.call(comand_2_asc)
    # calculate the stastics
    value = np.loadtxt(fname_opt, skiprows=6)
    all_num = value[value!=0].size*1.0  # all valid elements
    if type == 'lc':
        if all_num == 0:
            return np.array([327, 327, 327, 327, 327, 327])
        p_a0 = value[value == 42].size/all_num
        p_a1 = value[(value == 41) | (value == 43)].size/all_num
        p_a2 = value[(value == 51) | (value == 52)].size/all_num
        p_a3 = value[(value > 70) & (value < 75)].size/all_num
        p_a4 = value[(value > 90)].size/all_num
        p_a5 = value[value == 11].size/all_num
        p_name = ['ever', 'decid', 'shrub', 'tundra', 'wet', 'water']
        return np.array([p_a0, p_a1, p_a2, p_a3, p_a4, p_a5])
    elif type == 'snowf':
        if all_num == 0:
            return np.array([327])
        p_a0 = value[value == 4].size/all_num
        return np.array([p_a0])
    # i_0 = np.where(pp>5)
    # label0s = {'11': 'water', '12': 'perennial snow', '31': 'Barren Land', '41': 'Forest', '51': 'Shrub', '71': 'Grass', '90': 'wetland'}
    # label1s = ['Water', 'Perennial Snow', 'Barren Land', 'Forest', 'Shrub',  'Grass', 'wetland']
    # label_color = {'Water': 'aqua', 'Perennial Snow': 'snow', 'Barren Land': 'brown', 'Forest': 'forestgreen',
    #                'Shrub': 'olive',  'Grass': 'palegreen', 'wetland': 'steelblue'}


def ascat_snow_lc(sno, type=['snowf', 'lc']):
    # fomat: ascat_snowf_947_1126200.tif, ascat_lc_947_1126200.tif
    path0 = 'result_08_01/point/ascat/pixels'
    for type0 in type:  # snow and land cover
        ipt_tif_match = '%s/%s/ascat_%s_%s*.tif' % (path0, type0, type0, sno)
        save_array = np.zeros([251, 8])
        i0 = 0
        for ipt0 in sorted(glob.glob(ipt_tif_match)):
            tif_id = ipt0.split('/')[-1].split('.')[0].split('_')[-1][0: 7]
            if i0 == 0:
                doy0 = int(tif_id[1:4])
            opt_asc = '%s/ascii/ascat_%s_%s_%s.asc' % (path0, type0, sno, tif_id)
            sn_fracs = read_clips(ipt0, opt_asc, type=type0)  # ['ever', 'decid', 'shrub', 'tundra', 'wet', 'water']
            idx0 = sn_fracs.size
            save_array[i0, 0], save_array[i0, 1:idx0+1] = int(tif_id), sn_fracs
            i0+=1
            if i0 == 82:
                pause = 0
        print i0, ', the value of i0'
        print ipt_tif_match
        doy1 = int(tif_id[1:4])
        d0_str, d1_str = bxy.doy2date(doy0), bxy.doy2date(doy1)
        fname = '%s/npys/ascat_%s_%s_%s_%s.npy' % (path0, d0_str, d1_str, sno, type0)
        np.save(fname, save_array)
    return 0


def check_lc_snow_ascat(sno):
    # R3/result_08_01/point/ascat/pixels/npys
    if sno == '9004':
        pause = 0
    path0 = 'result_08_01/point/ascat/pixels/npys'
    fname_snow0 = '%s/ascat_*_%s_snowf.npy' % (path0, sno)
    fname_snow = glob.glob(fname_snow0)[0]
    test_lc = np.load(fname_snow)
    test_lc = test_lc[test_lc[:, 0]>0]

    fname_lc0 = '%s/ascat_*_%s_lc.npy' % (path0, sno)
    fname_lc = glob.glob(fname_lc0)[0]
    fname_no_path = fname_lc.split('/')[-1]
    time_zone = '_'.join(fname_no_path.split('_')[1: 3])
    test_lc1 = np.load(fname_lc)
    test_lc1 = test_lc1[test_lc1[:, 0]>0]
    test_lc1[:, 7] = 1  # the day before downloaded snow data, we set the initial snow-cover was 100%
    idx_melt_period = np.where(np.in1d(test_lc1[:, 0], test_lc[:, 0]))
    test_lc1[idx_melt_period, 7] = test_lc[:, 1]
    test_lc1[:, 1:]*=1000
    savename = '%s/ascat_%s_%s_snowlc.txt' % (path0, time_zone, sno)
    doy = test_lc[:, 0]/1e3-1e3
    fig0 = plt.figure()
    ax = fig0.add_subplot(1, 1, 1)
    ax.plot(doy.astype(int), test_lc[:, 1], 'o')
    plt.savefig('check_lc_snow_ascat')
    np.savetxt(savename, test_lc1, delimiter=',', fmt='%d')


def ascat_snowlc_npy(sno_sp):
    """
    for a given region, combine all ascat and snow_lc information
    :param sno_sp:
    :return:
    """
    value_path = 'result_08_01/point/ascat/time_series'
    # all statin
    for sno in sno_sp:
        print 'ascat_snowlc_npy: ', sno
        value_name0 = '%s/ascat_*_%s_value.npy' % (value_path, sno)
        value_name = glob.glob(value_name0)[-1]
        value0 = np.load(value_name)
        value1 = value0[0, :, :]
        value2 = value1[value1[:, -1] > -999]  # the ASCAT measure
        pixel_ids = value2[:, -1]
        snow_lc_name0 = 'result_08_01/point/ascat/pixels/npys/ascat_*_%s_snowlc.txt' % (sno)
        snow_lc_name = glob.glob(snow_lc_name0)[0]
        snow_lc = np.loadtxt(snow_lc_name, delimiter=',')  # the ASCAT ancillary
        mask0 = np.in1d(value2[:, -1], snow_lc[:, 0])
        i0 = 0
        measure_anc_array = np.zeros([280, 60]) - 999
        for id0 in snow_lc[:, 0]:
            pixel_value = value2[pixel_ids==id0]  # find the same pixel
            if pixel_value.shape[0]>1:
                pixel_value = pixel_value[0]
                # test temporally
                t0 = value2[3, 14]
                t1 = value2[4, 14]
                lat01, lon01 = value2[3:5, 0], value2[3:5, 1]
                sno_inf0 = site_infos.change_site(sno)
                dis = bxy.cal_dis(lat01, lon01, sno_inf0[1], sno_inf0[2])
                t_str0 = bxy.time_getlocaltime([t0, t1], ref_time=[2000, 1, 1, 0])
            elif pixel_value.size == 0:
                print 'data invalid: ', id0
                continue
            anc_value = snow_lc[i0]
            measure_anc_array[i0, 0: pixel_value.size] = pixel_value
            measure_anc_array[i0, pixel_value.size: pixel_value.size+anc_value.size] = anc_value
            i0+=1
        savename = 'result_08_01/point/ascat/pixels/results/ascat_0228_0528_%s.npy' % (sno)
        np.save(savename, measure_anc_array)
        pause = 0


def regions_extract(rid):
    # prior
    sno_sp_all = [['947', '948', '949', '950', '960', '1090'], ['962', '958', '2212'],
                  ['968', '1177', '1175'], ['2080', '963', '2081'],
                  ['9001', '9002', '9003', '9004', '9005', '9006', '9007'],
                  ['968', '962', '1175', '2065', '2210', '2211', '2212', '2213']]  # region 1 to 5
    sno_sp = sno_sp_all[rid-1]
    txtname = 'region_%d_all_data.txt' % rid
    # loops
    tair_all = []
    in_situ_all = []
    for sno in sno_sp:
        # initial of the previous result
        measurename = 'result_08_01/point/ascat/pixels/results/ascat_0228_0528_%s.npy' % (sno)  #
        measure_value = np.load(measurename)
        measure_value = measure_value[measure_value[:, 0] > -999]
        valid_index = [measure_value[:, i00] < 2 for i00 in [5, 6, 7]]  # true/false index for usable back scatter
        valid_index_all = valid_index[0] & valid_index[1] & valid_index[2]
        print 'unusable pixel was %d (fore), %d (mid), %d (aft)' \
              % (sum(~valid_index[0]), sum(~valid_index[1]), sum(~valid_index[2]))
        time_list = measure_value[:, 46][valid_index_all]
        # corrected back_scatter: triple_lets
        sigma_orig, incidence, other = measure_value[:, 2:5][valid_index_all], measure_value[:, 8:11][valid_index_all], \
                                       measure_value[:, 47:55][valid_index_all]
        sigma45 = data_process.angular_correct(sigma_orig, incidence)
        var0 = np.modf(time_list/1e3)
        hrs = (var0[0]*1e2).astype(int)
        doy = var0[1]-1e3
        tair_value, tair_date = read_site.read_measurements(sno, "Air Temperature Observed (degC)", 366+doy, hr=hrs)
        tairs = np.zeros([tair_value.size, 15])
        tairs[:, 0] = tair_date
        tairs[:, 1] = tair_value
        tairs[:, 2:5] = sigma45
        tairs[:, 5] = int(sno)
        tairs[:, 6:14] = other
        tairs[:, 14] = time_list
        tair_all.append(tairs)
        # other in situ data
        swe_v, swe_date = read_site.read_measurements(sno, "snow", 366+doy, hr=0)
        swe_v_1, swe_date_1 = read_site.read_measurements(sno, "snow", 366+doy-5, hr=0)
        d_swe =0.2*(swe_v - swe_v_1)
        if sno in ['947', '949', '950', '967', '1089']:
            swe_v = swe_v/2.5
        vwc_v, vwc_date = read_site.read_measurements(sno, "Soil Moisture Percent -2in (pct)", 366+doy, hr=hrs)
        tsoil_v, tsoil_date = read_site.read_measurements(sno, "Soil Temperature Observed -2in (degC)", 366+doy, hr=hrs)
        pp_v, pp_date = read_site.read_measurements(sno, "Precipitation Increment (mm)", 366+doy, hr=23)
        in_situ_tp = [swe_v, vwc_v, tsoil_v, pp_v, d_swe]
        in_situ = np.zeros([tair_value.size, 7])
        for i in [0, 1, 2, 3, 4]:
            in_situ[:, i] = in_situ_tp[i]
        in_situ[:, i] = in_situ_tp[i]
        in_situ[:, 6] = time_list
        in_situ[:, 5] = int(sno)
        in_situ_all.append(in_situ)

    sigma_air0 = tair_all[0]
    for airi in tair_all[1:]:
        sigma_air0 = np.vstack((sigma_air0, airi))
    savename = 'result_08_01/point/ascat/pixels/results/air_sigma/region_sigma_air.txt'
    np.savetxt(savename, sigma_air0, delimiter=',',
               header='date,tair,fore,mid,aft,siteno,ID,ever,decid,shrub,tundra,wet,water,snowf,siteNo,ID,end',
               fmt='%.2f, %.2f, %.2f, %.2f, %.2f, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d')

    # save in situ data
    in_situ_data0 = in_situ_all[0]
    for insi in in_situ_all[1:]:
        in_situ_data0  = np.vstack((in_situ_data0, insi))
    savename = 'result_08_01/point/ascat/pixels/results/air_sigma/region_other_in_situ.txt'
    np.savetxt(savename, in_situ_data0, delimiter=',',
               header='swe,vwc,t_soil,pp,dswe,site_no,ID,end',
               fmt='%.2f, %.2f, %.2f, %.2f, %.2f, %d, %d')

    # save all the meausrements
    save_all_value = np.hstack((sigma_air0, in_situ_data0[:, 0:5]))
    savename = 'result_08_01/point/ascat/pixels/results/air_sigma/%s' % (txtname)
    np.savetxt(savename, save_all_value, delimiter=',',
             header='date,tair,fore,mid,aft,siteno,ID,ever,decid,shrub,tundra,wet,water,snowf,ID,swe,vwc,t_soil,pp,dswe,end',
              fmt='%.2f, %.2f, %.2f, %.2f, %.2f, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.2f, %.2f, %.2f, %.2f, %.2f')
    # %.2f, %.2f, %.2f, %.2f, %d, %d
    return sno_sp


def regions_plotting(region_id=1, site_no=0, att_xyz=['tair', 'mid', '# date'], exception=False, xlim0=[-30, 30], xv=95):
    fname = 'result_08_01/point/ascat/pixels/results/air_sigma/region_%d_all_data.txt' % region_id
    with open(fname) as file0:
        for row0 in file0:
            atts = row0.split(',')
            print 'there are %d columns save in the txt' % len(atts)
            xyz_indx = [atts.index(att0) for att0 in att_xyz]
            print xyz_indx
            break
    region_value = np.loadtxt(fname, delimiter=',')
    # based on site location or not
    site_id = region_value[:, 5]
    site_idx = site_id == int(site_no)
    if sum(site_idx) == 0:
        site_idx[:] = True
    if exception is not False:
        for ex in exception:
            excep_idx = site_id == int(ex)
            site_idx = site_idx & ~excep_idx
    # input x, y, z, restricted by station region
    x_value = region_value[:, xyz_indx[0]][site_idx]
    sigma_mid = region_value[:, xyz_indx[1]][site_idx]
    z_value = region_value[:, xyz_indx[2]][site_idx]
    z_value[z_value == -99] = np.nan
    # specialized z value
    if att_xyz[2] == '# date':
        z_value -= 365
    elif att_xyz[2] == 'ID':
        var0 = np.modf(z_value/1e3)
        hrs = (var0[0]*1e2).astype(int)
        doy = var0[1]-1e3
        z_value = hrs
        att_xyz[2] = 'pass (hour)'
    elif att_xyz[2] == 'dswe\n':
        z_value *= 1
    if att_xyz[0] == "# date":
        x_value -= 365
    att_xyz[2], att_xyz[0], att_xyz[1] = ' ', ' ', ' '
    # value_xyz, att_xyz = trip_read(region_id=1, site_no=0, att_xyz=['tair', 'mid', '# date'], exception=False)
    # tair, sigma_mid, z_value = value_xyz[0], value_xyz[1], value_xyz[2]
    # plotting
    fig = plt.figure(figsize=[6, 4.5])
    ax = fig.add_subplot(1, 1, 1)
    m1 = ax.scatter(x_value, sigma_mid, c=z_value, cmap=plt.get_cmap('rainbow'))
    ax.plot(x_value, sigma_mid, 'k-')
    ax.set_xlim(xlim0)
    ax.set_xlabel(att_xyz[0]), ax.set_ylabel(att_xyz[1])
    ax.set_ylim([-18, -7.5])
    cax=plt.axes([.85, .1, .075, .8])
    plt.colorbar(m1, cax=cax)
    ax.text(0.92, 1.05, att_xyz[2], transform=ax.transAxes, va='top', fontsize=16)
    # draw some auxiliary line
    if region_id == 1:
        ax.axhline(y=-12.5, ls='--')
        ax.axhline(y=-11, ls='-.')
    ax.axvline(x=-5), ax.axvline(x=5)
    #plt.tight_layout()
    if rid == 5:
        ax.set_ylim([-20, -8])
    if int(site_no) == 9007:
        ax.set_ylim([-10, -5])
    ax.axvline(x=xv)
    figname = 'tp/region_%d_plotting_%d_%s_%s_%s' % \
              (region_id, int(site_no), att_xyz[0], att_xyz[1], att_xyz[2])
    plt.savefig(figname)
    plt.close()


def trip_read(region_id=1, site_no=0, att_xyz=['tair', 'mid', '# date'], exception=False):
    fname = 'result_08_01/point/ascat/pixels/results/air_sigma/region_%d_all_data.txt' % region_id
    with open(fname) as file0:
        for row0 in file0:
            atts = row0.split(',')
            print 'there are %d columns save in the txt' % len(atts)
            xyz_indx = [atts.index(att0) for att0 in att_xyz]
            print xyz_indx
            break
    region_value = np.loadtxt(fname, delimiter=',', skiprows=1)
    # based on site location or not
    site_id = region_value[:, 5]
    site_idx = site_id == int(site_no)
    if sum(site_idx) == 0:
        site_idx[:] = True
    if exception is not False:
        for ex in exception:
            excep_idx = site_id == int(ex)
            site_idx = site_idx & ~excep_idx
    # input x, y, z, restricted by station region
    x_value = region_value[:, xyz_indx[0]][site_idx]  # x_value
    y_value = region_value[:, xyz_indx[1]][site_idx]  # y_value
    z_value = region_value[:, xyz_indx[2]][site_idx]
    z_value[z_value == -99] = np.nan

    # specialized z value
    if att_xyz[2] == 'date':
        z_value -= 365
    elif att_xyz[2] == 'ID':
        var0 = np.modf(z_value/1e3)
        hrs = (var0[0]*1e2).astype(int)
        doy = var0[1]-1e3
        z_value = hrs
        att_xyz[2] = 'pass (hour)'
    return [x_value, y_value, z_value], att_xyz
    # plotting


def check_pass_time(region_id=1, site_no=0, exception=False, att_xyz=['ID']):
    fname = 'result_08_01/point/ascat/pixels/results/air_sigma/region_%d_all_data.txt' % region_id
    region_value = np.loadtxt(fname, delimiter=',', skiprows=1)
    with open(fname) as file0:
        for row0 in file0:
            atts = row0.split(',')
            print 'there are %d columns save in the txt' % len(atts)
            xyz_indx = [atts.index(att0) for att0 in att_xyz]
            print xyz_indx
            break
    # based on site location or not
    site_id = region_value[:, 5]
    site_idx = site_id == int(site_no)
    if sum(site_idx) == 0:
        site_idx[:] = True
    if exception is not False:
        for ex in exception:
            excep_idx = site_id == int(ex)
            site_idx = site_idx & ~excep_idx
    ID0 = region_value[:, xyz_indx[0]][site_idx]

    # plotting
    fig = plt.figure(figsize=[6, 4.5])
    var0 = np.modf(ID0/1e3)
    hrs = (var0[0]*1e2).astype(int)
    doy = var0[1]-1e3
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(doy, hrs, 'o')
    figname = 'tp/region_%d_overpass_hr.png' % region_id
    plt.savefig(figname)


def land_cover(rid, snos, trip_types, exce):
    mean_all, std_all, all_stations_lc = [], [], []
    for sno in snos:
        value_xyz, att_xyz = trip_read(region_id=rid, site_no=sno, att_xyz=trip_types, exception=exce)
        value_xyz2, att_xyz2 = trip_read(region_id=rid, site_no=sno, att_xyz=['tundra', 'wet', 'water'], exception=exce)
        type_percent = {}
        for per, name in zip(value_xyz, att_xyz):
            per[per == 327000] = np.nan
            type_percent[name] = [np.nanmean(per), np.nanstd(per)]
        for per, name in zip(value_xyz2, att_xyz2):
            per[per == 327000] = np.nan
            type_percent[name] = [np.nanmean(per), np.nanstd(per)]
        mean0 = np.zeros(6)
        std0 = np.zeros(6)
        for i0, key0 in enumerate(type_percent.keys()):
            if type_percent[key0][0] < 50:
                for i1 in range(2):
                    type_percent[key0][i1] = 0
            mean0[i0] = type_percent[key0][0]
            std0[i0] = type_percent[key0][1]
        # each element of this list corresponds to the sno in a sno list (e.g., region1: [947, 948, ...])
        all_stations_lc.append(type_percent)
        if sno == '948':
            print type_percent
        mean_all.append(mean0), std_all.append(std0)
    type_name = type_percent.keys()
    pause = 0
    mean_array = np.array(mean_all)
    std_array = np.array(std_all)
    type_mean = np.transpose(mean_array)
    type_std = np.transpose(std_array)
    N = len(snos)
    width = 0.15
    ind = np.arange(N)
    # plotting

    fig0 = plt.figure()
    if rid==5:
        fig0 = plt.figure(figsize=[8, 2])
    ax = fig0.add_subplot(1, 1, 1)
    ps = []  # save for adding legend
    i2 = 0
    label_color = {'water': 'aqua', 'wet': 'snow', 'shrub': 'brown', 'ever': 'forestgreen',
                                       'tundra': 'olive',  'decid': 'palegreen'}
    for typei in type_name:  # loop by types

        mean_stdi = np.array([type_dic[typei] for type_dic in all_stations_lc]).T  # loop by stations
        mean_stdi /= 10

        if typei == 'decid':
            pause=0
        if i2 == 0:
            tp_mean = mean_stdi[0]
            pi = ax.bar(ind, mean_stdi[0], width, yerr=mean_stdi[1], color=label_color[typei])
        else:
            pi = ax.bar(ind, mean_stdi[0], width, yerr=mean_stdi[1], bottom=tp_mean, color=label_color[typei])
            tp_mean += mean_stdi[0]
        ps.append(pi)
        i2 += 1

    # for mean1, std1 in zip(type_mean, type_std):
    #     if i2 == 0: # inx for types
    #         pi = ax.bar(ind, mean1/10.0, width, yerr=std1)
    #     else:
    #         pi = ax.bar(ind, mean1/10.0, width, yerr=std1, bottom=tp_mean)
    #     tp_mean = mean1
    #     ps.append(pi)
    #
    #     i2+=1
    tick_label0 = (tick0 for tick0 in snos)
    pic_tuple = (p0 for p0 in ps)
    type_tuple = (t0 for t0 in type_percent.keys())
    plt.xticks(ind, tick_label0)
    plt.legend(pic_tuple, type_tuple, loc=2, ncol=3, bbox_to_anchor=(0, 1.15))
    ax.set_ylim([0, 100])

    plt.savefig('tp/test_lc_percentage')
    plt.close()


def tb_melt_window(sno, file_path='result_08_01/point/smap_pixel/time_series', xv=0):
    orb = 'A'
    fname = '%s/tb_%s_%s_2016' % (file_path, sno, orb)
    # read head
    with open(fname) as file0:
        for row0 in file0:
            atts = row0.split(',')
            break
    tb_value0 = np.loadtxt(fname)
    print 'this data contain %d attributes (cols)' % len(atts)
    print tb_value0.shape
    # check distance
    lats_tb, lons_tb = tb_value0[:, 2], tb_value0[:, 1]
    coord_sno = site_infos.change_site(sno)
    dis_arr = bxy.cal_dis(coord_sno[1], coord_sno[2], lats_tb, lons_tb)
    if sno == '960' or sno == '1090':
        ref_dis = 20
    else:
        ref_dis = 10
    idx_dis = dis_arr<ref_dis
    tb_value0 = tb_value0[idx_dis, :]
    print 'ascending orbit mean', np.nanmean(dis_arr), 'with sigma of', np.nanstd(dis_arr)
    # read attributes
    doy = tb_value0[:, 0]
    tbv, tbh = tb_value0[:, atts.index('cell_tb_v_aft')], tb_value0[:, atts.index('cell_tb_h_aft')]  # npr
    npr_sno = (tbv-tbh)/(tbv+tbh)*1.0e2
    # tb diurnal
    orb2 = 'D'
    fname2 = '%s/tb_%s_%s_2016' % (file_path, sno, orb2)
    tb_value0 = np.loadtxt(fname2)
    lats_tb, lons_tb = tb_value0[:, 2], tb_value0[:, 1]
    dis_arr = bxy.cal_dis(coord_sno[1], coord_sno[2], lats_tb, lons_tb)
    idx_dis = dis_arr<ref_dis
    tb_value0 = tb_value0[idx_dis, :]
    print 'descending orbit mean', np.nanmean(dis_arr[idx_dis]), 'with sigma of', np.nanstd(dis_arr[idx_dis])
    doy_d = tb_value0[:, 0]
    tbv_d, tbh_d = tb_value0[:, atts.index('cell_tb_v_aft')], tb_value0[:, atts.index('cell_tb_h_aft')]  # npr
    doy_inter = np.intersect1d(doy, doy_d)
    idx0 = np.in1d(doy, doy_d)
    # idx0 = np.in1d(doy_inter, doy_d)  # days in ascending that also in descending
    idx1 = np.in1d(doy_d, doy)
    v_diurnal, h_diurnal = tbv_d[idx1] - tbv[idx0], tbh_d[idx1] - tbh[idx0]  # des - as, am - pm
    doy_diurnal = doy[idx0]
    fig, (ax1, ax2) = plt.subplots(2, figsize=[6, 4.5], sharex=True)
    t_win = (doy_diurnal>60) & (doy_diurnal<150)
    h_d_10 = np.nanmean(np.sort(h_diurnal[t_win])[-10: ])
    ax1.plot(doy_diurnal, v_diurnal, '-', doy_diurnal, h_diurnal, '--')
    ax1.axhline(y=h_d_10)
    ax1.set_ylim([-20, 30])
    ax2.plot(doy, npr_sno)
    ax2.axvline(x=xv)
    ax1.axvline(x=xv)
    figname = 'tp/region_5_%s' % sno
    plt.savefig(figname)
    plt.close()


def t_air_edges(site, orbit=0):
    sigma_g = 10
    sigma_ascat = 3
    ob_name = ['A', 'D']
    f_name = "./result_07_01/txtfiles/site_tb/tb_%s_%s_2016.txt" % (site, ob_name[orbit])
    att_list = ['cell_tb_time_seconds_aft', 'cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_qual_flag_v_aft',
                'cell_tb_qual_flag_h_aft', 'cell_tb_error_v_aft', 'cell_tb_error_h_aft', 'cell_lon', 'cell_lat']
    col_no = bxy.get_head_cols(f_name, headers=att_list)
    cell_tb = np.loadtxt(f_name)
    # assign the required value
    tb_v = cell_tb[:, col_no[att_list.index('cell_tb_v_aft')]]
    tb_h = cell_tb[:, col_no[att_list.index('cell_tb_h_aft')]]
    tb_sec = cell_tb[:, col_no[att_list.index('cell_tb_time_seconds_aft')]]
    time_t = bxy.time_getlocaltime(tb_sec)
    doy, pass_hr = time_t[3]+(time_t[0]-2015)*365+np.max([(time_t[0]-2016)*1, time_t[0]*0], axis=0), time_t[4]
    t_date = doy+pass_hr/24.0
    npr = (tb_v - tb_h)/(tb_v + tb_h)
    max_value, min_value, conv = test_def.edge_detect(t_date, npr, sigma_g, seriestype='npr')
    max_value_thaw = max_value[(max_value[:, 1] > 365+60) & (max_value[:, 1] < 365+150)]
    min_value_freeze = min_value[(min_value[:, 1] > 365+250) & (min_value[:, 1] > 365+340)]

    swe_daily, swe_date = read_site.read_measurements(site, "snow", doy)
    if site in ['2065', '2081']:
        t_air, t_air_date = read_site.read_measurements(site, "Air Temperature Average (degC)", doy, hr=pass_hr)
    else:
        t_air, t_air_date = read_site.read_measurements(site, "Air Temperature Observed (degC)", doy, hr=pass_hr)

    # add ascat time series
    ascat_series, ascat_pass_pm = data_process.get_ascat_series(site)
    max_ascat, min_ascat, conv_ascat = test_def.edge_detect\
        (ascat_series[:, 0], ascat_series[:, 1], sigma_ascat, seriestype='sig')  # days of time series started from 2015

    # located falling edges close to the rising edge of npr
    npr_edge_thaw = max_value_thaw[np.argmax(max_value_thaw[:, -1])]
    date_npr_thaw = npr_edge_thaw[1]
    min_ascat_thaw = min_ascat[np.abs(min_ascat[:, 1]-date_npr_thaw)<30]
    fall_edge_ascat = min_ascat_thaw[np.argmin(min_ascat_thaw[:, -1])]
    winter_ascat_conv = conv_ascat[1][(conv_ascat[0] > 365) & (conv_ascat[0] < 365 + 60)]
    noise_conv_ascat = np.array([np.mean(winter_ascat_conv), np.std(winter_ascat_conv)])  # mean and std

    # plotting
    fig0 = plt.figure()
    ax1 = fig0.add_subplot(3, 1, 1)
    ax2 = fig0.add_subplot(3, 1, 3)
    ax3 = fig0.add_subplot(3, 1, 2)
    ax1.plot(t_date, npr)
    ax1_1 = ax1.twinx()
    ax1_1.plot(conv[0], conv[1], 'g-')
    swe_daily[swe_daily < -20] = np.nan
    ax2.plot(swe_date, swe_daily)
    ax2_2 = ax2.twinx()
    ax2_2.plot(t_air_date, t_air, 'k.')
    ax3.plot(ascat_series[:, 0], ascat_series[:, 1])
    ax3_3 = ax3.twinx()
    ax3_3.plot(conv_ascat[0], conv_ascat[1], 'g-')


    # select the nearest peaks of indicator to the edge
    # max2, min2 = test_def.get_peak(npr, 0.005, t_date)
    # max_index, min_index = [], []
    # for max_date in max_value_thaw[:, 1]:
    #     max_index.append(np.argmin(np.abs(max2[:, 1]-max_date)))
    # for min_date in min_value_freeze[:, 1]:
    #     min_index.append(np.argmin(np.abs(min2[:, 1]-min_date)))
    # max_value_thaw, min_value_freeze = max2[max_index], min2[min_index]


    # vertical lines for onsets
    text0_y = np.max(npr)*np.array([1 - 0.1 * i for i in np.arange(0, max_value_thaw[:, 1].size)])
    for onset0, y0 in zip(max_value_thaw[:, 1], text0_y):
        ax1.axvline(x=onset0, color='r')
        ax2.axvline(x=onset0, color='r')
        t_label = '%.2f' % onset0
        ax1.text(onset0, y0, t_label, va='top')
        # add a color point
        print 'the thawing date', t_air_date[np.abs(t_air_date-onset0)<1]
        ax2_2.plot(t_air_date[np.abs(t_air_date-onset0)<1], t_air[np.abs(t_air_date-onset0)<1], 'r.')
    for onset1 in min_value_freeze[:, 1]:
        ax1.axvline(x=onset1, color='b')
        ax2.axvline(x=onset1, color='b')
    ax3.axvline(x=fall_edge_ascat[1], ls='--')

    for ax in [ax1, ax1_1, ax2, ax2_2, ax3]:
        ax.set_xlim([366, 366+366])
    ax2_2.axhline(y=0)


    fig_name = 'result_08_01/test%s.png' % (site)
    plt.savefig(fig_name)
    plt.close()


def ak_series(doy_array, att_list=['cell_tb_v_aft', 'cell_tb_h_aft'], ascat_atts=['resample', 'incidence']):

    # read smap
    date_str = []
    for doy0 in doy_array:
        date_str0 = bxy.doy2date(doy0, fmt='%Y%m%d')
        date_str.append(date_str0)
    h5_path = 'result_08_01/area/smap_area_result'
    # 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5'
    h5_list = sorted(os.listdir(h5_path))

    # read asacat
    if len(ascat_atts)>0:
        ascat_directory = 'result_08_01/ascat_resample_AS'
        match_list = []  # a list for matching the re-sampled data such as re-sampled sigma
        for att0 in ascat_atts:
            match0 = 'result_08_01/ascat_resample_AS/new/*_%s.npy' % att0
            match_list.append(sorted(glob.glob(match0)))
            # resample_match = 'result_08_01/ascat_resample_AS/new/*_resample.npy'  # an example
        resample_list, inc_list, pass_list = match_list[0], match_list[1], match_list[2]
        #     sorted(glob.glob('result_08_01/ascat_resample_AS/new/*_resample.npy'))
        # inc_list = sorted(glob.glob('result_08_01/ascat_resample_AS/new/*_incidence.npy'))
        if len(inc_list) != len(resample_list):
            print 'ascat files number error: %d are resampled, %d have incidence' % (len(inc_list), len(resample_list))
            return 0
        ascat_dict = {} # initial
        for att1 in ascat_atts:
            ascat_dict[att1] = np.zeros([300, 300, len(resample_list)])-999

        for i_date in range(0, len(resample_list)):  # read resample
            tp_value = np.load(resample_list[i_date])
            ascat_dict['resample'][:, :, i_date] = tp_value  # resample value
            tp_value2 = np.load(inc_list[i_date])
            ascat_dict['incidence'][:, :, i_date] = tp_value2
            tp_value3 = np.load(pass_list[i_date])
            ascat_dict['pass_utc'][:, :, i_date] = tp_value3
        # for i_date, resample0_path in enumerate(resample_list):  # read resample
        #     ymd_hr = resample0_path.split('/')[3].split('_')[1:3]
        #     doy = bxy.get_doy([ymd_hr[0]])
        #     hr = int(ymd_hr[1])*1.0/24
        #     # time_array[i_date] = doy + hr  # time
        #     f_path = resample0_path
        #     tp_value = np.load(f_path)
        #     ascat_dict['resample'][:, :, i_date] = tp_value  # resample value
        #     f_path2 = 'result_08_01/ascat_resample_AS/new/ascat_%s_%s_incidence.npy' % (ymd_hr[0], ymd_hr[1])  # ascat_20160101_1_incidence.npy
        #     tp_value2 = np.load(f_path2)
        #     ascat_dict['incidence'][:, :, i_date] = tp_value2
        # initial a h5 file
        h50 = h5py.File('result_08_01/area/combine_result/ascat_2016_3d.h5', 'a')
        # 'resample', 'incidence'
        for att in ascat_atts:
            print 'the h5 attribute is ', att
            if att in h50.keys():
                print 'the %s existed, no modification' % att
                continue
            else:
                h50[att] = ascat_dict[att]
        # add latitude, longitude of h5 files
        for ll_i, ll in enumerate(['latitude', 'longitude']):
            if ll in h50.keys():
                print 'the %s existed, no modification' % ll
                continue
            elif ll_i == 0:
                print 'the %s has been added' % ll
                h50[ll] = np.load('./result_05_01/onset_result/lon_ease_grid.npy')
            elif ll_i == 1:
                print 'the %s has been added' % ll
                h50[ll] = np.load('./result_05_01/onset_result/lat_ease_grid.npy')
        print 'the keys in h5 file include', h50.keys()
        h50.close()
        for att2 in ascat_atts:
            ascat_name = 'result_08_01/area/combine_result/ascat_%s_3d.npy' % att2
            np.save(ascat_name, ascat_dict[att2])

    # initial
    att_value_all = {}
    for att0 in att_list:
        att_value_all[att0] = np.zeros([90, 100, len(date_str)])-999

    i_date = 0
    nodata_id = 0
    if len(att_list)>0:
        for i_date, resample0_path in enumerate(date_str):
            h5_fname = 'SMAP_alaska_A_GRID_%s.h5' % resample0_path
            if h5_fname not in h5_list:
                print 'no data on %s' % resample0_path
                with open('smap_series_no_data.out', 'a') as writer1:
                    if nodata_id == 0:  # add a line of current time
                        time0 = datetime.now().timetuple()
                        time_str = '%d-%d, %d: %d \n' % (time0.tm_mon, time0.tm_mday, time0.tm_hour, time0.tm_min)
                        writer1.write(time_str)
                        writer1.write(resample0_path)
                        writer1.write('\n')
                        nodata_id += 1
                    else:
                        writer1.write(resample0_path)
                        writer1.write('\n')
                i_date += 1
                continue
            else:
                h0 = h5py.File(h5_path+'/'+h5_fname)
                for att0 in att_list:
                    if (h0[att0].value==0).any():
                        pause=0
                    att_value_all[att0][:, :, i_date] = h0[att0].value
                i_date += 1
                    # tbv_a_ak_series[i_date], tbh_a_ak_series[i_date] = h0['cell_tb_v_aft'].value, h0['cell_tb_h_aft'].value
        for att0 in att_list:
            save_name = 'result_08_01/area/combine_result/smap_%s.npy' % att0
            np.save(save_name, att_value_all[att0])


    # np.savetxt('result_08_01/area/combine_result/ascat_smap_doy.txt', doy0, fmt='%d', delimiter=',')
    return 0
    # onset


def combine_detection(thaw_window, freeze_window,
                      ascat_detect=False, tb_detect=False, npr_detect=True,
                      odd_plot=False, odd_plot_ascat=False, sigma_npr=7, sigma_ascat=3, single_pixel=False):
    melt_zone = 21
    smap_tbv = np.load('result_08_01/area/combine_result/smap_cell_tb_v_aft.npy')
    smap_tbh = np.load('result_08_01/area/combine_result/smap_cell_tb_h_aft.npy')
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102'
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.copy().ravel()
    lats_1d = h0['cell_lat'].value.copy().ravel()
    h0.close()
    # updated, read ascat data from h5 files
    ascat_h0 = h5py.File('result_08_01/area/combine_result/ascat_2016_3d.h5')
    ascat_sigma = ascat_h0['resample'].value.copy()
    ascat_incidence = ascat_h0['incidence'].value.copy()
    ascat_pass_utc = ascat_h0['pass_utc'].value.copy()
    ascat_lat = ascat_h0['latitude'].value.copy()
    ascat_lon = ascat_h0['longitude'].value.copy()
    ascat_h0.close()
    # ascat_sigma = np.load('result_08_01/area/combine_result/ascat_resample_3d.npy')
    # ascat_incidence = np.load('result_08_01/area/combine_result/ascat_incidence_3d.npy')
    tbv_2d = smap_tbv.reshape(-1, smap_tbv.shape[2])
    tbh_2d = smap_tbh.reshape(-1, smap_tbh.shape[2])
    row_table = np.loadtxt('ascat_row_table.txt', delimiter=',')
    col_table = np.loadtxt('ascat_col_table.txt', delimiter=',')
    # thaw window for ascat1
    thaw_ini_sec = bxy.get_secs([2016, 1, 30, 0, 0, 0], reftime=[2015, 1, 1, 0])
    thaw_end_sec = bxy.get_secs([2016, 6, 30, 0, 0, 0], reftime=[2015, 1, 1, 0])
    # change window into seconds
    ini_seconds = bxy.get_secs([2016, 1, 1, 0, 0, 0], reftime=[2015, 1, 1, 0])
    seconds_2015 = bxy.get_secs([2015, 1, 1, 0, 0, 0], reftime=[2000, 1, 1, 0])
    thaw_window = (thaw_window-1) * 3600 * 24 + ini_seconds
    freeze_window = (freeze_window-1) * 3600 * 24 + ini_seconds

    # output initial
    smap_onset0 = np.zeros([smap_tbv.shape[0], smap_tbv.shape[1]])
    nan_out_idx = 0
    # check locations:
    # h00 = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20160103.h5')
    # ease_lat, ease_lon = h00[u'cell_lat'].value, h00[u'cell_lon'].value
    # rc = bxy.geo_2_row([ease_lon, ease_lat], [-146.73390, 65.12422])
    # check_series = smap_tbv[rc[0], rc[1], :]
    # h5_list = sorted(glob.glob('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_*.h5'))
    # check_series2 = np.zeros(len(h5_list)) - 88
    # for i0, h5_name0 in enumerate(h5_list):
    #     daily_h0 = h5py.File(h5_name0, 'r')
    #     daily_tbv = daily_h0[u'cell_tb_v_aft'].value
    #     check_series2[i0] = daily_tbv[45, 48]
    #     daily_h0.close()
    # check_series3 = smap_tbv[45, 48, :]

    t_date = (np.arange(-60, 366+60)-1) * 3600 * 24 + ini_seconds
    t_ascat = np.loadtxt('ascat_secs_series.txt')
    onset_map_0_1d = np.zeros(tbv_2d.shape[0]) - 999
    onset_map_1_1d = np.zeros(tbv_2d.shape[0]) - 999
    onset_thaw_ascat, onset_melt_ascat, conv_melt_ascat = \
        np.zeros(tbv_2d.shape[0]) - 999, np.zeros(tbv_2d.shape[0]) - 999, np.zeros(tbv_2d.shape[0]) - 999
    onset_map_0_1d_tb, onset_map_1_1d_tb = np.zeros(tbv_2d.shape[0]) - 999, np.zeros(tbv_2d.shape[0]) - 999

    # ascat_sigma = np.load('result_08_01/area/combine_result/ascat_resample.npy')
    # ascat_incidence = np.load('result_08_01/area/combine_result/ascat_incidence.npy')
    sigma_2d = ascat_sigma.reshape(-1, ascat_sigma.shape[2])
    incidence_2d = ascat_incidence.reshape(-1, ascat_incidence.shape[2])
    onset_map_0_1d_ascat, onset_map_1_1d_ascat = np.zeros(sigma_2d.shape[0])-999, np.zeros(sigma_2d.shape[0])-999
    nan_out_idx = 0
    ascat_mask = np.load('./result_05_01/other_product/mask_ease2_125N.npy')
    ascat_mask_1d = ascat_mask.ravel()
    land_id_ascat = np.where(ascat_mask_1d == True)[0]
     # check odd pixel
    if odd_plot_ascat is not False:
        simga0, incidence0 = sigma_2d[odd_plot_ascat], incidence_2d[odd_plot_ascat]
        incidence0[incidence0==0]=np.nan
        incidence0[incidence0==-999]=np.nan
        simga0 -= (incidence0-45)*-0.12
        max_value, min_value, conv = test_def.edge_detect(t_date, simga0, sigma_npr, seriestype='sig')
        max_value_thaw = max_value[(max_value[:, 1] > 60) & (max_value[:, 1] < 150)]
        min_value_freeze = min_value[(min_value[:, 1] > 250) & (min_value[:, 1] < 340)]
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(1, 1, 1)
        id_nonan = ~np.isnan(simga0)
        ax0.plot(t_date[id_nonan], simga0[id_nonan])
        ax01 = ax0.twinx()
        ax01.plot(conv[0], conv[1], 'g-')
        # thaw_edge = max_value_thaw[:, 1][max_value_thaw[:, -1].argmax()]
        # ax0.axvline(x=thaw_edge)
        # print 'target no. %d thawed on doy %d' % (siries_plot, thaw_edge)
        print onset_map_0_1d[odd_plot]
        plt.savefig('result_08_01/tbv_test_ascat_w20151102_series.png')
        plt.close()


    # only loops the land area
    if odd_plot is False:
        mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
        # onset0 = np.ma.masked_array(onset0, mask=[(onset0==0)|(mask==0)])
        mask_1d = mask.reshape(1, -1)[0]
        land_id = np.where(mask_1d != 0)[0]
    else:
        land_id = [odd_plot]
    for i0 in land_id:
        smap_series_v, smap_series_h = tbv_2d[i0], tbh_2d[i0]
        lon0, lat0 = lons_1d[i0], lats_1d[i0]
        smap_series_v[smap_series_v<0] = np.nan
        smap_series_h[smap_series_h<0] = np.nan
        if sum(np.isnan(smap_series_h)) > 300:
            bxy.odd_out('nan_value_smap.out', i0, nodata_id=nan_out_idx)
            nan_out_idx += 1
            continue
        else:
            # npr onset
            if npr_detect is True:
                if i0 == 4037:
                    pause = 0
                npr = (smap_series_v-smap_series_h)*1.0/(smap_series_v+smap_series_h)
                max_value, min_value, conv = test_def.edge_detect(t_date, npr, sigma_npr, seriestype='npr')
                max_value_thaw = max_value[(max_value[:, 1] > thaw_window[0]) & (max_value[:, 1] < thaw_window[1])]
                min_value_freeze = min_value[(min_value[:, 1] > freeze_window[0]) & (min_value[:, 1] < freeze_window[1])]
                # check positions where onsets doesn't exist.
                if max_value_thaw.size == 0:
                    with open('onset_map0.txt', 'a-') as writer0:
                        writer0.writelines('no thaw onset was find at: %d' % i0)
                else:
                    thaw_onset0 = max_value_thaw[:, 1][max_value_thaw[:, -1].argmax()]
                    thaw_onset0_tuple = bxy.time_getlocaltime([thaw_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                    onset_map_0_1d[i0] = thaw_onset0_tuple[-2][0]
                    melt_zone0 = np.array([thaw_onset0-melt_zone*24*3600, thaw_onset0+melt_zone*24*3600])
                    if min_value_freeze.size>0:
                        onset_map_1_1d[i0] = min_value_freeze[:, 1][min_value_freeze[:, -1].argmin()]
                    else:
                        onset_map_1_1d[i0] = 0
            if ascat_detect == True:
                test = 0
                # 9xN array for ascat measurements within 36km
                lat_9 = ascat_lat[row_table[i0].astype(int), col_table[i0].astype(int)]
                lon_9 = ascat_lon[row_table[i0].astype(int), col_table[i0].astype(int)]
                dis_9 = bxy.cal_dis(lat0, lon0, lat_9, lon_9)  # distance to the center of 36 km  pixel
                sigma_series_9 = ascat_sigma[row_table[i0].astype(int), col_table[i0].astype(int), :]
                incidence_series_9 = ascat_incidence[row_table[i0].astype(int), col_table[i0].astype(int), :]
                t_ascat_9 = ascat_pass_utc[row_table[i0].astype(int), col_table[i0].astype(int), :]
                index_invalid_0 = (incidence_series_9) < 30 | (incidence_series_9 > 55) \
                                                         | (sigma_series_9 == -999) | (sigma_series_9 == 0)
                sigma_series_9[index_invalid_0], incidence_series_9[index_invalid_0], t_ascat_9[index_invalid_0] = \
                    np.nan, np.nan, np.nan

                # 2rd distance interpolation if necessary
                for daily_9 in sigma_series_9:
                    index_9 = (daily_9 != -999) & (daily_9 != 0)
                    if sum(index_9) > 0:
                        iter = 0
                        # sigma_series_mean
                    else:
                        iter = 1
                # index9 = (sigma_series_9 != -999) & (sigma_series_9 != 0)
                # if sum(index9) > 0:
                #     interp = 0
                #     d, vals = dis_9[index9], sigma_series_9[index9]
                # else:
                #     interp = -
                # sigma_series_mean = np.zeros(sigma_series_9.shape[1])
                # incidence_series_mean = np.zeros(sigma_series_9.shape[1])
                # t_ascat = np.zeros(sigma_series_9.shape[1])
                #
                # for i2 in range(0, sigma_series_9.shape[1]):
                #     sigma_series_mean[i2] = np.nanmean(sigma_series_9[:, i0]) \
                #         if np.isnan(sigma_series_9[:, i0]).all() else np.nan
                #     incidence_series_mean = np.nanmean(incidence_series_9[:, i0]) \
                #         if np.isnan(incidence_series_9[:, i0]).all() else np.nan
                #     t_ascat = np.nanmean(t_ascat_9[:, i0]) \
                #         if np.isnan(t_ascat_9[:, i0]).all() else np.nan
                sigma_series_mean = np.nanmean(sigma_series_9, axis=0)
                incidence_series_mean = np.nanmean(incidence_series_9, axis=0)
                t_ascat = np.nanmean(t_ascat_9, axis=0)

                # sigma_series_9_ma = np.ma.masked_array(sigma_series_9, mask=0])
                # sigma_series_9_ma = np.ma.masked_array(sigma_series_9, mask=-999])
                valid_index = (sigma_series_mean > -25) & (sigma_series_mean < -0.1)


                if sum(valid_index)<150:
                        # set a unvalid label
                    continue
                else:
                    a, b = np.polyfit(incidence_series_mean[valid_index], sigma_series_mean[valid_index], 1)  # angular
                    secs_valid = t_ascat[valid_index]
                    secs_valid-=seconds_2015
                    series_valid = sigma_series_mean[valid_index] - (incidence_series_mean[valid_index]-45)*a
                    # calculate the daily average based on pass seconds
                    ini_seconds_tp = secs_valid[0]
                    ini_tuple_tp = bxy.time_getlocaltime([ini_seconds_tp], ref_time=[2015, 1, 1, 0], t_out='US/Alaska')
                    # ini_tuple_tp2 = bxy.time_getlocaltime(secs_valid, ref_time=[2015, 1, 1, 0], t_out='utc')
                    # ini_secs = bxy.get_secs\
                    #     ([ini_tuple_tp[0], ini_tuple_tp[0], ini_tuple_tp[0], 18, 0, 0], reftime=[2015, 1, 1, 0])
                    non_outlier = bxy.reject_outliers(series_valid, m=100)
                    secs_valid = secs_valid[non_outlier]
                    series_valid = series_valid[non_outlier]
                    max_value_a, min_value_a, conv_a =\
                        test_def.edge_detect(secs_valid, series_valid, sigma_npr, seriestype='sig')
                    max_value_no_use, min_value_a, conv_a =\
                        test_def.edge_detect(secs_valid, series_valid, sigma_ascat, seriestype='sig')
                    # thaw onset and melt onset
                    max_value_a_thaw = max_value_a[(max_value_a[:, 1] > thaw_window[0]) & (max_value_a[:, 1] < thaw_window[1])]
                    if max_value_a_thaw[:, -1].size<1: # temp check get time tuple of thaw onsets candidate
                        # save index of pixels where thaw onset cannot be located
                        with open('ascat_no_thawing_onset.txt', mode='w') as f00:
                            f00.write('no thawing onset was located at 1d index: %d' % i0)
                        # temp_t_tuple = bxy.time_getlocaltime(max_value_a[:, 1], ref_time=[2015, 1, 1, 0], t_out='utc')
                        # temp_doy = temp_t_tuple[-2]
                        # doy_0 = bxy.time_getlocaltime(secs_valid, ref_time=[2015, 1, 1, 0], t_out='utc')[-2]
                        # plot_funcs.quick_plot(doy_0, series_valid)
                        # pause = 0
                        # ascat_thaw_onset0 = 0
                        # check finished
                    else:
                        ascat_thaw_onset0 = max_value_a_thaw[:, 1][max_value_a_thaw[:, -1].argmax()]  # thaw
                    # melt_zone0 = [ini_seconds*30*24*3600, ascat_thaw_onset0]
                    min_value_snowmelt = min_value_a[(min_value_a[:, 1] > melt_zone0[0]) & (min_value_a[:, 1] < melt_zone0[1])]
                    if min_value_snowmelt[:, -1].size < 1:
                        melt_onset0 = 0
                        melt_conv = -999
                    else:
                        melt_onset0 = min_value_snowmelt[:, 1][min_value_snowmelt[:, -1].argmin()]
                        melt_conv = min_value_snowmelt[:, -1][min_value_snowmelt[:, -1].argmin()]
                        if melt_conv > -0.5:
                            melt_onset0 = 0
                        # bxy.time_getlocaltime([melt_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                    pause = 0
                    # secs to day of year
                    thaw_onset0_tuple2 = bxy.time_getlocaltime([ascat_thaw_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                    melt_onset0_tuple2 = bxy.time_getlocaltime([melt_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                    onset_thaw_ascat[i0] = thaw_onset0_tuple2[-2][0]
                    onset_melt_ascat[i0] = melt_onset0_tuple2[-2][0]
                    conv_melt_ascat[i0] = melt_conv

            if i0 == odd_plot:
                # time series:
                print odd_plot
                print sum(valid_index)
                print 'the angular coefficient is: ', a
                max_value, min_value, conv = test_def.edge_detect(t_date, npr, sigma_npr, seriestype='npr')
                # max_value_a, min_value_a, conv_a =\
                #         test_def.edge_detect(secs_valid, series_valid, sigma_npr, seriestype='sig')
                # smap
                max_value_thaw = max_value[(max_value[:, 1] > thaw_window[0]) & (max_value[:, 1] < thaw_window[1])]
                thaw_onset0 = max_value_thaw[:, 1][max_value_thaw[:, -1].argmax()]
                thaw_onset0_tuple = bxy.time_getlocaltime([thaw_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                onset_odd_smap = thaw_onset0_tuple[-2][0]
                # ascat
                max_value_thaw_a = max_value_a[(max_value_a[:, 1] > thaw_window[0]) & (max_value_a[:, 1] < thaw_window[1])]
                if max_value_thaw_a[:, -1].size<1:
                    pause = 0
                ascat_thaw_onset0 = max_value_thaw_a[:, 1][max_value_thaw_a[:, -1].argmax()]  # thaw
                # melt_zone0 = [ini_seconds*30*24*3600, ascat_thaw_onset0]
                min_value_snowmelt = min_value_a[(min_value_a[:, 1] > melt_zone0[0]) & (min_value_a[:, 1] < melt_zone0[1])]
                if min_value_snowmelt[:, -1].size < 1:
                    melt_onset0 = 0
                else:
                    melt_onset0 = min_value_snowmelt[:, 1][min_value_snowmelt[:, -1].argmin()]
                # transform to doy then plot
                t_x_odd = []
                for item0 in [t_date, secs_valid, conv[0], conv_a[0]]:
                    t0 = bxy.time_getlocaltime(item0, ref_time=[2015, 1, 1, 0])
                    t0_doy = (t0[0]-2016)*366 + t0[-2]+t0[-1]/24.0
                    t_x_odd.append(t0_doy)
                plot_funcs.plot_subplot([[t_x_odd[0], npr], [t_x_odd[1], series_valid], [t_x_odd[1], incidence_series_mean[valid_index]]],
                                        [[t_x_odd[2], conv[1]], [t_x_odd[3], conv_a[1]]],
                                        vline=[onset_map_0_1d[i0], onset_thaw_ascat[i0], onset_melt_ascat[i0]])

                # for row_i, col_i in zip(row_table[i0], col_table[i0]):  # 9 corresponded ascat measurements
                #     sigma_series_i = ascat_sigma[row_i, col_i, :]
                #     sigma_incidence_i = ascat_incidence[row_i, col_i, :]
                #     valid_index = (sigma_series_i > -25) & (sigma_series_i < -0.1)
                #     print 'The days with valid ascat measurements was %d' % sum(valid_index)

            # tb onset
            # if tb_detect is True:
            #     max_value, min_value, conv = test_def.edge_detect(t_date, smap_series_v, sigma_npr, seriestype='tb')
            #     max_value_freeze = max_value[(max_value[:, 1] >= 365+150) & (max_value[:, 1] <= 365+340)]
            #     min_value_thaw = min_value[(min_value[:, 1] >= 365+60) & (min_value[:, 1] <= 365+150)]
            #     if max_value_thaw.size == 0:
            #         with open('onset_map0.txt', 'a-') as writer0:
            #             writer0.writelines('no thaw onset was find at: %d' % i0)
            #     else:
            #         if i0 == 3920:
            #             pause = 0
            #         onset_map_1_1d_tb[i0] = max_value_freeze[:, 1][max_value_freeze[:, -1].argmax()]
            #         onset_map_0_1d_tb[i0] = min_value_thaw[:, 1][min_value_thaw[:, -1].argmin()]
            #         if i0 == 4263:
            #             print 'thaw edge target is: ', onset_map_0_1d[i0]


    if odd_plot is False:
        # save npr onset
        th_name = 'test_onset0_%s.npy' % sigma_npr
        fr_name = 'test_onset1_%s.npy' % sigma_npr
        np.save(th_name, onset_map_0_1d.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save(fr_name, onset_map_1_1d.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save('test_onset0_tb.npy', onset_map_0_1d_tb.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save('test_onset1_tb.npy', onset_map_1_1d_tb.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        # save ascat
        th_ascat = 'thaw_onset_ascat_%d.npy' % sigma_npr
        ml_ascat = 'melt_onset_ascat_%d.npy' % sigma_npr
        ml_conv = 'melt_conv_ascat_%d.npy' % sigma_npr
        np.save(th_ascat, onset_thaw_ascat.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save(ml_ascat, onset_melt_ascat.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save(ml_conv, conv_melt_ascat.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        print onset_map_0_1d.reshape(smap_tbv.shape[0], smap_tbv.shape[1])[47, 67]




        # edge detection
        # for i1 in range(0, 9):
        #     row_num, col_num = row_table[i0, i1], col_table[i0, i1]
        #     if row_num < 0 or col_num < 0:
        #         continue
        #     else:
        #         sigma1 = ascat_sigma[row_num, col_num, :]
        #         incidence1 = ascat_incidence[row_table[i0, i1], col_table[i0, i1], :]
        #         sigma1[(sigma1==0) | (sigma1==-999)] = np.nan
        #         if ~np.isnan(sigma1).all():
        #             pause = 0
        #         # normalized and edge detection

    return 0


def get_ascat_sec():
    file_list = sorted(glob.glob('result_08_01/ascat_resample_AS/new/*resample.npy'))
    sec_list = []
    for f0 in file_list:
        string0 = f0.split('/')[-1].split('_')[1]  # yyymmdd
        string1 = f0.split('/')[-1].split('_')[2]  # hour
        t_list_i = [int(string0[0:4]), int(string0[4:6]), int(string0[6:]), int(string1), 0, 0]
        # obj_list.append(t_obj_i)
        sec_i = bxy.get_secs(t_list_i,reftime=[2015, 1, 1, 0])
        sec_list.append(sec_i)
    np.savetxt('ascat_secs_series.txt', np.array(sec_list).T, delimiter=',', fmt='%.2f')


def station_sigma():
    '''
    quick plot: time series of sigma
    :return:
    '''
    site_nos = site_infos.get_sno_list('string')
    obs=['_D_', '_A_']  # orbit of smap
    k_width = 7
    for site_no in site_nos:
        sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
        data_process.ascat_plot_series(site_no, orb_no=0, inc_plot=True, sigma_g=10,
                                               order=1, txt_path='./result_08_01/point/ascat/ascat_site_series/')
        tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                         seriestype='tb', tbob=obs[1], sig0=k_width, order=1)  # result tb
        tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = test_def.main(site_no, [], sm_wind=7, mode='annual',
                                                                          tbob=obs[1], sig0=k_width, order=1)  # result npr
        doy_ascat = np.modf(sigseries[0])
        if site_no in ['2065', '2081']:
            air_ascat, t_ascat = read_site.read_measurements(site_no, "Air Temperature Average (degC)", doy=doy_ascat[1], hr=doy_ascat[0]*24)
        else:
            air_ascat, t_ascat = read_site.read_measurements(site_no, "Air Temperature Observed (degC)", doy=doy_ascat[1], hr=doy_ascat[0]*24)
        # plot ascat series, ascat edge, and air temperature
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(2, 1, 1)
        ax1 = fig0.add_subplot(2, 1, 2)
        # m1 = ax.scatter(x_value, sigma_mid, c=z_value, cmap=plt.get_cmap('rainbow'))
        valid_air = air_ascat > -99
        m1 = ax0.scatter(air_ascat[valid_air], sigseries[1][valid_air],
                    c=sigseries[0][valid_air], cmap=plt.get_cmap('coolwarm'))
        cax = fig0.add_axes([0.12, 0.9, 0.5, 0.05])
        plt.colorbar(m1, cax=cax, orientation='horizontal')
        ax1.plot(sigseries[0], sigseries[1], 'k.')
        ax1_1 = ax1.twinx()
        ax1_1.plot(sigconv[0]-365, sigconv[1], 'g-')
        figname = 'result_08_01/point/ascat_thaw/%s.png' % site_no
        plt.savefig(figname)
        plt.close()
    return 0


def melt_map(s_info_list):
    # s_info_list = [[0,  -162.7, 69.1], [0,  -155.2, 70.1], [0, -153.5, 68.8], [0, -147.5, 68.8], [0, -153.5, 67.8],
    #                [0, -159.1, 60.5], [0, -159.0, 61.7], [0, -150.3, 64.7], [0, -147.3, 64.4], [0, -150.0, 62.0], [0, -162.5, 65.5]
    #                ,[0, -162.5, 63.0],  [0, -150.3, 66.7], [0, -147.3, 66.7]]
    # # s_info_list = [[1, 1, 1]]  # no special pixel
    points_index = []
    for s_info in s_info_list:
        for kernel0 in [7]:
            # 70.26666, -148.56666
            # s_info = [0, -1, -1]
            odd_latlon = [s_info[2], s_info[1]]
            thaw_win = np.array([30, 180])
            fr_win = np.array([250, 340])
            odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)
            # calculate onset
            if s_info[1]<0:
                # get 1d index
                for d_str in ['20151102']:
                    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % d_str
                    h0 = h5py.File(h5_name)
                    lons_1d = h0['cell_lon'].value.ravel()
                    lats_1d = h0['cell_lat'].value.ravel()
                    dis_1d = bxy.cal_dis(s_info[1], s_info[2], lons_1d, lats_1d)
                    p_index = np.argmin(dis_1d)
                    points_index.append(p_index)
                combine_detection(thaw_win, fr_win, sigma_npr=kernel0, sigma_ascat=3, ascat_detect=True, odd_plot=p_index)
                # write the sepcific pixel infos in a txt file
                with open('odd_pixel_infos.txt', 'a') as odd_info0:
                    odd_info0.write('Odd pixel: %d \n' % p_index)
                    odd_info0.write('1d index (36 km grid): %d \n' % p_index)
                    odd_info0.write('location info: %.2f, %.2f, %.2f \n' % (s_info[0], s_info[1], s_info[2]))
                    odd_info0.close()
                # copy time series map to the target folder and named by odd pixel_no
                cm_line = "cp result_08_01/test_plot_subplot.png result_08_01/temp/temp_comparison/pixel_no_%d.png" % p_index
                os.system(cm_line)
            else:
                combine_detection(thaw_win, fr_win, sigma_npr=kernel0, sigma_ascat=3, ascat_detect=True)
    np.savetxt('pixel_index.txt', np.array(points_index), delimiter=',', fmt='%d')

if __name__ == "__main__":
    # melt_map([[1, 1, 1]])
    melt_map([[0,  -162.7, 69.1], [0,  -155.2, 69.5], [0, -153.5, 68.8], [0, -147.5, 68.2], [0, -153.5, 67.8],
                   [0, -159.1, 60.5], [0, -159.0, 62.2], [0, -163.0, 61.2], [0, -150.3, 64.7], [0, -147.3, 64.4], [0, -150.0, 62.0], [0, -162.5, 65.5]
                   ,[0, -162.5, 63.0],  [0, -150.3, 66.7], [0, -147.3, 66.2]])
    # quit0()
    # add all odd pixels and labeled in the melt map
    if os.path.exists('pixel_index.txt'):
        odd_pixel_index = np.loadtxt('pixel_index.txt')
        data_process.ascat_onset_map('A', product='customize', mask=False, mode=['_norm_'], points_index=odd_pixel_index.astype(int),
                                    version='old', std=7, f_win=np.array([250, 340]), t_win=np.array([30, 180]),
                                     custom=['thaw_onset_ascat_7.npy', 'melt_onset_ascat_7.npy', 'melt_conv_ascat_7.npy'])
    quit0()
    # # 0827/2018
    # s_info_list = [[0,  -162.7, 69.1], [0,  -163.5, 68.1], [0,  -163.0, 68.1], [0,  -162.7, 69.1]]
    # for s_info in s_info_list:
    #     melt_map(s_info)
    # quit0()
    # doy = np.arange(-60, 366+60)
    # # # Read_radar.read_ascat_alaska(doy)
    # # for doy0 in doy:
    # #     t_str = bxy.doy2date(doy0, fmt="%Y%m%d")  # 1 is 20160101
    # #     spt_quick.ascat_area_plot2(t_str)
    # ak_series(doy, ascat_atts=['resample', 'incidence', 'pass_utc'])
    # quit0()
    # station_sigma()
    # quit0()
    # # 0821/2018
    # spt_quick.ascat_point_plot()
    # # 0815/2018
    for kernel0 in [7]:
        s_info = site_infos.change_site('1090')
        # 70.26666, -148.56666
        #
        s_info = [0, s_info[2], s_info[1]]
        s_info = [0,  -162.7, 69.1]
        # s_info = [0, -1, -1]
        odd_latlon = [s_info[2], s_info[1]]
        thaw_win = np.array([30, 180])
        fr_win = np.array([250, 340])
        odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)
        # calculate onset
        if s_info[1]<0:
            # get 1d index
            for d_str in ['20151102']:
                h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % d_str
                h0 = h5py.File(h5_name)
                lons_1d = h0['cell_lon'].value.ravel()
                lats_1d = h0['cell_lat'].value.ravel()
                dis_1d = bxy.cal_dis(s_info[1], s_info[2], lons_1d, lats_1d)
                p_index = np.argmin(dis_1d)
            combine_detection(thaw_win, fr_win, sigma_npr=kernel0, ascat_detect=True, odd_plot=p_index)
        else:
            combine_detection(thaw_win, fr_win, sigma_npr=kernel0, ascat_detect=True)
    data_process.ascat_onset_map('A', odd_point=np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]]), product='customize', mask=False, mode=['_norm_'],
                                    version='old', std=kernel0, f_win=fr_win, t_win=thaw_win, custom=['thaw_onset_ascat_7.npy', 'melt_onset_ascat_7.npy'])
    quit0()
    # # 0704/2018
    # # data_process.smap_ascat_position()
    # # ak_series(np.arange(-60, 366+55)
    # new_process(['947'])
    # draw_pie_landcover()
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    onset_latlon = np.loadtxt('result_08_01/point/onset_result/onset_result.csv', delimiter=',')
    for kernel0 in [7]:
        s_info = site_infos.change_site('968')
        # 70.26666, -148.56666
        # s_info = [0, 61.6155, -142.9327]
        s_info = [0, 59.5, -157.8]
        s_info = [0, -1, -1]
        odd_latlon = [s_info[2], s_info[1]]
        thaw_win = [30, 180]
        fr_win = [250, 340]
        odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)
        if s_info[1]>0:
            combine_detection(thaw_win, fr_win, sigma_npr=kernel0, odd_plot=odd_points_1d)
        else:
            combine_detection(thaw_win, fr_win, sigma_npr=kernel0)
        print 'the target is (%.3f, %.3f)' % (s_info[2], s_info[1])

        data_process.ascat_onset_map('A', odd_point=np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]]), product='grid_test', mask=False, mode=['_norm_'],
                                    version='old', std=kernel0, f_win=fr_win, t_win=thaw_win)
        # data_process.ascat_onset_map('A', odd_point=onset_latlon[:, 4:9], product='grid_test', mask=False, mode=['_norm_'],
        #                             version='old', std=kernel0, f_win=fr_win, t_win=thaw_win)
    quit0()
    doy = np.arange(-60, 366+60)
    for doy0 in doy:
        Read_radar.read_ascat_alaska(doy0, year0=2016)
    for doy0 in doy:
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d")  # 1 is 20160101
        spt_quick.ascat_area_plot2(t_str)
    ak_series(np.arange(-60, 366+55))
    quit0()

    for doy0 in range(365, 366+60):
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d")  # 1 is 20160101
        # spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap/', orbit='A')
        spt_quick.ascat_area_plot2(t_str)
    #ak_series()  # save as smap 3d array
    #data_process.smap_alaska_grid()

    s_info = site_infos.change_site('968')
    # 70.26666, -148.56666
    odd_latlon = [s_info[2], s_info[1]]
    odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)
    print 'the target is (%.3f, %.3f)' % (s_info[2], s_info[1])
    combine_detection(odd_plot=odd_points_1d, odd_plot_ascat=6534)

    data_process.ascat_onset_map('A', odd_point=[odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]], product='grid_test', mask=False, std=4, mode=['_norm_'],
                                version='old')
    data_process.ascat_onset_map(['AS'], odd_point=[ s_info[2], s_info[1]])
    data_process.ascat_onset_map(['AS'], product='npr', odd_point=[s_info[2], s_info[1]], mask=True, version='new')
    quit0()
    for m in ['area_5']:
        print 'test series of pixel in %s' % m
        for ob in ['AS']:
            for mo in ['_norm_']:
                print 'mode is %s, orbit is %s' % (mo, ob)
                data_process.ascat_result_test(m, mode=mo, key=ob, odd_rc=(89, 190))
    quit0()

    # 0614/2018, read alaska again
    Read_radar.radar_read_alaska('_A_', ['alaska'], ['2015.12.01', '2016.12.31'], 'vv')
    # quit0()
    # spt_quick.smap_mask()
    doy = np.arange(-60, 366)
    doy_id = 0
    for doy0 in doy:
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d")
        print doy0, t_str
        spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap/', orbit='A')
        doy_id += 1
        if doy_id > 10:
            break
    data_process.ascat_onset_map('A', odd_point=[], product='grid_test', mask=False, std=4, mode=['_norm_'],
                                version='old')

    quit0()
    ak_series()  # save as smap 3d array
    data_process.smap_alaska_grid(2017)
    quit0()
    combine_detection()
    quit0()
    doy = np.arange(-90, 366+60)
    for doy0 in doy:
        Read_radar.read_ascat_alaska(doy0, year0=2016)
    quit0()
    # 0612/2018, t_air, edges and SNR
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210',
                '2212', '2213']  #'1089'\
    site_nos = ['967', '1090', '2210']
    # site_nos = ['968']
    for site0 in site_nos:
        print site0
        t_air_edges(site0)
    quit0()
    # 0508/2018, some area work
    site_nos_new = ['9001', '9002', '9003', '9004', '9005', '9006', '9007',
                     '947', '949', '950',
                    '960', '962', '967', '968', '1090', '1175', '1177', '1233',
                    '2065', '2081', '2210', '2211', '2212', '2213']  # '957', '948', '958', '963', '2080',
    for sno in site_nos_new:
        print sno
        # 1. read corner infomation
        # s_info = site_infos.change_site(sno)
        # s_info_t = np.array([s_info[2], s_info[1], int(s_info[0])]).reshape(3, -1)
        # print sno
        # spt_quick.ascat_sub_pixel(s_info_t, dis0=4, site_nos=[sno])
        # 2. after shape file
        # gdal_clips(sno, ipt='lc')
        # gdal_clips(sno, ipt='snowf')
        # ascat_snow_lc(sno)

    # 3. combine snow and landcover
    # for sno in site_nos_new:
    #     print sno
    #     check_lc_snow_ascat(sno)
    ascat_snowlc_npy(site_nos_new)
    rid = 1
    sno_sp = regions_extract(rid)
    beams = 'fore'
    # land_cover(rid, sno_sp, ['ever', 'decid', 'shrub'], False)  # 2nd var is a list of sites
    # check_pass_time(rid)
    regions_plotting(region_id=rid, att_xyz=['# date', beams, 'snowf'], xlim0=[1, 366])
    regions_plotting(region_id=rid, att_xyz=['# date', beams, 'tair'], xlim0=[1, 366])

    # # 0516/2018
    t0 = 130
    for sno in site_nos_new:
        tb_melt_window(sno)
    beams = 'fore'
    sno_sp = regions_extract(rid)
    print sno_sp
    for sno in sno_sp:
        #regions_plotting(sno)
        # regions_plotting(region_id=rid, site_no=sno, att_xyz=['tair', beams, '# date'])  # xlim0=[1, 366]
        regions_plotting(region_id=rid, site_no=sno, att_xyz=['# date', beams, 'ID'], xlim0=[1, 366], xv=t0)
    quit0()
    ## 0514/2018 updated, transient freezing effect
    test_method('thaw', txt=True)
    # disscus_sm_variation()
    # site_nos = ['947', '968']
    # sub_no = 1
    # for sn in site_nos:
    #     onset_01, dist = smap_ft_compare(sn, period=['all', 0, 365], orb=1, subplot_id=sub_no)
    #     # change sub_no to draw two rows
    #     if sub_no > 1:
    #         sub_no = 1
    #     else:
    #         sub_no = 2
    new_process(['947'])
    #discuss_combining()
    quit0()
    plot_funcs.plot_tair_npr_onset('result_08_01/onset_result/onset_tair_npr.txt')
    # quit0()
    # 0508/2018, some area work
    site_nos_new = ['957', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'ns6', 'ns7']
    site_nos_new = ['957', '9001', '9002', '9003', '9004', '9005', '9006', '9007']  # '957',

    site_nos_new = ['957', '9001', '9002', '9003', '9004', '9005', '9006', '9007',
                    '948', '958', '963', '2080', '947', '949', '950',
                    '960', '962', '967', '968', '1090', '1175', '1177', '1233',
                    '2065', '2081', '2210', '2211', '2212', '2213'
                    ]
    for sno in site_nos_new:
        print sno
        # s_info = site_infos.change_site(sno)
        # s_info_t = np.array([s_info[2], s_info[1], int(s_info[0])]).reshape(3, -1)
        # print sno
        # spt_quick.ascat_sub_pixel(s_info_t, dis0=4, site_nos=[sno])
        # after shape file
        # gdal_clips(sno, ipt='lc')
        # gdal_clips(sno, ipt='snowf')
        # ascat_snow_lc(sno)
        # check_lc_snow_ascat(sno)
    ascat_snowlc_npy(site_nos_new)
    rid = 5
    sno_sp = regions_extract(rid)
    beams = 'fore'
    land_cover(rid, sno_sp, ['ever', 'decid', 'shrub'], False)  # 2nd var is a list of sites
    check_pass_time(rid)
    regions_plotting(region_id=rid, att_xyz=['# date', beams, 'snowf'], xlim0=[1, 366])
    regions_plotting(region_id=rid, att_xyz=['# date', beams, 'tair'], xlim0=[1, 366])
    # regions_plotting(region_id=rid, att_xyz=['tair', beams, 'snowf'])
    # regions_plotting(region_id=rid, att_xyz=['tair', beams, 'swe'])
    # regions_plotting(region_id=rid, att_xyz=['tair', beams, 'ID'])
    # regions_plotting(region_id=rid, att_xyz=['tair', beams, 'dswe'])
    # regions_plotting(region_id=rid, att_xyz=['dswe', beams, 'swe'])
    for sno in sno_sp:
        #regions_plotting(sno)
        # regions_plotting(region_id=rid, site_no=sno, att_xyz=['tair', beams, '# date'])  # xlim0=[1, 366]
        regions_plotting(region_id=rid, site_no=sno, att_xyz=['# date', beams, 'ID'], xlim0=[1, 366])
        # regions_plotting(region_id=rid, site_no=sno, att_xyz=['dswe', beams, 'swe'])
    # bxy.test_read_txt()
    quit0()
    tp = [[-147.02034702, 68.58702006],  [-157.93985569, 69.16732504],
        [-156.55042833, 66.33659274], [-162.97367991, 68.26540917], [-161.23434328, 65.2680856],
        [-150.86761887, 65.78373252], [-156.36380702, 62.78914008]]

    data_process.ascat_onset_map(['AS'], odd_point=tp)
    data_process.ascat_onset_map(['AS'], product='npr', odd_point=tp, mask=True, version='new')
    quit0()
    odd_points = [[-156.55042833, 66.33659274], ]
    tp = [[-153.38525605, 60.80840073],
        [-153.48717832, 62.25746204], [-148.24980309, 61.72569201], [-148.41475430, 61.36947935],
        [-145.91968862, 64.71503588], [-153.50964993, 70.70806989], [-163.28627617, 68.65911367],
        [-162.24943920, 69.13594698], [-156.34578665, 67.67279153],
         [-151.20377421, 60.73384959], [-150.67175906, 60.04628798], [-157.73233549, 68.15037569],
        [-156.28640511, 70.27669083], [-147.55250323, 69.12119637],
        [-143.77940611, 69.13717664], [-143.60888602, 69.79829215], [-153.06530376, 70.50435315],
        [-147.02034702, 68.58702006],  [-157.93985569, 69.16732504],
        [-156.55042833, 66.33659274], [-162.97367991, 68.26540917], [-161.23434328, 65.2680856],
        [-150.86761887, 65.78373252], [-156.36380702, 62.78914008]]
    data_process.ascat_onset_map(['AS'], odd_point=tp)
    data_process.ascat_onset_map(['AS'], product='npr', odd_point=tp, mask=True, version='new')
    quit0()
    for m in ['area_5']:
        print 'test series of pixel in %s' % m
        for ob in ['AS']:
            for mo in ['_norm_']:
                print 'mode is %s, orbit is %s' % (mo, ob)
                data_process.ascat_result_test(m, mode=mo, key=ob, odd_rc=(89, 190))
    quit0()
    call_data_process()

    # 0506/2018
    site_nos_new = ['948', '957', '958', '963', '2080', '947', '949', '950',
                '960', '962', '967', '968', '1090', '1175', '1177', '1233',
                '2065', '2081', '2210', '2211', '2212', '2213']
    for sno in site_nos_new:
    #     # s_info = site_infos.change_site(sno)
    #     # s_info_t = np.array([s_info[2], s_info[1], int(s_info[0])]).reshape(3, -1)
    #     # print sno
    #     # spt_quick.ascat_sub_pixel(s_info_t, dis0=4, site_nos=[sno])
    #     # after shape file
        gdal_clips(sno, ipt='lc')
        gdal_clips(sno, ipt='snowf')
        ascat_snow_lc(sno)
        check_lc_snow_ascat(sno)

    ascat_snowlc_npy(site_nos_new)
    rid = 2
    sno_sp = regions_extract(rid)
    beams = 'fore'

    land_cover(rid, sno_sp, ['ever', 'decid', 'shrub'], False)  # 2nd var is a list of sites
    check_pass_time(rid)
    regions_plotting(region_id=rid, att_xyz=['tair', beams, 'ID'])
    regions_plotting(region_id=rid, att_xyz=['tair', beams, '# date'])
    regions_plotting(region_id=rid, att_xyz=['tair', beams, 'snowf'])
    regions_plotting(region_id=rid, att_xyz=['tair', beams, 'swe'])
    regions_plotting(region_id=rid, att_xyz=['tair', beams, 'ID'])
    regions_plotting(region_id=rid, att_xyz=['tair', beams, 'dswe'])
    regions_plotting(region_id=rid, att_xyz=['dswe', beams, 'swe'])
    for sno in sno_sp:
        #regions_plotting(sno)
        regions_plotting(region_id=rid, site_no=sno, att_xyz=['tair', beams, '# date'])  # xlim0=[1, 366]
        regions_plotting(region_id=rid, site_no=sno, att_xyz=['# date', beams, 'swe'], xlim0=[1, 366])
        # regions_plotting(region_id=rid, site_no=sno, att_xyz=['dswe', beams, 'swe'])
    # bxy.test_read_txt()
    quit0()

    sno_sp = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    sno_sp_region = ['947', '949', '950', '960', '1090']
    for sno in sno_sp:
        # ascat_snow_lc(sno)
        # gdal_clips(sno, ipt='snowf')
        # gdal_clips(sno, ipt='lc')
        # ascat_snow_lc(sno)
        check_lc_snow_ascat(sno)
    quit0()
    for sno in sno_sp:
        gdal_clips(sno, ipt='lc')
    quit0()
    for sno in sno_sp:
        s_info = site_infos.change_site(sno)
        s_info_t = np.array([s_info[2], s_info[1], int(s_info[0])]).reshape(3, -1)
        print sno
        spt_quick.ascat_sub_pixel(s_info_t, dis0=4, site_nos=[sno])
        # get_corner('ascat_20160409_0509_947_000.txt')
        # check_ascat_sub('test_ascat_sub_pixel.npy', 'test_ascat_sub_corner.npy')
    quit0()
    for dstr in ['20160101', '20160102', '20160103', '20160104', '20160105', '20160106', '20160107']:
        npyname = 'result_08_01/area/ascat/ascat_%s_alaska.npy' % dstr
        ascat_fp_ak(npyname, dtr=dstr)
    quit0()
    site_nos = ['947', '968']
    sub_no = 1
    for sn in site_nos:
        onset_01, dist = smap_ft_compare(sn, period=['all', 0, 365], orb=1, subplot_id=sub_no)
        # change sub_no to draw two rows
        if sub_no > 1:
            sub_no = 1
        else:
            sub_no = 2
    quit0()

    # plot_snow_depth()
    discuss_combining()
    # thawing_snow_depth()
    # check_distance()
    quit0()
    x = 4*np.pi/0.21
    real0 = 25
    img0 = 3
    im_z = bxy.im_permitivity(real0, img0)
    print x, im_z
    print 1/(x*im_z)
    quit0()
    site_nos = ['962', '947', '968']
    Read_radar.read_amsr2(['962', '968'], ['2016.06.01', '2016.06.10'])
    data_process.check_amsr2_result0('20160601', '947', 'D')
    quit0()
    # for site_no in site_nos:
    #     Read_radar.amsr2_series(site_no,
    #                             ['Brightness Temperature (res06,6.9GHz,V)', 'Brightness Temperature (res23,18.7GHz,V)',
    #                              'Brightness Temperature (res23,36.5GHz,V)', 'Earth Azimuth',
    #                              'Brightness Temperature (res06,6.9GHz,H)', 'Brightness Temperature (res23,18.7GHz,H)',
    #                              'Brightness Temperature (res23,36.5GHz,H)'], orbit='D')
    for sno in site_nos:
        amsr2_plot(sno)
        amsr2_plot(sno, pol='H')
        amsr2_plot(sno, orb='D')
        amsr2_plot(sno, orb='D', pol='H')
    # read amsr2 data
    # data_process.check_amsr2()
    # data_process.check_amsr2_result0('20151203', '947', 'D')
    quit0()
    site_nos = ['947', '968']
    sub_no = 1
    for sn in site_nos:
        onset_01, dist = smap_ft_compare(sn, period=['all', 0, 365], orb=1, subplot_id=sub_no)
        # change sub_no to draw two rows
        if sub_no > 1:
            sub_no = 1
        else:
            sub_no = 2
    quit0()

    new_process(['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'])
    quit0()

    site_nos = ['947', '968']
    ft = 'thaw'
    indic_npr, indic_tb, indic_ascat = '%s/npr' % ft, '%s/tb' % ft, '%s/ascat' % ft
    for site_no in site_nos:
        trans2winter_ratio, a_list = data_process.test_winter_trans(site_no, indic=indic_npr, w=[7])
        trans2winter_ratio_tb, b_list = data_process.test_winter_trans(site_no, indic=indic_tb, w=[7])
        trans2winter_ratio_ascat, c_list = data_process.test_winter_trans(site_no, indic=indic_ascat, w=[7])
        print 'the SNR for %s are npr: %.2f, tb: %.2f, and ascat: %.2f' % (site_no, trans2winter_ratio, trans2winter_ratio_tb, trans2winter_ratio_ascat)
        print 'the thaw std for %s are npr: %.4f, tb: %.2f, and ascat: %.2f' % (site_no, a_list[1], b_list[1], c_list[1])
    quit0()
    site_nos = ['947', '949', '950', '968', '962', '2212']
    site_nos = ['967', '2081', '2210']
    site_nos = ['1175', '1177', '1233', '2065', '2213']
    site_nos = ['1090']
    site_nos = ['947', '968']
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    sub_no = 1
    for sn in site_nos:
        onset_01, dist = smap_ft_compare(sn, period=['all', 0, 365], orb=1, subplot_id=sub_no)
        if sub_no > 1:
            sub_no = 1
        else:
            sub_no = 1  # change if want to draw to rows
    # onset_01, dist = smap_ft_compare('947', period=['all', 0, 365], orb=1)
    # onset_01, dist = smap_ft_compare('949', period=['all', 0, 365], orb=1)
    # onset_01, dist = smap_ft_compare('968', period=['all', 0, 365], orb=1)


    # data_process.check_amsr2()
    # Read_radar.read_amsr2(['947', '949', '950', '960', '962', '967', '968', '1090', '1175',
    #                        '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'],
    #                       ['2016.05.01', '2016.12.31'])
    # Read_radar.read_amsr2(['947', '949', '950', '960', '962', '967', '968', '1090', '1175',
    #                        '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'],
    #                       ['2015.12.01', '2016.04.30'])'new_smap_ft_compare_%s' % sno
    # quit0()


    #     # Read_radar.amsr2_series(site_no,
    #     #                         ['Brightness Temperature (res06,6.9GHz,H)', 'Brightness Temperature (res23,18.7GHz,H)',
    #     #                          'Brightness Temperature (res06,36.5GHz,H)'])
    #     amsr2_plot(site_no)


    # data_process.check_amsr2_result0('20151201', '947', 'A')
    # ft_product(orb0=1)
    # ft_product(orb0=0)
    # orbit_compare('thaw')
    # orbit_compare('freeze')
    # data_process.check_station('962', 271.75)
    x = 0

    # site0 = 2210
    # site_list = [site0*100+i for i in range(0, 9)]
    # # spt_quick.ascat_point_plot(center='sub_2210_center.txt', dis0=9)
    # ascat_sub9([site_list], sub_dir='./result_05_01/ascat_point/')

    # x1 = np.load('./result_05_01/ascat_point/ascat_s947_2016.npy')
    # utc_sec = x1[:, 1]
    # orb_info = x1[:, -1]
    # tz_utc = pytz.timezone('utc')
    # tz_ak = pytz.timezone('US/Alaska')
    # as_ind, des_ind = orb_info < 0.5, orb_info > 0.5
    # passtime_obj_list = [datetime(2000, 1, 1, 0, 0, tzinfo=tz_utc)+timedelta(seconds=sec_i) for sec_i in utc_sec[des_ind]]
    # i = [1, 3, 5, 10, 15, 25]
    # for i0 in i:
    #     pass0 = passtime_obj_list[i0].astimezone(tz=tz_ak)
    #     print pass0.timetuple()

   # test_method('thaw')
   # test_method('freeze')
   # ascat_within_tb(disref=[9, 5], subpixel=True)