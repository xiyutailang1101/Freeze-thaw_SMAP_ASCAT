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
import temp_test


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
                                               txt_path=ascat_series)  # 0 for ascending
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


def ak_series(doy_array, att_list=['cell_tb_v_aft', 'cell_tb_h_aft'],
              ascat_atts=['sigma', 'incidence', 'pass_utc'], orbit='A', ascat_format='npy', smap_format='npy',
              ascat=False, smap=False):
    """
    :param doy_array:  current is not usable
    :param att_list:
    :param ascat_atts:
    :param orbit:
    :param ascat_format:
    :return:
    """
    # 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5'
    # read asacat
    if ascat is not False:
        if len(ascat_atts) > 0:
            if ascat_format == 'h5':
                new_h5_name = 'result_08_01/area/combine_result/ascat_2016_3d_%s.h5' \
                              % 'all'  # new a h5 file or use the old version
                h50 = h5py.File(new_h5_name, 'a')
                match_list = []  # a list for matching the re-sampled data such as re-sampled sigma
                match0 = 'result_08_01/ascat_resample_all/ascat_*.h5'
                resample_list = sorted(glob.glob(match0))
                # result_08_01/ascat_resample_all/ascat_20151101_10_A.h5
                date_list = [path0.split('/')[-1] for path0 in resample_list]
                ymd_list = [f0.split('_')[2] for f0 in date_list]
                hh_list = [f1.split('_')[3] for f1 in date_list]
                pass_secs_all = np.zeros(len(hh_list))
                i_sec = 0
                satellite_type = site_infos.get_satellite_type()
                sate_orbit = np.array([satellite_type.index(path0.split('_')[1]+'_'+path0.split('_')[4])
                             for path0 in date_list])
                for ymd0, hh0 in zip(ymd_list, hh_list):
                    secs0 = bxy.get_secs([int(ymd0[0: 4]), int(ymd0[4: 6]), int(ymd0[6: 8]), int(hh0), 0, 0])
                    pass_secs_all[i_sec] = secs0
                    i_sec += 1
                i_arg = np.argsort(pass_secs_all)
                resample_list_sort = [resample_list[i_arg0] for i_arg0 in i_arg]
                np.savetxt('ak_series_h5_test.txt', resample_list_sort, delimiter=',', fmt='%s')

                ascat_dict = {} # initial
                for att1 in ascat_atts:  # consider how to add more data
                    ascat_dict[att1] = np.zeros([300, 300, len(resample_list)])-999
                for i_date in range(0, len(resample_list)):  # read resample
                    h5_0 = h5py.File(resample_list_sort[i_date], 'r')
                    for att2 in ascat_atts:
                        ascat_dict[att2][:, :, i_date] = h5_0[att2]
                    h5_0.close()
                # write data to the 3d h5 file
                if 'sate_orbit' not in h50.keys():
                    h50['sate_orbit'] = sate_orbit
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
            elif ascat_format == 'npy':
                match_list ={}  # a list for matching the re-sampled data such as re-sampled sigma
                for att0 in ascat_atts:
                    match0 = 'result_08_01/ascat_resample_npy/ascat_*%s*.npy' % att0
                    att0_list = sorted(glob.glob(match0))
                    match_list[att0] = att0_list
                ascat_dict = {} # initial
                for att1 in ascat_atts:
                    ascat_dict[att1] = np.zeros([300, 300, len(att0_list)])-999
                for i_date in range(0, len(att0_list)):  # read resample
                    tp_value = np.load(match_list['resample'][i_date])
                    ascat_dict['resample'][:, :, i_date] = tp_value  # resample value
                    tp_value2 = np.load(match_list['incidence'][i_date])
                    ascat_dict['incidence'][:, :, i_date] = tp_value2
                    tp_value3 = np.load(match_list['pass_utc'][i_date])
                    ascat_dict['pass_utc'][:, :, i_date] = tp_value3
    ################################################
    # the smap data, applied
    if smap is not False:
        # # initial
        # read smap
        date_str = []
        for doy0 in doy_array:
            date_str0 = bxy.doy2date(doy0, fmt='%Y%m%d')
            date_str.append(date_str0)
        h5_path = 'result_08_01/area/smap_area_result'
        # get h5_list
        h5_list = []
        for d0 in date_str:
            match_name = '%s/SMAP_alaska_%s_GRID_%s.h5' % (h5_path, orbit, d0)
            f0 = glob.glob(match_name)  # all single orbit
            h5_list.append(f0)
        h5_list = filter(None, h5_list)
        if len(h5_list) < 1:
            print 'no Gridded h5 data was found'
            return 0
        att_value_all = {}
        for att0 in att_list:
            att_value_all[att0] = np.zeros([90, 100, len(h5_list)])-999
        i_date = 0
        nodata_id = 0
        if len(att_list)>0:
            for i_date, resample0_path in enumerate(h5_list):
                # h5_fname = 'SMAP_alaska_A_GRID_%s.h5' % resample0_path
                # if h5_fname not in h5_list:
                #     print 'no data on %s' % resample0_path
                #     with open('smap_series_no_data.out', 'a') as writer1:
                #         if nodata_id == 0:  # add a line of current time
                #             time0 = datetime.now().timetuple()
                #             time_str = '%d-%d, %d: %d \n' % (time0.tm_mon, time0.tm_mday, time0.tm_hour, time0.tm_min)
                #             writer1.write(time_str)
                #             writer1.write(resample0_path)
                #             writer1.write('\n')
                #             nodata_id += 1
                #         else:
                #             writer1.write(resample0_path)
                #             writer1.write('\n')
                #     i_date += 1
                #     continue
                # else:
                # h0 = h5py.File(h5_path+'/'+h5_fname)

                # retrieve the time_str
                # str0 = resample0_path[0].split('/')[-1].split('.')[0].split('_')[-1]
                h0 = h5py.File(resample0_path[0])
                for att0 in att_list:
                    if (h0[att0].value==0).any():
                        pause=0
                    att_value_all[att0][:, :, i_date] = h0[att0].value
                h0.close()
                        # tbv_a_ak_series[i_date], tbh_a_ak_series[i_date] = h0['cell_tb_v_aft'].value, h0['cell_tb_h_aft'].value
            if smap_format == 'npy':
                for att0 in att_list:
                    save_name = 'result_08_01/area/combine_result/smap_%s_%s.npy' % (att0, orbit)
                    np.save(save_name, att_value_all[att0])
            elif smap_format == 'h5':
                new_name = 'result_08_01/area/combine_result/smap_2016_%s_3d.h5' % orbit
                new_h5 = h5py.File(new_name, 'a')
                for att0 in att_list:
                    print att0
                    print att_value_all[att0].size
                    new_h5[att0] = att_value_all[att0]




        # np.savetxt('result_08_01/area/combine_result/ascat_smap_doy.txt', doy0, fmt='%d', delimiter=',')
    return 0
    # onset


def combining2(doy_array, y=2016, smap_atts=['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft'],
              ascat_atts=['sigma', 'incidence', 'pass_utc'], orbit='A', ascat_format='npy', smap_format='npy',
              ascat=False, smap=False, year=2016, onset_save=False, pixel_plot=False,
               land_id=np.array([3770, 4648, 5356]), id_name=np.array([3770, 4648, 5356])):
    '''
    :param doy_array: the time period, in unit of days
    :param y: =2016, the based year
    :param smap_atts:
    :param ascat_atts: specify the attributes same as that in spt_quick.ascat_area_plot2
    :param orbit:
    :param ascat_format:
    :param smap_format:
    :param ascat:
    :param smap:
    :param year:
    :param onset_save: save onset for the regional results
    :param pixel_plot: plot the time-series at given land_id, the time series at station will be saved
    :param land_id: 1d indices in smap grid 36 km
    :param id_name: 1d indices corresponded to SNOTEL stations
    :return: onset_save is False: the time series of indicators, see line 2644
    '''
    # some initial parameters
    melt_onset0 = -1
    ini_secs = bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12])
    end_secs = bxy.get_total_sec('%d1231' % y, reftime=[2000, 1, 1, 12])
    thaw_window = [ini_secs + doy0*3600*24 for doy0 in [60, 150]]
    freeze_window = [ini_secs + doy0*3600*24 for doy0 in [240, 340]]
    secs_winter = [bxy.get_total_sec(str0) for str0 in ['%d0101' % y, '%d0315' % y]]
    row_table = np.loadtxt('ascat_row_table.txt', delimiter=',')  # row table of 12.5 grid to 36 grid
    col_table = np.loadtxt('ascat_col_table.txt', delimiter=',')
    path_ascat = []
    if onset_save == True:
        # mask the ocean
        mask = np.load(('./result_05_01/other_product/mask_ease2_360N.npy'))
        mask_1d = mask.reshape(1, -1)[0]
        land_id = np.where(mask_1d != 0)[0]

    for doy0 in doy_array:
        time_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=y)
        match_name = 'result_08_01/ascat_resample_all/ascat_*%s*.h5' % time_str0
        path_ascat += glob.glob(match_name)
    lon_samp, lat_smap = data_process.get_base('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151102.h5',
                                           'cell_lon', 'cell_lat')  # lat/lon grid of alaska
    lons_125, lats_125 = data_process.get_base('result_08_01/ascat_resample_all/ascat_metopB_20160101_11_A.h5',
                                               'longitude', 'latitude')
    lons_1d = lon_samp.ravel()
    lats_1d = lat_smap.ravel()

    # read ascat_data into dictionary. each key0 corresponded to the key0 of the ascat h5 file (300, 300, time)
    start0 = bxy.get_time_now()
    ascat_dict = data_process.ascat_alaska_grid(ascat_atts, path_ascat)  # keys contain 'sate_type'
    start1 = bxy.get_time_now()
    print("----read ascat part: %s seconds ---" % (start1-start0))
    # read smap data into dictionary, data shape: 2d, (grid_index, time_unit)
    date_str = []  # period of interest
    for doy0 in doy_array:
        date_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=y)
        date_str.append(date_str0)
    start0 = bxy.get_time_now()
    smap_dict = data_process.smap_alaska_grid(date_str, smap_atts, 'A', lon_samp.size)
    smap_dict_d = data_process.smap_alaska_grid(date_str, smap_atts, 'D', lon_samp.size)
    start1 = bxy.get_time_now()
    print("----read smap part: %s seconds ---" % (start1-start0))
    l = 2
    smap_thaw_a, smap_thaw_d, ascat_thaw_360, ascat_melt_360, \
    ascat_thaw_125, ascat_melt_125 =np.zeros([l, lons_1d.size]), np.zeros([l, lons_1d.size]), np.zeros([l, lons_1d.size]), \
                                    np.zeros([l, lons_1d.size]), \
                                    np.zeros([l, 300, 300]), np.zeros([l, 300, 300])
    ascat_melt_t2, smap_thaw_t2, smap_tb_t3 = ascat_melt_360.copy(), ascat_melt_360.copy(), ascat_melt_360.copy()
    smap_thaw_t3 = ascat_melt_360.copy()
    smap_thaw_obdv, smap_thaw_obdh = np.zeros([l, lons_1d.size]), np.zeros([l, lons_1d.size])  # zero value for saving results
    table_result = []
    for i1 in land_id:
        lon0, lat0 = lons_1d[i1], lats_1d[i1]
        print 'the pixel of interest is %d, coord: %.3f, %.3f' % (i1, lat0, lon0)
        # smap pixel-based, default keys: ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft']
        smap_pixel = {}
        smap_pixel_d = {}
        for att0 in smap_atts:
            smap_pixel[att0] = smap_dict[att0][i1]
            smap_pixel_d[att0] = smap_dict_d[att0][i1]
        x_time, diff_tbv, diff_tbh = data_process.smap_extract_ad(y, smap_pixel, smap_pixel_d)
        npr, npr_d = data_process.grid_extract(smap_pixel), data_process.grid_extract(smap_pixel_d)  # npr
        tb_a, tb_d = data_process.grid_extract(smap_pixel, key='tb'), data_process.grid_extract(smap_pixel_d, key='tb')
        t_npr, t_npr_d = smap_pixel['cell_tb_time_seconds_aft'], \
                         smap_pixel_d['cell_tb_time_seconds_aft']
        # edge detection smap
        k0 = 5
        conv_obdh, thaw_secs_obdh = data_process.get_onset(x_time, diff_tbh, year0=y,  # horizontal difference
                                                            thaw_window=thaw_window, freeze_window=freeze_window)
        conv_obdv, thaw_secs_obdv = data_process.get_onset(x_time, diff_tbv, year0=y,  # vertical difference
                                                            thaw_window=thaw_window, freeze_window=freeze_window)
        # conv_npr, thaw_secs_npr = data_process.get_onset(t_npr, npr, thaw_window=thaw_window, k=7,
        #                                                  freeze_window=freeze_window)  # thaw_secs_npr, thaw/melt initial
        conv_npr_d, thaw_secs_npr_d = data_process.get_onset(t_npr_d, npr_d, year0=y, thaw_window=thaw_window, k=7,
                                                             freeze_window=freeze_window)
        conv_npr, thaw_secs_npr = data_process.get_onset(t_npr, npr, year0=y,   # npr_a_t1
                                   thaw_window=[bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=k0, type='npr')  # npr asc up
        npr_a_t1 = thaw_secs_npr
        conv_npr_a = conv_npr
        npr_a_t2_0_x = data_process.get_onset_zero_x(conv_npr_a, npr_a_t1, zero_x=5e-3)
        conv_tb_a, tb_a_onset2 = \
            data_process.get_onset(t_npr, tb_a[0], year0=y,
                                   thaw_window=[bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=k0, type='tb')  # tb asc down
        # tb_a_t3_0_cross = conv_tb_a[0][(conv_tb_a[0]>tb_a_onset2) & (conv_tb_a[1] > -0.5)][0]
        tb_a_t3_0_cross = data_process.get_onset_zero_x(conv_tb_a, tb_a_onset2, zero_x=-0.01)
        # find peak smap
        smap_thaw_a[0, i1], smap_thaw_d[0, i1] = thaw_secs_npr, thaw_secs_npr_d
        smap_thaw_obdv[0, i1], smap_thaw_obdh[0, i1] = thaw_secs_obdv, thaw_secs_obdh
        smap_thaw_t2[0, i1] = npr_a_t2_0_x
        smap_thaw_t3[0, i1] = tb_a_t3_0_cross
        # plot smap if applicable
        if pixel_plot:
            plot_funcs.plot_subplot([np.array([x_time, diff_tbh]), np.array([x_time, diff_tbv]),
                                     np.array([t_npr, npr]), np.array([t_npr_d, npr_d])],
                                    [conv_obdh, conv_obdv, conv_npr, conv_npr_d],
                                    main_label=['obdh', 'obdv', 'npr_a', 'npr_d'],
                                    x_unit='doy', x_lim=[ini_secs, end_secs],
                                    figname='result_08_01/20181202plot_smap_%d.png' % i1)

        # based on ascat
        start0 = bxy.get_time_now()
        ascat_pixel = {}
        row_no, col_no = row_table[i1].astype(int), col_table[i1].astype(int)  # index connect smap and ascat (9) pixels
        for att0 in ascat_atts:  # each smap pixel crresponded to 9 NN ascat pixels
            ascat_pixel[att0] = ascat_dict[att0][row_no, col_no, :]  # shape: 9 * time
        ascat_pixel['sate_type'] = ascat_dict['sate_type']
        index_valid_0 = (ascat_pixel[ascat_atts[1]]) > 0 & \
                        (ascat_pixel[ascat_atts[1]] < 90) & \
                        (ascat_pixel[ascat_atts[0]] != -999) & \
                        (ascat_pixel[ascat_atts[0]] < -0.1)   # set masks. 0: sigma value, 1: incidence angle
        lat_9 = lats_125[row_no, col_no]
        lon_9 = lons_125[row_no, col_no]
        dis_9 = bxy.cal_dis(lat0, lon0, lat_9, lon_9)   # calculate distance
        i_nn = np.argsort(dis_9)
        # read nearest measurements, spatially mean from 12.5 to 36,
        for key0 in ascat_atts:
            ascat_pixel[key0][~index_valid_0] = np.nan
            new_key = '%s_9' % key0
            for nn0 in i_nn:
                ascat_pixel['%s_%d' % (key0, nn0)] = ascat_pixel[key0][nn0]
            ascat_pixel[new_key] = np.nanmean(ascat_pixel[key0][0:3], axis=0)  # mean value of the nearest 3 pixels
        if np.isnan(ascat_pixel[ascat_atts[0]+'_9']).size > 1e4:
            continue  # not enough valid measurements
        else:
            # angular dependency based on satellite type
            i_mean_valid = ~np.isnan(ascat_pixel[ascat_atts[0]+'_9'])
            for type0 in [0., 1., 2., 3.]:
                i_type = (ascat_dict['sate_type'] == type0) & i_mean_valid
                inc0, sigma0 = ascat_pixel[ascat_atts[1]+'_9'][i_type], ascat_pixel[ascat_atts[0]+'_9'][i_type]
                if inc0.size < 20:
                    a0 = -0.12
                    print 'pixel %d has too less type %d' % (i1, type0)
                else:
                    a0, b0 = np.polyfit(inc0, sigma0, 1)
                for nn0 in range(0, 10):
                    key00 = '%s_%d' % (ascat_atts[0], nn0)
                    key01 = '%s_%d' % (ascat_atts[1], nn0)
                    ascat_pixel[key00][i_type] -= (ascat_pixel[key01][i_type]-40)*a0
                    # if ascat_pixel['sigma0_trip_aft'][0][~np.isnan(ascat_pixel['sigma0_trip_aft'][0])].max()>0:
                    #     pause = 0
            # np.savez('ascat_nine_2016_%d.npz' % i1, **ascat_pixel)

            # edge detection ascat, thaw and melt
            # the index of ascat_grid, thaw detect and melt detect, assign value
            for nn0 in [9]:  # range(0, 10):
                # print nn0, i1
                key00 = '%s_%d' % (ascat_atts[0], nn0)  # sigma
                key01 = '%s_%d' % (ascat_atts[2], nn0)  # utc secs
                if nn0 == 2:
                    pause = 0
                if sum(~np.isnan(ascat_pixel[key00])) < doy_array.size*150/365.0:
                    print "too much nan value in pixel %d, NN: %d" % (nn0, i1)
                    print "this pixel could be out of boundary"
                    if nn0 < 9:
                        ascat_thaw_125[0, row_no[i_nn[nn0]], col_no[i_nn[nn0]]] = 0
                        ascat_melt_125[0, row_no[i_nn[nn0]], col_no[i_nn[nn0]]] = 0
                        ascat_melt_125[1, row_no[i_nn[nn0]], col_no[i_nn[nn0]]] = 0
                    else:
                        ascat_thaw_360[0, i1], ascat_melt_360[0, i1] = 0, 0
                        ascat_melt_360[1, i1] = 0
                    continue
                max_value_ascat, min_value_ascat, conv_ascat \
                    = test_def.edge_detect(ascat_pixel[key01], ascat_pixel[key00], 7,
                                           seriestype='npr', is_sort=False)
                max_value_thaw = max_value_ascat[(max_value_ascat[:, 1] > thaw_window[0]) &
                                                 (max_value_ascat[:, 1] < thaw_window[1])]
                min_value_freeze = min_value_ascat[(min_value_ascat[:, 1] > freeze_window[0]) &
                                                   (min_value_ascat[:, 1] < freeze_window[1])]
                if max_value_thaw.size == 0:  # check positions where onsets doesn't exist.
                    with open('onset_map0.txt', 'a-') as writer0:
                        writer0.writelines('no thaw onset was find at: %d' % 0)
                else:  #  detect melt event
                    thaw_onset0 = max_value_thaw[:, 1][max_value_thaw[:, -1].argmax()]
                    melt_zone0 = [thaw_secs_npr, thaw_onset0]  # the meltzone buffer
                    min_detect_snowmelt = min_value_ascat[(min_value_ascat[:, 1] > melt_zone0[0]) &
                                                     (min_value_ascat[:, 1] < melt_zone0[1])]
                    min_detect_winter = min_value_ascat[(min_value_ascat[:, 1] > secs_winter[0]) &
                                                   (min_value_ascat[:, 1] < secs_winter[1])]
                    min_conv_winter_mean = np.nanmean(min_detect_winter[:, -1])
                    if min_detect_snowmelt[:, -1].size < 1:
                        melt_onset0 = 0
                        melt_conv = -999
                        melt_lvl0 = 0
                    else:
                        # consider the significant snow melt event
                        levels = np.abs(min_detect_snowmelt[:, -1]/min_conv_winter_mean)
                        valid_index_melt = levels>3.8
                        if sum(valid_index_melt) > 0:
                            # if sum(valid_index_melt) > 1:
                            melt_onset0 = min_detect_snowmelt[:, 1][valid_index_melt][0]
                            # else:
                            #     melt_onset0 = min_sec_snowmelt[:, 1][valid_index_melt][0]
                            melt_lvl0 = levels[valid_index_melt][0]
                        else:
                            melt_onset0 = min_detect_snowmelt[:, 1][min_detect_snowmelt[:, -1].argmin()]
                            melt_lvl0 = levels[min_detect_snowmelt[:, -1].argmin()]

                        melt_conv = min_detect_snowmelt[:, -1][min_detect_snowmelt[:, -1].argmin()]

                # sigma based results new, compacted
                '''
                results tables:
                smap_thaw_a/d: t1;     ascat_melt_360: t1
                smap_thaw_npr_max: t2; ascat_melt_sigma_min: t2
                smap_thaw_tb_min: t3;  ascat_thaw_360: t3
                '''
                # t1, t2, t3 based on sigma
                # try both orbits, try asc, then des, till the results are significant
                th_win_new = [bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12]) +
                              doy0*3600*24 for doy0 in [60, 150]]
                conv_s, s_t3_up, s_melt_t1 = data_process.get_onset_new(ascat_pixel[key01], ascat_pixel[key00], year0=y,
                                            thaw_window=th_win_new, k=7, type='npr', melt_window=npr_a_t1, mode=2)
                # s_melt_t1: [melt_onset0, level, conv_edge_onset, conv_winter_mean, conv_winter_std]
                sig_t2_0_x = data_process.get_onset_zero_x(conv_s, s_melt_t1[0], zero_x=-0.5e-2)
                if (nn0 != 0) & (nn0 < 9):  # not NN or mean sigma
                    ascat_thaw_125[0, row_no[i_nn[nn0]], col_no[i_nn[nn0]]] = thaw_onset0
                    ascat_melt_125[0, row_no[i_nn[nn0]], col_no[i_nn[nn0]]] = melt_onset0
                    ascat_melt_125[1, row_no[i_nn[nn0]], col_no[i_nn[nn0]]] = melt_lvl0
                else:  # the NN or nearest
                    # ascat_thaw_360[0, i1], ascat_melt_360[0, i1] = thaw_onset0, melt_onset0
                    # ascat_melt_360[1, i1] = melt_lvl0
                    ascat_thaw_360[0, i1], ascat_melt_360[0, i1] = s_t3_up, s_melt_t1[0]  # t3
                    ascat_melt_t2[0, i1] = sig_t2_0_x  # t2
                    ascat_melt_360[1, i1] = s_melt_t1[1]  # sigma t1 level

                    # plot pixel of interest
                    if pixel_plot:
                        key_inc = '%s_%d' % (ascat_atts[1], nn0)
                        # plotting code 1
                        # plot_funcs.plot_subplot([[t_npr, npr],
                        #                          [ascat_pixel[key01], ascat_pixel[key00]],
                        #                          [ascat_pixel[key01], ascat_pixel[key_inc]]],
                        #                         [conv_npr, conv_ascat],
                        #                         main_label=['npr', 'sigma0', 'inc'],
                        #                         vline=[[thaw_secs_npr, melt_onset0], ['k-', 'r-']],
                        #                         x_unit='doy', x_lim=[ini_secs, end_secs],
                        #                         annote=[1, 'winter: %.2f, edge: %.2f'
                        #                                 % (min_conv_winter_mean, melt_lvl0)],
                        #                         figname='result_08_01/20181202plot_ascat_%d_%d.png' % (i1, nn0))
                        plot_funcs.plot_subplot([np.array([t_npr, tb_a[0], tb_a[1]]),
                             np.array([t_npr_d, tb_d[0], tb_d[1]]),
                              np.array([ascat_pixel[key01], ascat_pixel[key00]])],
                            [conv_tb_a, conv_tb_a, conv_ascat],
                            main_label=['tb PM ($K$)', 'tb AM (K)$', '$\sigma^0$ (dB)'],
                            vline=[[tb_a_onset2, tb_a_t3_0_cross, s_t3_up],
                                   ['k-', 'b-', 'r-'], ['tb_down', 'tb_min', 'sig_up']],  # tb down, tb min, sigma up
                            x_unit='sec', x_lim=[ini_secs, end_secs],
                            figname='result_08_01/20181202plot_ascat_%d_%d_tb.png' % (id_name[land_id==i1][0], i1))

                        if i1 in land_id:  # save data used for drawing test
                            np.save('x.npy', np.array([t_npr, ascat_pixel[key01], conv_npr[0], conv_ascat[0]]))
                            txtname = ['npr_A', 'npr_D', 'ascat', 'conv_npr_A', 'conv_npr_D', 'conv_ascat', 'vline', 'lim',
                                       'tb_A', 'tb_D', 'sate_type', 'ascat_angle']
                            i_draw=0
                            for xy00 in [[t_npr, npr], [t_npr_d, npr_d],
                                         [ascat_pixel['%s_9' % ascat_atts[2]], ascat_pixel['%s_9' % ascat_atts[0]]],  # '%s_%d'
                                         conv_npr, conv_npr_d, conv_ascat,
                                         [thaw_secs_npr, thaw_secs_npr_d, melt_onset0],
                                         [ini_secs, end_secs],
                                         [t_npr, tb_a[0], tb_a[1]], [t_npr_d, tb_d[0], tb_d[1]],
                                         ascat_pixel['sate_type'], ascat_pixel['%s_9' % ascat_atts[1]]]:
                                print 'temp file saved ', txtname[i_draw]
                                np.save('result_agu/npy_station_plot/%s_%d_%d_%d.npy'
                                        % (txtname[i_draw], i1, id_name[land_id==i1][0], y), xy00)
                                i_draw+=1
                        conv_npr[1]*=100
                        conv_npr_d[1]*=100

                        # plotting code 2
                        text_example = 'winter: %.3f $\pm$ %.3f, edge: %.3f, level: %.3f' \
                                    % (s_melt_t1[3], s_melt_t1[4], s_melt_t1[2], s_melt_t1[1])
                        plot_funcs.plot_subplot([np.array([t_npr, npr*100]),
                                                 np.array([t_npr_d, npr_d*100]),
                                                 np.array([ascat_pixel[key01], ascat_pixel[key00]])],
                                                [conv_npr, conv_npr_d, conv_ascat],
                                                main_label=['NPR PM ($10^{-2})$', 'NPR AM$',
                                                            '$\sigma^0$ (dB)'],
                                                vline=[[npr_a_t1, s_melt_t1[0], npr_a_t2_0_x, sig_t2_0_x],
                                                ['k-', 'r-', 'k:', 'r:'], ['npr_up', 'sig_down', 'npr_max', 'sig_min']],
                                                x_unit='doy', x_lim=[ini_secs, end_secs],
                                                y_lim=[[2], [[-15, -8]]],
                                                y_lim2=[[0, 1, 2], [[-2, 2], [-2, 2], [-2, 2]]],
                                                figname='result_08_01/20181202plot_ascat_%d_%d.png' % (id_name[land_id==i1][0], i1),
                                                annote=[-1, text_example])

        if onset_save == False:
            onsets_doy = bxy.time_getlocaltime([npr_a_t1, s_melt_t1[0], npr_a_t2_0_x, sig_t2_0_x, tb_a_t3_0_cross,
                                                s_t3_up])[3]
            temp_list = [id_name[land_id==i1][0]]
            for item0 in onsets_doy:
                temp_list.append(item0)
            temp_list.append(s_melt_t1[1])
            print 'the onsets are', temp_list
            table_result.append(temp_list)

    if onset_save ==True:
        save0 = 0
        # smap_thaw_a, smap_thaw_d, ascat_thaw_360, ascat_melt_360, \
        # ascat_thaw_125, ascat_melt_125 =np.zeros([l, lons_1d.size]), np.zeros([l, lons_1d.size]), np.zeros([l, lons_1d.size]), \
        #                             np.zeros([l, lons_1d.size]), \
        #                             np.zeros([l, 300, 300]), np.zeros([l, 300, 300])
        # smap_thaw_obdv, smap_thaw_obdh = np.zeros([l, lons_1d.size]), np.zeros([l, lons_1d.size])
        '''
        results tables:
        smap_thaw_a/d: t1 check;     ascat_melt_360: t1 check
        smap_thaw_npr_max: t2 check; ascat_melt_sigma_min: t2 check
        smap_thaw_tb_min: t3 check;  ascat_thaw_360: t3 check
        '''
        np.savez('20181104_result.npz', smap_thaw_a=smap_thaw_a, smap_thaw_d=smap_thaw_d, ascat_melt_2=ascat_melt_t2,
                 smap_thaw_2=smap_thaw_t2, smap_thaw_3=smap_thaw_t3,
                 ascat_thaw_360=ascat_thaw_360, ascat_melt_360=ascat_melt_360, ascat_thaw_125=ascat_thaw_125,
                 ascat_melt_125=ascat_melt_125, smap_thaw_obdv=smap_thaw_obdv, smap_thaw_obdh=smap_thaw_obdh)


    else:
        np.savetxt('result_08_01/table_result.txt', table_result,
           delimiter=',', header='station,t1,t1_b,t2,t2_b,t3,t3_b,t1_level',
           fmt='%d, %d, %d, %d, %d, %d, %d, %.3f')
    #     # test plot the results
        #     # smap_thaw_a, smap_thaw_d, ascat_thaw_360, ascat_melt_360, smap_thaw_obdv, smap_thaw_obdh,
        #     # ascat_thaw_125, ascat_melt_125
        #     if land_id.size < 50:  # a few odd points, plotting activated
        #         plot_name = ['smap_thaw_a', 'smap_thaw_d', 'ascat_t_360', 'ascat_f_360', 'obdv', 'obdh']
        #         for i0, var0 in enumerate():
        #             aa = 0
        #     # get ad difference of ascat
        #     i_asc, i_des = (ascat_dict['sate_type'] < 2) & (i_mean_valid), \
        #                    (ascat_dict['sate_type'] > 1) & (i_mean_valid)
        #     data_process.cal_ascat_ad(i_asc, i_des, ascat_pixel, ascat_atts, i1)
        #     continue
        #     asc_tp, des_tp = bxy.time_getlocaltime(ascat_pixel[ascat_atts[2]+'_9'][i_asc], ref_time=[2000, 1, 1, 0]), \
        #                      bxy.time_getlocaltime(ascat_pixel[ascat_atts[2]+'_9'][i_des], ref_time=[2000, 1, 1, 0])
        #     ia_2016, id_2016 = asc_tp[0] == 2016, des_tp[0] == 2016
        #
        #     all_tp = bxy.time_getlocaltime(ascat_pixel[ascat_atts[2]+'_9'][i_mean_valid], ref_time=[2000, 1, 1, 0])
        #     np.save('test_pixel_mean', np.array([ascat_dict['sate_type'][i_mean_valid], all_tp]))
        #     np.save('test_asc', np.array([asc_tp[-2][ia_2016], asc_tp[-1][ia_2016]]))
        #     np.save('test_des', np.array([des_tp[-2][id_2016], des_tp[-1][id_2016]]))
        #     print 'file is saved'
        #     same_doy, i_a, i_d = np.intersect1d(asc_tp[-2][ia_2016], des_tp[-2][id_2016], return_indices=True)
        #     sane_d2, i_a2, i_d2 = np.intersect1d(np.flip(asc_tp[-2][ia_2016]),
        #                                          np.flip((des_tp[-2][id_2016])), return_indices=True)
        #     i_a0, i_d0 = -i_a2 + asc_tp[-2][ia_2016].size, -i_d2 + des_tp[-2][id_2016].size
        #
        #     # calculate A/D backscatter, the index: [i_asc][ia_2016][i_a]
        #     n = 2 + (len(ascat_atts)-1)*20
        #     daily_sigma = np.zeros([n, same_doy.size]) - 999.  # secs (ascending), asc_sig, des_sig
        #     for i0, same_doy0 in enumerate(same_doy):
        #         indice_a, indice_d = np.arange(i_a[i0], i_a0[i0]), np.arange(i_d[i0], i_d0[i0])
        #         sec_asc = ascat_pixel[ascat_atts[2]+'_9'][i_asc][ia_2016][indice_a[0]]
        #         sec_des = ascat_pixel[ascat_atts[2]+'_9'][i_des][id_2016][indice_d[-1]]
        #         daily_sigma[0, i0], daily_sigma[n/2, i0] = sec_asc, sec_des
        #         i_tp = 0
        #         for main_att in [ascat_atts[0], ascat_atts[1]]:
        #             for id in range(0, 10):
        #                 key_x = '%s_%d' % (main_att, id)
        #                 daily_sigma[1+id+i_tp, i0] = ascat_pixel[key_x][i_asc][ia_2016][indice_a[0]]
        #                 daily_sigma[n/2+1+id+i_tp, i0] = ascat_pixel[key_x][i_des][id_2016][indice_d[-1]]
        #             i_tp+=10
        #         # sec_asc = dict_pixel[ascat_atts[2]+'_9'][i_asc][ia_2016][indice_a[0]]
        #         # inc_asc = dict_pixel[ascat_atts[1]+'_9'][i_asc][ia_2016][indice_a[0]]
        #         # asc_sig = dict_pixel[ascat_atts[0]+'_9'][i_asc][ia_2016][indice_a[0]]
        #         #
        #         # sec_des = dict_pixel[ascat_atts[2]+'_9'][i_des][id_2016][indice_d[-1]]
        #         # des_sig = dict_pixel[ascat_atts[0]+'_9'][i_des][id_2016][indice_d[-1]]
        #         # inc_des = dict_pixel[ascat_atts[1]+'_9'][i_des][id_2016][indice_d[-1]]
        #         # daily_sigma[:, i0] = np.array([sec_asc, asc_sig, inc_asc, sec_des, des_sig, inc_des])
        #     np.save('test_ascat_obd_36_%d' % (i1), daily_sigma)
        # start1 = bxy.get_time_now()
        # print("----loop alaska part: %s seconds ---" % (start1-start0))
    return 0


def combine_detection(thaw_window, freeze_window,
                      ascat_detect=False, tb_detect=False, npr_detect=True,
                      odd_plot=False, odd_plot_ascat=False, sigma_npr=7, sigma_ascat=3, single_pixel=False,
                      onset_save=False, odd_id=False, melt_zone=45):
    """
    :param thaw_window:
    :param freeze_window:
    :param ascat_detect:
    :param tb_detect:
    :param npr_detect:
    :param odd_plot:
    :param odd_plot_ascat:
    :param sigma_npr:
    :param sigma_ascat:
    :param single_pixel:
    :return: output_combine: a list whose each element is [pixel_index, index of target pixel,
                                                            [t_x],  x_time for each elem. of next list
                                                            [npr, ascat, conv_n, conv_a], indicators and conv. response]

    """
    # some global value
    satellite_type = ['metopA_A', 'metopA_D', 'metopB_A', 'metopB_D']
    smap_tbv = np.load('result_08_01/area/combine_result/smap_cell_tb_v_aft.npy')
    smap_tbh = np.load('result_08_01/area/combine_result/smap_cell_tb_h_aft.npy')
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102'
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.copy().ravel()
    lats_1d = h0['cell_lat'].value.copy().ravel()
    h0.close()
    # updated, read ascat data from h5 files
    ascat_h0 = h5py.File('result_08_01/area/combine_result/ascat_2016_3d_all.h5')
    ascat_sigma = ascat_h0['sigma'].value.copy()
    ascat_incidence = ascat_h0['incidence'].value.copy()
    ascat_pass_utc = ascat_h0['pass_utc'].value.copy()
    ascat_lon = ascat_h0['latitude'].value.copy()
    ascat_lat= ascat_h0['longitude'].value.copy()
    ascat_type = ascat_h0['sate_orbit'].value.copy()
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
    secs_winter = (np.array([0, 61]) - 1) * 3600 * 24 + ini_seconds

    # output initial
    smap_onset0 = np.zeros([smap_tbv.shape[0], smap_tbv.shape[1]])
    nan_out_idx = 0
    # secs_winter = [bxy.get_total_sec(str0) for str0 in ['20160101', '20160315']]
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
    onset_thaw_ascat, onset_melt_ascat, conv_melt_ascat, level_melt_ascat = \
        np.zeros(tbv_2d.shape[0]) - 999, np.zeros(tbv_2d.shape[0]) - 999, np.zeros(tbv_2d.shape[0]) - 999, \
        np.zeros(tbv_2d.shape[0]) - 999
    ascat_winter_mean, ascat_winter_std = np.zeros(tbv_2d.shape[0]) - 999, np.zeros(tbv_2d.shape[0]) - 999
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
    dis_all = np.array([])  # distance check
    output_combine = []  # each one is a specific pixel [pixel_index, [t_x], [npr, ascat, conv_n, conv_a]]
    odd_type = np.dtype({'names': ['p_id', 'melt_onset_a', 'thaw_onset', 'thaw_onset_a', 'mean_a', 'std_a', 'level_a',
                                   'conv_a'],
                         'formats': ['i', 'i', 'i', 'i', 'f', 'f', 'f', 'f']})
    if odd_plot is False:
        mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
        # onset0 = np.ma.masked_array(onset0, mask=[(onset0==0)|(mask==0)])
        mask_1d = mask.reshape(1, -1)[0]
        land_id = np.where(mask_1d != 0)[0]
    elif odd_plot == 'all':
        mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
        # onset0 = np.ma.masked_array(onset0, mask=[(onset0==0)|(mask==0)])
        mask_1d = mask.reshape(1, -1)[0]
        land_id = np.where(mask_1d != 0)[0]
    elif type(odd_plot) is list:
        land_id = odd_plot
        melt_table = np.zeros([len(odd_plot), 10]) - 999
    else:
        land_id = [odd_plot]
        melt_table = np.zeros([len(odd_plot), 10]) - 999
    # start loop at all regions or just stations/pixles
    for i0 in land_id:
        if type(odd_plot) is list:
            odd_check = i0 in odd_plot
        else:
            odd_check = i0 == odd_plot
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
                    thaw_onset0 = max_value_thaw[:, 1][max_value_thaw[:, -1].argmax()]  # thaw onset from npr
                    thaw_onset0_tuple = bxy.time_getlocaltime([thaw_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                    onset_map_0_1d[i0] = thaw_onset0_tuple[-2][0]
                    melt_zone0 = np.array([thaw_onset0-melt_zone*24*3600, thaw_onset0+15*24*3600])  # the meltzone buffer
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
                row_no = row_table[i0].astype(int)
                col_no = col_table[i0].astype(int)
                sigma_series_9 = ascat_sigma[row_no, col_no, :]
                incidence_series_9 = ascat_incidence[row_no, col_table[i0].astype(int), :]
                t_ascat_9 = ascat_pass_utc[row_no, col_table[i0].astype(int), :]
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
                sigma_series_mean = np.nanmean(sigma_series_9, axis=0)
                incidence_series_mean = np.nanmean(incidence_series_9, axis=0)
                t_ascat = np.nanmean(t_ascat_9, axis=0)
                valid_index = (sigma_series_mean > -25) & (sigma_series_mean < -0.1) \
                              & (incidence_series_mean > 31) & (incidence_series_mean < 52)

                if sum(valid_index)<150:
                    # set a unvalid label
                    continue
                else:
                    # check distance
                    dis_2_cent = dis_9.ravel()
                    dis_all = np.concatenate((dis_all, dis_2_cent))
                    # angular dependency for diferent sate tpye
                    series_valid = sigma_series_mean.copy()
                    angulars = np.zeros([4, 3])
                    for type0 in [0, 1, 2, 3]:
                        type_id = (ascat_type == type0) & valid_index
                        inc0, sigma0 = incidence_series_mean[type_id], \
                                       sigma_series_mean[type_id]
                        a0, b0 = np.polyfit(inc0, sigma0, 1)
                        # remove angular dependency separately
                        series_valid[type_id] = sigma_series_mean[type_id] - (incidence_series_mean[type_id]-45)*a0
                        angulars[type0] = np.array([type0, a0, b0])
                        if ~odd_check:
                            plotting = 0  # plot the angular dependency
                            loc0 = bxy.trans_in2d(type0, [2, 2])
                            ax_angle = plt.subplot2grid((2, 2), (loc0))
                            ax_angle.plot(inc0, sigma0, 'k.')  # scatter plot inc vs sigma
                            inc_x = np.arange(20, 60)
                            inc_y = a0*inc_x + b0
                            ax_angle.plot(inc_x, inc_y, 'r-')
                            ax_angle.text(0.02, 0.95, satellite_type[type0], transform=ax_angle.transAxes, va='top', fontsize=16)
                    # check saving the angular coefficients
                    if ~odd_check:
                        fig0 = 'result_08_01/1026/angular_%d.png' % i0
                        plt.savefig(fig0)
                        plt.close()
                        txt_name0 = 'result_08_01/1026/angular_%d.txt' % i0
                        np.savetxt(txt_name0, angulars)
                    # a, b = np.polyfit(incidence_series_mean[valid_index], sigma_series_mean[valid_index], 1)  # angular
                    secs_valid = t_ascat[valid_index]
                    secs_valid -= seconds_2015
                    series_valid = series_valid[valid_index]
                    inc_valid = incidence_series_mean[valid_index]
                    non_outlier = bxy.reject_outliers(series_valid, m=100)  # remove outliers
                    non_outlier = non_outlier & ((inc_valid > 35) & (inc_valid < 50))  # narrow the inc ranges
                    secs_valid = secs_valid[non_outlier]
                    series_valid = series_valid[non_outlier]
                    # daily average if necessary
                    max_value_a, min_value_a, conv_a =\
                        test_def.edge_detect(secs_valid, series_valid, sigma_npr, seriestype='sig')
                    max_value_no_use, min_value_a, conv_a =\
                        test_def.edge_detect(secs_valid, series_valid, sigma_ascat, seriestype='sig')
                    # thaw onset and melt onset
                    max_value_a_thaw = \
                        max_value_a[(max_value_a[:, 1] > thaw_window[0]) & (max_value_a[:, 1] < thaw_window[1])]
                    conv_winter = conv_a[1][(conv_a[0] > secs_winter[0]) & (conv_a[0] < secs_winter[1])]
                    mean0, std0 = np.nanmean(conv_winter), np.nanstd(conv_winter)
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
                    melt_zone0[1] = ascat_thaw_onset0
                    min_value_snowmelt = min_value_a[(min_value_a[:, 1] > melt_zone0[0]) &
                                                     (min_value_a[:, 1] < melt_zone0[1])]
                    min_value_winter = min_value_a[(min_value_a[:, 1] > secs_winter[0]) &
                                                   (min_value_a[:, 1] < secs_winter[1])]
                    min_winter_mean = np.nanmean(min_value_winter[:, -1])  # mean conv when a dropping sigma edge detected
                    if min_value_snowmelt[:, -1].size < 1:
                        melt_onset0 = 0
                        melt_conv = -999
                    else:
                        # consider the significant snow melt event
                        # levels = np.abs(min_value_snowmelt[:, -1] - mean0)/std0
                        levels = np.abs(min_value_snowmelt[:, -1]/min_winter_mean)
                        valid_index_melt = levels>3.8
                        if sum(valid_index_melt) > 0:
                            if sum(valid_index_melt) > 1:
                                melt_onset0 = min_value_snowmelt[:, 1][valid_index_melt][0]
                            else:

                                melt_onset0 = min_value_snowmelt[:, 1][valid_index_melt][0]
                        else:
                            melt_onset0 = min_value_snowmelt[:, 1][min_value_snowmelt[:, -1].argmin()]
                        # melt_onset0 = min_value_snowmelt[:, 1][min_value_snowmelt[:, -1].argmin()]
                        melt_conv = min_value_snowmelt[:, -1][min_value_snowmelt[:, -1].argmin()]
                        # if melt_conv > -0.5:
                        #     melt_onset0 = 0
                        # bxy.time_getlocaltime([melt_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                    pause = 0
                    # secs to day of year
                    thaw_onset0_tuple2 = bxy.time_getlocaltime([ascat_thaw_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                    melt_onset0_tuple2 = bxy.time_getlocaltime([melt_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
                    onset_thaw_ascat[i0] = thaw_onset0_tuple2[-2][0]
                    onset_melt_ascat[i0] = melt_onset0_tuple2[-2][0]
                    conv_melt_ascat[i0] = melt_conv
                    level_melt_ascat[i0] = (melt_conv - mean0)/std0
                    # level_melt_ascat[i0] = np.abs(melt_conv/min_winter_mean)
                    ascat_winter_mean[i0], ascat_winter_std[i0] = mean0, std0
            if odd_check:
                conv_a_winter = conv_a[1][(conv_a[0] > secs_winter[0]) & (conv_a[0] < secs_winter[1])]
                mean0_winter, std_winter = np.nanmean(conv_a_winter), np.nanstd(conv_a_winter)
                # smap
                # ascat
                max_value_thaw_a = max_value_a[(max_value_a[:, 1] > thaw_window[0]) & (max_value_a[:, 1] < thaw_window[1])]
                thaw_index = max_value_thaw_a[:, -1].argmax()
                ascat_thaw_conv0 = max_value_thaw_a[:, 2][thaw_index]
                min_value_snowmelt = min_value_a[(min_value_a[:, 1] > melt_zone0[0]) & (min_value_a[:, 1] < melt_zone0[1])]
                if min_value_snowmelt[:, -1].size < 1:
                    melt_conv0 = 0
                else:
                    melt_conv0 = melt_conv
                # transform to doy then plot
                t_x_odd = []
                for item0 in [t_date, secs_valid, conv[0], conv_a[0]]:
                    # t0 = bxy.time_getlocaltime(item0, ref_time=[2015, 1, 1, 0])
                    t0 = item0 + bxy.get_total_sec('20150101')
                    # t0_doy = (t0[0]-2016)*366 + t0[-2]+t0[-1]/24.0
                    t_x_odd.append(t0)
                vline_list = [thaw_onset0, ascat_thaw_onset0, melt_onset0]
                vline_list2 = [item0+bxy.get_total_sec('20150101') for item0 in vline_list]
                odd_str0 = odd_id[land_id.index(i0)]
                figname0='result_08_01/customize_test_plot_subplot_%s_%d.png' % (odd_str0, i0)
                # set a period to zoom in the time series
                x_period = [bxy.get_total_sec(t_str) for t_str in ['20160328', '20160403']]
                x_period = [t_x_odd[0][0], t_x_odd[0][-1]]
                plot_funcs.plot_subplot([[t_x_odd[0], npr], [t_x_odd[1], series_valid],
                                         [t_x_odd[1], inc_valid[non_outlier]]],
                                        [[t_x_odd[2], conv[1]], [t_x_odd[3], conv_a[1]]],
                                        vline=[vline_list2, ['r-', 'b-', 'b-']],
                                        # vline=[onset_map_0_1d[i0], 'b-'],
                                        main_label=['npr', 'sigma0', 'incidence'],
                                        x_lim=x_period, y_lim2=[[1], [[-3, 3]]], y_lim=[[1], [[-20, -5]]],
                                        # x_lim=[t_x_odd[0][0], t_x_odd[0][-1]],
                                        figname=figname0, red_dots=False, x_unit='doy', main_check=1)
                # after plotting, save the ascat/npr time series for this pixel/station
                series_name_site = ['result_08_01/grided_series_npr_%s.npy' % odd_str0,
                               'result_08_01/grided_series_ascat_%s.npy' % odd_str0]
                series_vlaue_site = [[t_x_odd[0], npr], [t_x_odd[1], series_valid]]
                for var_name, var in zip(series_name_site, series_vlaue_site):
                    np.save(var_name, var)
                fig0 = plt.figure()
                ax0 = fig0.add_subplot(111)
                x_temp = t_x_odd[1]
                overpass_hr = bxy.time_getlocaltime(x_temp, ref_time=[2000, 1, 1, 12])
                ax0.plot(x_temp, overpass_hr[-1])
                plt.savefig('result_08_01/check_125ease_overpass.png')
                plt.close()
                output_combine.append([i0, t_x_odd, [npr, series_valid, conv[1], conv_a[1]],
                                       [thaw_onset0, ascat_thaw_onset0, melt_onset0,
                                        ]])
                # save the onset information for odd checking

                row0 = np.array([int(odd_str0), thaw_onset0, ascat_thaw_onset0, ascat_thaw_conv0, melt_onset0,
                                 melt_conv0, mean0_winter, std_winter, min_winter_mean])
                melt_table[land_id.index(i0), 0:row0.size] = row0

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

    # save the result
    # row0 = np.array([int(odd_str0), thaw_onset0, ascat_thaw_onset0, ascat_thaw_conv0, melt_onset0,
    #                              melt_conv0])
    # if ~odd_check:
    #     melt_table_name = 'result_08_01/melt_table.txt'
    #     heads0 = 'id, thaw_npr, thaw_ascat, conv_ascat, melt_ascat, conv_melt, conv_winter, conv_std_winter'
    #     melt_table_valid = melt_table[:, melt_table[1] != -999]
    #     print melt_table_valid.shape
    #     np.savetxt(melt_table_name, melt_table_valid, delimiter=',',
    #                header='id, thaw_npr, thaw_ascat, conv_ascat, melt_ascat, conv_melt, mean, std, conv_min_winter',
    #                fmt='%d, %d, %d, %.2f, %d, %.2f, %.2f, %.2f, %.2f')
    if onset_save is True:
        # save npr onset
        pre_path = 'result_08_01'
        th_name = '%s/test_onset0_%s.npy' % (pre_path, sigma_npr)
        fr_name = '%s/test_onset1_%s.npy' % (pre_path, sigma_npr)
        np.save(th_name, onset_map_0_1d.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save(fr_name, onset_map_1_1d.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save('test_onset0_tb.npy', onset_map_0_1d_tb.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save('test_onset1_tb.npy', onset_map_1_1d_tb.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        # save ascat
        th_ascat = '%s/thaw_onset_ascat_%d.npy' % (pre_path, sigma_npr)
        ml_ascat = '%s/melt_onset_ascat_%d.npy' % (pre_path, sigma_npr)
        ml_conv = '%s/melt_conv_ascat_%d.npy' % (pre_path, sigma_npr)
        np.save(th_ascat, onset_thaw_ascat.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save(ml_ascat, onset_melt_ascat.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        np.save(ml_conv, conv_melt_ascat.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        print 'the onset results are updated'
        print onset_map_0_1d.reshape(smap_tbv.shape[0], smap_tbv.shape[1])[47, 67]
        # save significance
        level_name = '%s/melt_level_%d.npy' % (pre_path, sigma_npr)
        mean_name = '%s/ascat_winter_mean_%d.npy' % (pre_path, sigma_npr)
        std_name = '%s/ascat_winter_std_%d.npy' % (pre_path, sigma_npr)
        for name0, var0 in zip([level_name, mean_name, std_name],
                               [level_melt_ascat, ascat_winter_mean, ascat_winter_std]):
            np.save(name0, var0.reshape(smap_tbv.shape[0], smap_tbv.shape[1]))
        level_melt_ascat[i0] = (melt_conv - mean0)/std0
        ascat_winter_mean[i0], ascat_winter_std[i0] = mean0, std0
    else:
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        x_temp = np.arange(0, dis_all.size)
        print 'the mean distance to the center', np.mean(dis_all)
        ax0.plot(x_temp, dis_all)
        plt.savefig('result_08_01/check_125ease_distance.png')
        plt.close()
        print 'the onset results are not saved'
    return output_combine




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


def melt_map(s_info_list, pixel_index=False, pixel_id=False, ascat_sigma=7):
    """
    :param s_info_list: [[0, longitude, latitude]]
    :param pixel_index and pixel_id: two list, 1st contains the 1d index, 2nd  the string format site id
    :return: melt_out:
    """
    # s_info_list = [[0,  -162.7, 69.1], [0,  -155.2, 70.1], [0, -153.5, 68.8], [0, -147.5, 68.8], [0, -153.5, 67.8],
    #                [0, -159.1, 60.5], [0, -159.0, 61.7], [0, -150.3, 64.7], [0, -147.3, 64.4], [0, -150.0, 62.0], [0, -162.5, 65.5]
    #                ,[0, -162.5, 63.0],  [0, -150.3, 66.7], [0, -147.3, 66.7]]
    # # s_info_list = [[1, 1, 1]]  # no special pixel
    points_index, x_time_points, value_points = [], [], []
    write_lat_lon = 0
    for s_info in s_info_list:
        for kernel0 in [7]:
            # 70.26666, -148.56666
            # s_info = [0, -1, -1]
            odd_latlon = [s_info[1], s_info[2]]
            thaw_win = np.array([30, 180])
            fr_win = np.array([250, 340])
            odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)
            # calculate onset
            if s_info[2]<0:
                # get 1d index
                for d_str in ['20151102']:
                    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % d_str
                    h0 = h5py.File(h5_name)
                    lons_1d = h0['cell_lon'].value.ravel()
                    lats_1d = h0['cell_lat'].value.ravel()
                    dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
                    p_index = np.argmin(dis_1d)
                    points_index.append(p_index)
                # return [npr_time, ascat_time, npr/ascat conv time]
                # and [npr, ascat sigma, npr/ascat conv]
            #     x_time, value = combine_detection(thaw_win, fr_win, sigma_npr=kernel0, sigma_ascat=3, ascat_detect=True, odd_plot=p_index)
            #     x_time_points.append(x_time), value_points.append(value)
            #     # write the sepcific pixel infos in a txt file
            #     with open('odd_pixel_infos.txt', 'a') as odd_info0:
            #         odd_info0.write('Odd pixel: %d \n' % p_index)
            #         odd_info0.write('1d index (36 km grid): %d \n' % p_index)
            #         odd_info0.write('location info: %s, %.2f, %.2f \n' % (s_info[0], s_info[2], s_info[1]))
            #         odd_info0.close()
            #     # copy time series map to the target folder and named by odd pixel_no
            #     cm_line = "cp result_08_01/test_plot_subplot.png result_08_01/temp/temp_comparison/pixel_no_%d.png" % p_index
            #     os.system(cm_line)
            # else:
            #     combine_detection(thaw_win, fr_win, sigma_npr=kernel0, sigma_ascat=3, ascat_detect=True)
            with open('result_08_01/pixel_index_lat_lon.txt', 'a') as f0:
                if write_lat_lon == 0:
                    f0.write('lat, lon, index\n')
                    write_lat_lon += 1
                f0.write('%.2f, %.2f, %d\n' % (odd_latlon[0], odd_latlon[1], p_index))
    if pixel_index is not False:
        points_index = pixel_index
    combined_output = combine_detection(thaw_win, fr_win, sigma_npr=kernel0, sigma_ascat=ascat_sigma, ascat_detect=True,
                                        odd_plot=points_index, odd_id=pixel_id)
    np.savetxt('pixel_index.txt', np.array(points_index), delimiter=',', fmt='%d')
    return combined_output


def compare_metop(site_no):
    txt_path0 ='./result_08_01/point/ascat/ascat_site_series/'
    # initial
    kernel = 3
    sate_orbt = ['B', 0, 1, 'A']
    pass_test = [sate_orbt[0], sate_orbt[2]]
    npyname = txt_path0+'ascat_s%s_2016%s.npy' % (site_no, pass_test[0])
    dis0 = 10
    sp_window = [[2016, 2, 1, 0, 0, 0], [2016, 6, 2, 8, 0, 0]]
    sec_window = []
    for t0 in sp_window:
        sec_window.append(bxy.get_secs(t0))
    ismean = False

    # read data
    # plot distance, pass hour, incidence angle
    # for a station and specific satellite and orbit
    test_site = np.load(npyname)
    print npyname, 'orbit is ', pass_test[1]
    test_site = test_site[test_site[:, -2] == pass_test[1]]  # descending
    loc_timeall = bxy.time_getlocaltime(test_site[:, 14], ref_time=[2000, 1, 1, 0])
    pass_title = '%s, %s_%d' % (npyname.split('/')[-1], pass_test[0], pass_test[1])
    plot_funcs.plot_subplot([[test_site[:, 14], test_site[:, -1]], [test_site[:, 14], test_site[:, 9]],
                             [test_site[:, 14], loc_timeall[-1]]], [],
                            figname='result_08_01/test_dis_inc_passhr'+site_no+'.png',
                            text=pass_title, x_unit='sec')
    sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series(site_no, orb_no=sate_orbt[1]+2, inc_plot=True, sigma_g=kernel, pp=False,
                                               order=1, txt_path=txt_path0, daily_mean=ismean, min_dis=dis0, time_window=sec_window)
    sigconvB, sigseriesB, ons_newB, ggB, sig_passB, peakdate_sigB = \
                data_process.ascat_plot_series(site_no, orb_no=sate_orbt[1], inc_plot=True, sigma_g=kernel, pp=False,
                                               order=1, txt_path=txt_path0, sate='A', daily_mean=ismean, min_dis=dis0, time_window=sec_window)
    sigconv1, sigseries1, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series(site_no, orb_no=sate_orbt[2], inc_plot=True, sigma_g=kernel, pp=False,
                                               order=1, txt_path=txt_path0, daily_mean=ismean, min_dis=dis0, time_window=sec_window)

    # sigconvB1, sigseriesB1, ons_newB, ggB, sig_passB, peakdate_sigB = \
    #             data_process.ascat_plot_series(site_no, orb_no=1, inc_plot=True, sigma_g=10, pp=False,
    #                                            order=1, txt_path=txt_path0, sate='A')
    # sigconv[0] -= 365
    # sigconv1[0] -= 365
    # sigconvB[0] -= 365

    # add air temperature & snow cover
    read_filed = "Air Temperature Observed (degC)"
    # read_filed = "Soil Temperature Observed -2in (degC)"
    if site_no in ['2065', '2081']:
        read_filed = "Air Temperature Average (degC)"
    metop_b_sect = np.arange(sec_window[0], sec_window[1], 4*3600)
    # metop_b_allsec = np.concatenate((sigseries1[0], sigseries[0]))
    # sec_seq = np.argsort(metop_b_allsec)
    # metop_b_sect = metop_b_allsec[sec_seq]
    # metop_sigma_all_air = np.concatenate((sigseries1[1], sigseries[1]))[sec_seq]
    doy_air = bxy.time_getlocaltime(metop_b_sect, ref_time=[2000, 1, 1, 0])
    T_air, T_doy = read_site.read_measurements(site_no, read_filed, doy_air[-2]+365, hr=doy_air[-1])
    swe, swe_doy = read_site.read_measurements(site_no, 'snow', doy_air[-2]+365, hr=0)
    T_doy -= 365
    swe_doy -= 365
    T_air[T_air<-90], swe[swe<-90] = np.nan, np.nan
    air_sec = bxy.get_secs([2016, 1, 1, 0, 0, 0])+T_doy*3600*24


    # add soil temperature, only for sitation 1233
    # read_filed = "Soil Temperature Observed -2in (degC)"
    # if site_no in ['2065', '2081']:
    #     read_filed = "Soil Temperature Observed -2in (degC)"
    # # metop_b_sect = np.arange(sec_window[0], sec_window[1], 2*3600)
    # metop_b_allsec = np.concatenate((sigseries1[0], sigseries[0]))
    # sec_seq = np.argsort(metop_b_allsec)
    # metop_b_sect = metop_b_allsec[sec_seq]
    # doy_air = bxy.time_getlocaltime(metop_b_sect, ref_time=[2000, 1, 1, 0])
    # T_soil, T_doy = read_site.read_measurements(site_no, read_filed, doy_air[-2]+365, hr=doy_air[-1])
    # swe, swe_doy = read_site.read_measurements(site_no, 'snow', doy_air[-2]+365, hr=0)
    # T_doy -= 365
    # swe_doy -= 365
    # T_soil[T_soil<-90], swe[swe<-90] = np.nan, np.nan
    # air_sec = bxy.get_secs([2016, 1, 1, 0, 0, 0])+T_doy*3600*24
    # metop_sigma_all_soil = np.concatenate((sigseries1[1], sigseries[1]))[sec_seq]

    # plotting
    title0 = 'orbit_%s_%d_%s_%d_%s_%d' \
             % (sate_orbt[0], sate_orbt[1], sate_orbt[3], sate_orbt[1], sate_orbt[0], sate_orbt[2])

    plot_funcs.plot_subplot([[sigseries[0], sigseries[1]], [sigseriesB[0], sigseriesB[1]], [sigseries1[0], sigseries1[1]], [air_sec, T_air]],
                            [[sigconv[0], sigconv[2]], [sigconvB[0], sigconvB[2]],
                             [sigconv1[0], sigconv1[2]], [air_sec, swe]],
                            text=title0, x_unit='sec', h_line=[3, 0],
                            figname='result_08_01/ascat_sigma'+site_no+'.png')
    # plot_funcs.plot_subplot([[T_air, metop_sigma_all_air], [T_soil, metop_sigma_all_soil]],
    #                         [[0, 0]],
    #                         text=title0,
    #                         figname='result_08_01/ascat_sigma'+site_no+'.png')


def north_region():
    # located
    conv_grid = np.load('melt_conv_ascat_7.npy')
    d_str = ['20151102']
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % d_str[0]
    h0 = h5py.File(h5_name)
    lon_gd = h0['cell_lon'].value
    lat_gd = h0['cell_lat'].value
    # lat_gd, lon_gd = spt_quick.get_grid()
    north_mask = (conv_grid < -2) & (lat_gd > 67)
    array_1d_01 = [a0.ravel() for a0 in [conv_grid, lat_gd, lon_gd]]
    # clip region
    inputif_path = '/home/xiyu/Data/nlcd'
    shap_path = 'result_08_01/area/shp/north_region.shp'
    ipt_tif = '%s/1km03_uncombine_tf.tif' % inputif_path
    output_tif = 'result_08_01/area/clip_result/north_region_class.tif'
    bash_command = "gdalwarp -cutline %s -crop_to_cutline %s %s" % (shap_path, ipt_tif, output_tif)
    os.system(bash_command)
    # read clip and resample
    fname_ipt, fname_opt = output_tif, 'result_08_01/area/clip_result/north_region_class.txt'
    comand_2_asc = ["gdal_translate", "-of", "AAIGrid", fname_ipt, fname_opt]
    sb.call(comand_2_asc)
    value = np.loadtxt(fname_opt, skiprows=6)
    class_1d = value.ravel()  # calculate percentage of types
    all_class = class_1d.size
    class_no = [30, 40, 50, 70, 80, 90]
    p_types = np.zeros(len(class_no))
    for i0, type0 in enumerate(class_no):
        p_types[i0] = sum((class_1d>type0)&(class_1d<type0+10))/all_class


def build_subgrid(box=[-160, -145, 67, 71]):
    d_str = ['20151102']
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % d_str[0]
    h0 = h5py.File(h5_name)
    shape0 = h0['cell_lon'].value.shape
    lon_gd = h0['cell_lon'].value.ravel()
    lat_gd = h0['cell_lat'].value.ravel()
    indx0 = [np.argmin(np.abs(lon_gd - box[0])), np.argmin(np.abs(lon_gd - box[1]))]  # west/east
    indx1 = [np.argmin(np.abs(lat_gd - box[2])), np.argmin(np.abs(lat_gd - box[3]))]  # south/north
    indx0_2d = np.array([bxy.trans_in2d(i0, shape0) for i0 in indx0])
    indx1_2d = np.array([bxy.trans_in2d(i0, shape0) for i0 in indx1])
    lon_new = h0['cell_lon'].value[np.min(indx0_2d): np.max(indx0_2d), np.min(indx1_2d): np.max(indx1_2d)]
    lat_new = h0['cell_lat'].value[np.min(indx0_2d): np.max(indx0_2d), np.min(indx1_2d): np.max(indx1_2d)]
    np.save('/home/xiyu/Data/easegrid2/ease_alaska_north_lon.npy', lon_new)
    np.save('/home/xiyu/Data/easegrid2/ease_alaska_north_lat.npy', lat_new)
    puase0 = 0


def exp_soil_tb_sigma(site_no):
    # variables
    obs = [0, '_A_', 18] # 0: As, 1:Des
    k_width_sig = 3
    k_width_tb =5
    # initial output
    # get site info
    s_info = site_infos.change_site(site_no)
    # tb time series and timing extract
    # sigma time series and timing extract
    sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series_v20(site_no, orb_no=obs[0], inc_plot=True, sigma_g=k_width_sig,
                order=1)
    tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual',
                seriestype='tb', tbob=obs[1], sig0=k_width_tb, order=1)
    # find turning point of tbv0
    min_date_v, min_date_h = data_process.turning_point(tbv0[0], tbv0[1], 0.5), \
                             data_process.turning_point(tbh0[0], tbh0[1], 0.5)
    # find T_soil greater than 0
    sec_2015 = bxy.get_total_sec('201501020000', fmt='%Y%m%d%H%M')
    check_2015 = bxy.time_getlocaltime([sec_2015], ref_time=[2000, 1, 1, 0])
    print 'the initial time is ', check_2015
    t_in_secs = np.arange(366, 365+365)*24*3600+sec_2015+18*3600
    secs_station = t_in_secs  # the x time for in situ measurements
    t_tuple_ascat = bxy.time_getlocaltime(secs_station, ref_time=[2000, 1, 1, 0])
    m_name0, m_name1 = "Soil Moisture Percent -2in (pct)", "Soil Temperature Observed -2in (degC)"
    S_soil, S_doy = read_site.read_measurements(site_no, m_name0, t_tuple_ascat[-2]+365, hr=18)
    T_soil, T_doy = read_site.read_measurements(site_no, m_name1, t_tuple_ascat[-2]+365, hr=18)
    S_soil[S_soil<-50] = np.nan
    T_soil[T_soil<-50] = np.nan
    onset_s, onset_t, onset_t1 \
        = data_process.sm_onset(S_doy-365, S_soil, T_soil)  # onset_t1, date when T_soil greater than 0
    # plotting
    sigma_ascat0 = [(sigseries[0]+365)*24*3600+sec_2015, sigseries[1]]  # time series of sigma
    sigma_conv0 = [(sigconv[0])*24*3600+sec_2015, sigconv[1]]  # sub axis one
    tb_h = [(tbv0[0]+365)*24*3600+sec_2015, tbv0[1]]
    tbh_conv0 = [(gau0[0]+365)*24*3600+sec_2015, gau0[1]]
    insitu_t, insitu_sm = [secs_station, T_soil], [secs_station, S_soil]
    print onset_t1
    vline0 = [(v0+365)*24*3600+sec_2015 for v0 in [min_date_v[1], onset_t1[0]]]
    figname = 'result_08_01/soil_zero_timing_%s.png' % site_no
    plot_funcs.plot_subplot([sigma_ascat0, tb_h, insitu_sm], [sigma_conv0, tbh_conv0, insitu_t],
                            x_unit='doy', symbol2='b-', vline=[vline0, ['g-', 'r-']], figname=figname,
                            x_lim=[secs_station[0], secs_station[-1]], red_dots=False)


def exp_soil_tb_sigma_copy(site_no):
    # variables
    obs = [0, '_A_', 18] # 0: As, 1:Des
    k_width_sig = 3
    k_width_tb =5
    # initial output
    # get site info
    s_info = site_infos.change_site(site_no)
    # tb time series and timing extract
    # sigma time series and timing extract
    sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series_v20(site_no, orb_no=obs[0], inc_plot=True, sigma_g=k_width_sig,
                order=1)
    tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual',
                seriestype='tb', tbob=obs[1], sig0=k_width_tb, order=1)
    # find turning point of tbv0
    min_date_v, min_date_h = data_process.turning_point(tbv0[0], tbv0[1], 0.5), \
                             data_process.turning_point(tbh0[0], tbh0[1], 0.5)
    # find T_soil greater than 0
    sec_2015 = bxy.get_total_sec('201501020000', fmt='%Y%m%d%H%M')
    check_2015 = bxy.time_getlocaltime([sec_2015], ref_time=[2000, 1, 1, 0])
    print 'the initial time is ', check_2015
    t_in_secs = np.arange(366, 365+365)*24*3600+sec_2015+18*3600
    secs_station = t_in_secs  # the x time for in situ measurements
    t_tuple_ascat = bxy.time_getlocaltime(secs_station, ref_time=[2000, 1, 1, 0])
    m_name0, m_name1 = "Soil Moisture Percent -2in (pct)", "Soil Temperature Observed -2in (degC)"
    S_soil, S_doy = read_site.read_measurements(site_no, m_name0, t_tuple_ascat[-2]+365, hr=18)
    T_soil, T_doy = read_site.read_measurements(site_no, m_name1, t_tuple_ascat[-2]+365, hr=18)
    S_soil[S_soil<-50] = np.nan
    T_soil[T_soil<-50] = np.nan
    onset_s, onset_t, onset_t1 \
        = data_process.sm_onset(S_doy-365, S_soil, T_soil)  # onset_t1, date when T_soil greater than 0
    # plotting
    sigma_ascat0 = [(sigseries[0]+365)*24*3600+sec_2015, sigseries[1]]  # time series of sigma
    sigma_conv0 = [(sigconv[0])*24*3600+sec_2015, sigconv[1]]  # sub axis one
    tb_h = [(tbv0[0]+365)*24*3600+sec_2015, tbv0[1]]
    tbh_conv0 = [(gau0[0]+365)*24*3600+sec_2015, gau0[1]]
    insitu_t, insitu_sm = [secs_station, T_soil], [secs_station, S_soil]
    print onset_t1
    vline0 = [(v0+365)*24*3600+sec_2015 for v0 in [min_date_v[1], onset_t1[0]]]
    figname = 'result_08_01/soil_zero_timing_%s.png' % site_no
    plot_funcs.plot_subplot([sigma_ascat0, tb_h, insitu_sm], [sigma_conv0, tbh_conv0, insitu_t],
                            x_unit='doy', symbol2='b-', vline=[vline0, ['g-', 'r-']], figname=figname,
                            x_lim=[secs_station[0], secs_station[-1]], red_dots=False)


def exp_tibet(site_no):
        # variables
    obs = [0, '_A_', 18] # 0: As, 1:Des
    k_width_sig = 3
    k_width_tb =5
    # initial output
    # get site info
    s_info = site_infos.change_site(site_no)
    # tb time series and timing extract
    # sigma time series and timing extract
    # sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
    #             data_process.ascat_plot_series_v20(site_no, orb_no=obs[0], inc_plot=True, sigma_g=k_width_sig,
    #             order=1)
    # tbv0, tbh0, npr0, gau0, ons0, tb_pass, peakdate0 = test_def.main(site_no, [], sm_wind=7, mode='annual',
    #             seriestype='tb', tbob=obs[1], sig0=k_width_tb, order=1)
    # 1 edge detection
    # 1_0 read time series
    ascat_name = 'result_08_01/20181101/ascat_series/ascat_s%s_2016all.npy' % site_no
    smap_name = 'result_08_01/20181101/smap_series/tb_%s_A_tibet' % site_no
    ascat_array, smap_array = np.load(ascat_name), np.loadtxt(smap_name)
    with open(smap_name) as f0:
        for row in f0:
            head_smap = row.split(',')
            break
    # 1_0_0 smap
    i_time, i_v, i_h = head_smap.index('cell_tb_time_seconds_aft'), \
                       head_smap.index('cell_tb_v_aft'), head_smap.index('cell_tb_h_aft')
    smap_time, smap_v, smap_h = smap_array[:, i_time], smap_array[:, i_v], smap_array[:, i_h]
    npr = (smap_v-smap_h)/(smap_v+smap_h)
    # 1_0_1 ascat, the metop A and B are mixed
    ascat_sigma = ascat_array[:, 3]
    ascat_incidence = ascat_array[:, 9]
    ascat_pass_utc = ascat_array[:, 14]
    ascat_type = ascat_array[:, 45]
    a, b = np.polyfit(ascat_incidence, ascat_sigma, 1)  # angular correction

    utc_time = ascat_pass_utc
    min_dis = 19
    sec_end = np.max(utc_time)
    local_time0 = bxy.time_getlocaltime([np.min(utc_time)], ref_time=[2000, 1, 1, 0])
    ini_hr = 4
    sec_ini = bxy.get_secs([local_time0[0], local_time0[1], local_time0[2], ini_hr, 0, 0], [2000, 1, 1, 0, 0])
    sec_span = 8*3600
    sec_step = 8*3600
    i2 = 0
    is_mean = False
    write_no = 0
    # initial a time series array, change sigma_out
    series0 = bxy.get_secs([local_time0[0], local_time0[1], local_time0[2], 0, 0, 0], [2000, 1, 1, 0, 0])
    total_days = 300
    series_sec = np.arange(series0, series0+total_days*24*3600, 24*3600)
    sigma_out = np.zeros([3, total_days*10]) - 999
    sigma_out[0, 0: series_sec.size] = series_sec  # daily mean, the x axis is integral day with unit of secs
    while sec_ini < sec_end:
        t_current = bxy.time_getlocaltime([sec_ini, sec_ini+sec_span], ref_time=[2000, 1, 1, 0])
        daily_idx = (utc_time > sec_ini) & (utc_time < sec_ini + sec_span) & (ascat_array[:, -1] < min_dis) \
                    & (ascat_array[:, 6] < 2) & (ascat_array[:, 9]>30)
        daily_sigma, daily_sec, dis_daily, daily_inc = \
            ascat_sigma[daily_idx], utc_time[daily_idx], ascat_array[:, -1][daily_idx], ascat_incidence[daily_idx]
        t_temp = bxy.time_getlocaltime(daily_sec, ref_time=[2000, 1, 1, 0])
        # write no data day
        if t_temp.size < 1:
            i2 += 1
        else:
            if is_mean is True:
                for t0 in [0]:
                    # value0 = np.mean(daily_sigma[t_temp[-1] == t0])
                    # value1 = np.mean(daily_inc[t_temp[-1] == t0])
                    # doy0_new = np.mean(t_temp[-2][t_temp[-1] == t0]) + t0/24.0
                    value0, value1, doy0_new = np.mean(daily_sigma), np.mean(daily_inc), np.mean(daily_sec)
                    # sigma_out[0, i2], sigma_out[1, i2], sigma_out[2, i2] = doy0_new, value0, value1
                    sigma_out[1, i2], sigma_out[2, i2] = value0, value1
                    i2 += 1
            else:
                # re-sampled hourly
                u_v, u_i = np.unique((daily_sec/3600).astype(int), return_index=True)  # seconds integral hour
                temp_v = np.zeros([u_i.size, 3]) - 999
                for i3 in range(0, u_i.size):
                    temp_v[0, i3] = u_v[i3]*3600
                    if i3 < u_i.size-1:
                        temp_v[i3, 1], temp_v[i3, 2] \
                            = np.mean(daily_sigma[u_i[i3]: u_i[i3+1]]), np.mean(daily_inc[u_i[i3]: u_i[i3+1]])
                    else:
                        temp_v[i3, 1], temp_v[i3, 2] \
                            = np.mean(daily_sigma[u_i[i3]: ]), np.mean(daily_inc[u_i[i3]: ])
                    sigma_out[:, i2] = temp_v[i3]
                    i2 += 1
        sec_ini += sec_step

    #sigma_c = ascat_sigma - (ascat_incidence - 40) * a
    # 1_1_0 edge detection
    out_valid = (sigma_out[0] > -999) & (sigma_out[1] > -999) & (sigma_out[2] > -999)
    ascat_pass_utc_daily, sigma_c_daily, sigma_inc_daily = sigma_out[0][out_valid],  sigma_out[1][out_valid],  sigma_out[2][out_valid]
    sigma_c_daily = sigma_c_daily - (sigma_inc_daily - 45) * a
    max_value, min_value, conv = test_def.edge_detect(smap_time, npr, 5, seriestype='npr')  # num is the kernel size
    max_value_a, min_value_a, conv_a =\
                        test_def.edge_detect(ascat_pass_utc_daily, sigma_c_daily, 5, seriestype='sig')
    # read site data
    site_measure_raw = read_site.read_tibet(site_no)
    k_size = 96
    k = np.zeros(k_size) + 1.0/k_size
    k2 = np.zeros(k_size)
    k2[k_size/2] = 1
    site_measure0 = np.zeros(site_measure_raw.shape) - 999
    site_measure0[:, 0] = site_measure_raw[:, 0]
    site_measure0[:, 1] = np.convolve(site_measure_raw[:, 1], k, 'same')
    site_measure0[:, 2] = np.convolve(site_measure_raw[:, 2], k, 'same')
    site_measure1 = site_measure0[k_size: -k_size, :]
    daily_index = np.arange(0, site_measure1.shape[0], 96)
    site_measure2 = site_measure1[daily_index, :]
    site_measure2[site_measure2[:, 1] < -90, 1] = np.nan
    site_measure2[site_measure2[:, 2] < -90, 2] = np.nan
    site_measure = site_measure2
    # plot them
    figname = 'result_08_01/tibet_%s.png' % site_no
    plot_funcs.plot_subplot([[smap_time, npr], [ascat_pass_utc_daily, sigma_c_daily], [site_measure[:, 0], site_measure[:, 1]]],
                            [conv, conv_a, [site_measure[:, 0], site_measure[:, 2]]],
                            x_unit='mmdd', symbol2='b-', figname=figname, main_label=['npr', 'sigma', 'vwc'], h_line=[0],
                            x_lim=[smap_time[0], smap_time[-1]], red_dots=False)
    return 0


def combine_detection_ad(pixel_id=[], pixel_plot=[], onset_save=False, ascat_detect=False, year=2016):
    # read source data
    path0 = 'result_08_01/area/combine_result'
    dis_all = np.array([])
    # ascat
    satellite_type = site_infos.get_satellite_type()
    ascat_h0 = h5py.File('result_08_01/area/combine_result/ascat_2016_3d_all.h5')
    ascat_sigma = ascat_h0['sigma'].value.copy()
    ascat_incidence = ascat_h0['incidence'].value.copy()
    ascat_pass_utc = ascat_h0['pass_utc'].value.copy()
    ascat_lon = ascat_h0['latitude'].value.copy()
    ascat_lat= ascat_h0['longitude'].value.copy()
    ascat_type = ascat_h0['sate_orbit'].value.copy()
    ascat_h0.close()

    # base map
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102'
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.copy().ravel()
    lats_1d = h0['cell_lat'].value.copy().ravel()
    region_sz = lons_1d.size  # the number of pixels of the alaska grid
    base_shape = h0['cell_lon'].value.shape
    h0.close()
    row_table = np.loadtxt('ascat_row_table.txt', delimiter=',')
    col_table = np.loadtxt('ascat_col_table.txt', delimiter=',')

    # smap
    att_list_smap = ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_v_fore', 'cell_tb_h_fore', 'cell_tb_time_seconds_fore',
                     'cell_tb_time_seconds_aft']
    att_list_smap = ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft']  # names (keys) of time series
    name_smap_a = '%s/smap_2016_A_3d.h5' % path0
    name_smap_d = '%s/smap_2016_D_3d.h5' % path0
    smap_ha, smap_hd = h5py.File(name_smap_a), h5py.File(name_smap_d)
    # get the x_time
    smap_input_a, smap_input_d = [], []
    for att0 in att_list_smap:  # looping to load data in order of v, h, secs.
        smap_input_a.append(smap_ha[att0].value.copy().reshape(region_sz, -1))
        smap_input_d.append(smap_hd[att0].value.copy().reshape(region_sz, -1))
    smap_ha.close()
    smap_hd.close()
    # mask the ocean
    mask = np.load(('./result_05_01/other_product/mask_ease2_360N.npy'))
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]
    if onset_save is False:
        land_id = pixel_id  # just test at station/pixles
        odd_check = True
        station_check = True
        station_pixel = [5336]
    # initial parameters
    thaw_window_doy, freeze_window_doy = [60, 150], [240, 340]
    ini_secs = bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12])
    thaw_window = [ini_secs + doy0*3600*24 for doy0 in thaw_window_doy]
    freeze_window = [ini_secs + doy0*3600*24 for doy0 in freeze_window_doy]
    thaw_onsets = np.zeros(region_sz)
    onset_map_0_1d = thaw_onsets
    onset_map_0_1d_v = thaw_onsets.copy()
    # loop started
    for i0 in land_id:
        # 0: v, 1: h, 2: time, didn't use the fore-ward mode
        lon0, lat0 = lons_1d[i0], lats_1d[i0]
        odd_check = i0 in pixel_plot
        t_a_secs, t_d_secs = smap_input_a[2][i0], smap_input_d[2][i0]
        t_tup_a, t_tup_d = bxy.time_getlocaltime(t_a_secs), bxy.time_getlocaltime(t_d_secs)
        i_year_a, i_year_d = t_tup_a[0] == year, t_tup_d[0] == year
        same_doy, i_same_a, i_same_d = np.intersect1d(t_tup_a[-2][i_year_a], t_tup_d[-2][i_year_d],
                                                      assume_unique=True, return_indices=True)
        # need check the distribution of overpass hr
        # check_a_pass, check_d_pass = t_tup_a[-1][i_year_a][i_same_a], t_tup_d[-1][i_year_d][i_same_d]
        # temp_test.check_smap_overpass(check_a_pass, check_d_pass)

        # calculate the orbit difference, the index is [:, i_year_a][:, i_same_a], the time_x: same_doy
        smap_1 = np.array(smap_input_a)[:, i0, :][:, i_year_a][:, i_same_a]  # 1: ascending
        smap_0 = np.array(smap_input_d)[:, i0, :][:, i_year_d][:, i_same_d]  # 2: descending
        smap_masked = []
        for smap0 in [smap_0[0], smap_0[1], smap_1[0], smap_1[1]]:  # reject the -9999
            smap0_new = bxy.nan_set(smap0, -9999.0)
            smap_masked.append(smap0_new)
        diff_tbv, diff_tbh = smap_masked[0]-smap_masked[2], smap_masked[1]-smap_masked[3]
        x_time = smap_0[2]

        # obtain edge, thaw onest, use h-pol first
        conv, thaw_onset_sec = data_process.get_onset(x_time, diff_tbh, thaw_window=thaw_window, freeze_window=freeze_window)
        thaw_onset0_tuple = bxy.time_getlocaltime([thaw_onset_sec], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
        onset_map_0_1d[i0] = thaw_onset0_tuple[-2][0]
        conv_vpol, thaw_onset_sec_vpol = data_process.get_onset(x_time, diff_tbv, thaw_window=thaw_window, freeze_window=freeze_window)
        thaw_onset0_tuple_vpol = bxy.time_getlocaltime([thaw_onset_sec_vpol], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
        onset_map_0_1d_v[i0] = thaw_onset0_tuple_vpol[-2][0]
        # plot if necessary
        fig_name_ad = 'result_08_01/smap_ad_difference_%d.png' % i0
        if odd_check:
            plot_funcs.plot_subplot([[smap_0[2], diff_tbh], [smap_0[2], diff_tbv]],
                                    [[conv[0], conv[1]]],
                                    figname=fig_name_ad, x_unit='sec', main_label=['Diff_tbh', 'Diff_tbv'])
        if station_check:
            if i0 in station_pixel:
                print 'time series in stations are plotted'
        if ascat_detect == True:
            test = 0
            # 9xN array for ascat measurements within 36km
            lat_9 = ascat_lat[row_table[i0].astype(int), col_table[i0].astype(int)]
            lon_9 = ascat_lon[row_table[i0].astype(int), col_table[i0].astype(int)]
            dis_9 = bxy.cal_dis(lat0, lon0, lat_9, lon_9)  # distance to the center of 36 km  pixel
            row_no = row_table[i0].astype(int)
            col_no = col_table[i0].astype(int)
            sigma_series_9 = ascat_sigma[row_no, col_no, :]
            incidence_series_9 = ascat_incidence[row_no, col_table[i0].astype(int), :]
            t_ascat_9 = ascat_pass_utc[row_no, col_table[i0].astype(int), :]
            index_invalid_0 = (incidence_series_9) < 30 | (incidence_series_9 > 55) \
                                                     | (sigma_series_9 == -999) | (sigma_series_9 == 0)
            sigma_series_9[index_invalid_0], incidence_series_9[index_invalid_0], t_ascat_9[index_invalid_0] = \
                np.nan, np.nan, np.nan
            sigma_series_mean = np.nanmean(sigma_series_9, axis=0)
            incidence_series_mean = np.nanmean(incidence_series_9, axis=0)
            t_ascat = np.nanmean(t_ascat_9, axis=0)
            valid_index = (sigma_series_mean > -25) & (sigma_series_mean < -0.1) \
                          & (incidence_series_mean > 31) & (incidence_series_mean < 52)
            if sum(valid_index)<150:
                # set a unvalid label
                continue
            else:
                # check distance
                dis_2_cent = dis_9.ravel()
                dis_all = np.concatenate((dis_all, dis_2_cent))
                # angular dependency for diferent sate tpye
                series_valid = sigma_series_mean.copy()
                angulars = np.zeros([4, 3])
                for type0 in [0, 1, 2, 3]:
                    type_id = (ascat_type == type0) & valid_index
                    inc0, sigma0 = incidence_series_mean[type_id], \
                                   sigma_series_mean[type_id]
                    a0, b0 = np.polyfit(inc0, sigma0, 1)
                    # remove angular dependency separately
                    series_valid[type_id] = sigma_series_mean[type_id] - (incidence_series_mean[type_id]-45)*a0
                    angulars[type0] = np.array([type0, a0, b0])
                # different overpass and satellites, check the orbit difference,
                # t_ascat, series_valid, angulars
                i_a_orbit, i_d_orbit = ((ascat_type == 0) | (ascat_type == 2)) & valid_index, \
                                       ((ascat_type == 1) | (ascat_type == 3)) & valid_index
                t_ascat_a, t_ascat_d = t_ascat[i_a_orbit], t_ascat[i_d_orbit]
                t_ascat_a_tp, t_ascat_d_tp = bxy.time_getlocaltime(t_ascat_a, ref_time=[2000, 1, 1, 0]), \
                                             bxy.time_getlocaltime(t_ascat_d, ref_time=[2000, 1, 1, 0])
                i_year_asa, i_year_asd = t_ascat_a_tp[0] == year, t_ascat_d_tp[0] == year
                # check pass time
                t_ascat_all_tp = bxy.time_getlocaltime(t_ascat[valid_index], ref_time=[2000, 1, 1, 0])
                t0, t1 = 200, 250
                print 'time %d to %d is \n' % (t0, t1), \
                    np.array([t_ascat_all_tp[-2][t0: t1],
                              t_ascat_all_tp[-1][t0: t1],
                              ascat_type[valid_index][t0: t1]]).T

                pass_a, pass_d = np.array([t_ascat_a_tp[-1][i_year_asa][0: 10], t_ascat_a_tp[-2][i_year_asa][0: 10]]), \
                                 np.array([t_ascat_d_tp[-1][i_year_asd][0: 10], t_ascat_d_tp[-2][i_year_asd][0: 10]])
                print 'the pass of asc', pass_a.T
                print 'the pass of des', pass_d.T
                return 0
                same_ascat_doy, i_same_ascat_a, i_same_ascat_d = \
                    np.intersect1d(t_ascat_a_tp[-2][i_year_asa], t_ascat_d_tp[-2][i_year_asd],
                                   return_indices=True)
                # [i_a_orbit][i_year_asa][i_same_ascat_a]
                print 'the size of a: ', t_ascat_a_tp[-2][i_year_asa].size
                print 'the size of indices: ', i_same_ascat_a.size
                print 'the size of same da: ', same_ascat_doy.size

                print t_ascat_a[i_year_asa].size, i_same_ascat_a.size
                print t_ascat_a_tp[-2][i_year_asa][i_same_ascat_a].size
                ascat_as = np.array(t_ascat_a[i_year_asa][i_same_ascat_a],
                                    series_valid[i_a_orbit][i_year_asa][i_same_ascat_a],
                                    angulars[i_a_orbit][i_year_asa][i_same_ascat_a])
                ascat_des = np.array(t_ascat_d[i_year_asd][i_same_ascat_d],
                                     series_valid[i_d_orbit][i_year_asd][i_same_ascat_d],
                                     angulars[i_d_orbit][i_year_asd][i_same_ascat_d])
                # check timing:
                f_ascat_doy = plt.figure()
                ax_ascat0 = f_ascat_doy.add_subplot(1, 1, 1)
                as_doy, des_doy = bxy.time_getlocaltime(ascat_as[0], ref_time=[2000, 1, 1, 0]), \
                                  bxy.time_getlocaltime(ascat_des[0], ref_time=[2000, 1, 1, 0])
                ax_ascat0.plot(as_doy[-2], des_doy[-2], 'k.')
                plt.savefig('result_08_01/ascat_pass_overlap.png')
                plt.close()
                f2 = plt.figure()
                ax2 = f2.add_subplot(1, 1, 1)
                ax2.plot(np.arange(0, as_doy[-1].size), as_doy[-1]-des_doy[-1], 'k.')
                plt.savefig('result_08_01/ascat_different_overpass.png')
                plt.close()
    return 0


def detection_ad_ascat():
    return 0


def odd_plot_in_map(points_info, s_info= [0,  -151.5, 62.1]):
    # plot result
    # s_info = site_infos.change_site('1090')
    # # 70.26666, -148.56666
    # s_info = [0, s_info[2], s_info[1]]
    # s_info = [0,  -151.5, 62.1]
    # # s_info = [0, 1, 1]
    odd_latlon = [s_info[2], s_info[1]]
    thaw_win = np.array([30, 180])
    fr_win = np.array([250, 340])
    odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)

    # all_result = np.load('20181104_result.npz')
    # for fname in all_result.files:
    #      np.save('result_agu/result_2019/%s_onset.npy' % fname, all_result[fname])
    # quit0()
    filepath = 'result_agu/result_2019'
    custom3 = [
                    # 'result_agu/smap_thaw_d_onset.npy',  #0
                    # 'result_agu/smap_thaw_a_onset.npy',  #1
                    # 'result_agu/smap_thaw_obdv_onset.npy',  #2
                    # '%s/ascat_thaw_360_onset.npy' % filepath,  #3
                    '%s/smap_thaw_obdh_onset.npy' % filepath  #4
              ]
    for c0 in custom3:
        # no odd pixel are plotted
        custom_name = [
                        # '%s/smap_thaw_a_onset.npy' % filepath,
                        # '%s/ascat_melt_360_onset.npy' % filepath,
                        # '%s/smap_thaw_2_onset.npy' % filepath,
                        # '%s/ascat_melt_2_onset.npy' % filepath,
                        '%s/smap_thaw_3_onset.npy' % filepath,
                        '%s/ascat_thaw_360_onset.npy' % filepath,
                        # 'result_08_01/melt_onset_ascat_7.npy',
                        c0]
        input_onset = data_process.prepare_onset(custom_name)

        points_index = points_info
        data_process.ascat_onset_map('A', odd_point=np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]]),
                                 points_index=np.array(points_index),
                                 product='input_onset', mask=False, mode=['_norm_'], version='old', std=7,
                                 f_win=fr_win, t_win=thaw_win,
                                 custom=custom_name, input_onset=input_onset)

def read_ak_yearly_melt(year0):
    # prepare onset_npy file
    mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]  # all land id in alaska
    onset_value = np.load('onset_%d.npz' % year0)
    melt_onset = onset_value['arr_0']
    melt_doy = bxy.time_getlocaltime(melt_onset)[-2]
    melt_doy[melt_onset<1] = -1

    # add value to array
    onset_array0 = np.zeros(9000)
    onset_array0[land_id] = melt_doy
    return onset_array0.reshape(90, 100)


def quick_plot_map_v2(value, resolution, points=False, points_index=False,
                      fig_name='finer_ascat', z_value=[30, 180], s_info= [0,  -151.5, 62.1], year0=2016,
                      p_name=False):
    """
    plot ak map, mark the pixel of interest
    :param points_info:
    :param s_info:
    :param year0:
    :return:
    """
    lons_grid, lats_grid, p_sensor = ind2latlon(points_index, resolution=resolution)
    out_bound = lons_grid>-141.
    value[out_bound] = 0
    value_ma = np.ma.masked_array(value, mask=[(value == -999)])
    p_odd_latlon = np.array([lons_grid.ravel()[p_sensor], lats_grid.ravel()[p_sensor]])
    data_process.pass_zone_plot(lons_grid, lats_grid, value_ma, './result_08_01/', fname=fig_name,
                                odd_points=p_odd_latlon.T, odd_index=p_sensor,
                                z_max=z_value[0], z_min=z_value[1], prj='aea', title_str='test', txt=p_name)  # fpath1
    return 0

    input_onset16 = read_ak_yearly_melt(2016)
    input_onset17 = read_ak_yearly_melt(2017)
    input_onset18 = read_ak_yearly_melt(2018)
    odd_latlon = [s_info[2], s_info[1]]
    thaw_win = np.array([30, 180])
    fr_win = np.array([250, 340])
    odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)
    # all_result = np.load('20181104_result.npz')
    # for fname in all_result.files:
    #      np.save('result_agu/result_2019/%s_onset.npy' % fname, all_result[fname])
    # quit0()

    points_index = points_info
    data_process.ascat_onset_map('A', odd_point=np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]]),
                                 points_index=np.array(points_index),
                                 product='input_onset', mask=False, mode=['_norm_'], version='old', std=7,
                                 f_win=fr_win, t_win=thaw_win,
                                 custom=['result_agu/result_2019/smap_thaw_obdh_onset.npy'
                                         , 'result_agu/result_2019/smap_thaw_obdh_onset.npy',
                                         'result_agu/result_2019/smap_thaw_obdh_onset.npy'],
                                 input_onset=[input_onset16, input_onset17, input_onset18])


def quick_plot_map(points_info, s_info= [0,  -151.5, 62.1], year0=2016):
    input_onset16 = read_ak_yearly_melt(2016)
    input_onset17 = read_ak_yearly_melt(2017)
    input_onset18 = read_ak_yearly_melt(2018)
    odd_latlon = [s_info[2], s_info[1]]
    thaw_win = np.array([30, 180])
    fr_win = np.array([250, 340])
    odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)
    # all_result = np.load('20181104_result.npz')
    # for fname in all_result.files:
    #      np.save('result_agu/result_2019/%s_onset.npy' % fname, all_result[fname])
    # quit0()

    points_index = points_info
    data_process.ascat_onset_map('A', odd_point=np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]]),
                                 points_index=np.array(points_index),
                                 product='input_onset', mask=False, mode=['_norm_'], version='old', std=7,
                                 f_win=fr_win, t_win=thaw_win,
                                 custom=['result_agu/result_2019/smap_thaw_obdh_onset.npy'
                                         , 'result_agu/result_2019/smap_thaw_obdh_onset.npy',
                                         'result_agu/result_2019/smap_thaw_obdh_onset.npy'],
                                 input_onset=[input_onset16, input_onset17, input_onset18])
    return 0
    filepath = 'result_agu/result_2019'
    custom3 = [
                    # 'result_agu/smap_thaw_d_onset.npy',  #0
                    # 'result_agu/smap_thaw_a_onset.npy',  #1
                    # 'result_agu/smap_thaw_obdv_onset.npy',  #2
                    # '%s/ascat_thaw_360_onset.npy' % filepath,  #3
                    '%s/smap_thaw_obdh_onset.npy' % filepath  #4
              ]
    for c0 in custom3:
        # no odd pixel are plotted
        custom_name = [
                        # '%s/smap_thaw_a_onset.npy' % filepath,
                        # '%s/ascat_melt_360_onset.npy' % filepath,
                        # '%s/smap_thaw_2_onset.npy' % filepath,
                        # '%s/ascat_melt_2_onset.npy' % filepath,
                        '%s/smap_thaw_3_onset.npy' % filepath,
                        '%s/ascat_thaw_360_onset.npy' % filepath,
                        # 'result_08_01/melt_onset_ascat_7.npy',
                        c0]
        input_onset = data_process.prepare_onset(custom_name)

        points_index = points_info
        data_process.ascat_onset_map('A', odd_point=np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]]),
                                 points_index=np.array(points_index),
                                 product='input_onset', mask=False, mode=['_norm_'], version='old', std=7,
                                 f_win=fr_win, t_win=thaw_win,
                                 custom=custom_name, input_onset=input_onset)


def drawing_parameter(ids):
    table_write = 0
    tables = np.zeros([ids.shape[1], 8]) - 999
    for id0, id_name in zip(ids[0], ids[1]):
        npy_list = []
        for file in ['main1', 'main2', 'main3', 'sec1', 'sec2', 'sec3', 'vline',
                     'lim', 'maintb0', 'maintb2', 'sate_type', 'ascat_angle']:
            npy_name = 'result_agu/%s_%d_%d.npy' % (file, id0, id_name)
            f0_reader = np.load(npy_name)
            npy_list.append(f0_reader)
        npy_list[0][1]*=100
        npy_list[1][1]*=100
        thaw_onset0 = data_process.peak_find(npy_list[0])
        melt_onset0 = data_process.peak_find(npy_list[2], p=-1.0)
        # timing I and II, based on NPR and sigma_0
        k0 = 5
        conv_sigma, sup_onset2 = \
            data_process.get_onset(npy_list[2][0], npy_list[2][1],
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=k0, type='sig')  # sigma_0 up
        conv_npr_a, npr_a_t1 = \
            data_process.get_onset(npy_list[0][0], npy_list[0][1],
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=k0, type='npr')  # npr asc up
        npy_list[1][1][npy_list[1][1]<-1] = -99
        conv_npr_d, npr_d_t1 = \
            data_process.get_onset(npy_list[1][0], npy_list[1][1],
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=k0, type='npr')  # npr des up
        conv_sigma, sigma_t3_up, sigma_melt_t1 = \
            data_process.get_onset_new(npy_list[2][0], npy_list[2][1],
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=7, type='npr', melt_window=npr_a_t1)

        npr_a_t2_0_x = data_process.get_onset_zero_x(conv_npr_a, npr_a_t1)
        sig_t2_0_x = data_process.get_onset_zero_x(conv_sigma, sigma_melt_t1[0], zero_x=-0.5)

        thaw_onset0_tuple = bxy.time_getlocaltime([thaw_onset0], ref_time=[2015, 1, 1, 0], t_source='US/Alaska')
        npy_list[6] = np.array([np.nan, melt_onset0, thaw_onset0])

        # show confidence level in the plot
        # order: 0: onset sects 1: level, 2: conv_edge_onset, 3: conv_winter_mean, 4: conv_winter_std
        text_example = '%.3f $\pm$ %.3f,  %.3f (%.3f)' \
                       % (sigma_melt_t1[3], sigma_melt_t1[4], sigma_melt_t1[2], sigma_melt_t1[1])
        # plot t1 & t2
        # plot_funcs.plot_subplot([npy_list[0],
        #                      npy_list[1],
        #                      npy_list[2]],
        #                     [conv_npr_a, npy_list[4], conv_sigma],
        #                     main_label=['NPR PM ($10^{-2})$', 'NPR AM ($10^{-2})$', '$\sigma^0$ (dB)'],
        #                     vline=[[npr_a_t1, sigma_melt_t1[0], npr_a_t2_0_x, sig_t2_0_x],
        #                            ['k-', 'r-', 'k:', 'r:'], ['npr_up', 'sig_down', 'npr_max', 'sig_min']],
        #                     x_unit='sec', x_lim=npy_list[7],
        #                     figname='result_agu/result_2019/t1_and_t2/20181202plot_ascat_%d_%dk0.png' % (id_name, id0),
        #                     annote=[-1, text_example])

    # timing III: 8: tb asc, 9: tb des
        npy_list[8][1:, :][npy_list[8][1:, :] == -9999] = np.nan
        npy_list[9][1:, :][npy_list[9][1:, :] == -9999] = np.nan
        tbmin_onset2 = data_process.peak_find(npy_list[8][0:2, :], p=-1.0)  # tb min on V
        tbmin_onset2_d = data_process.peak_find(npy_list[9][0:2, :], p=-1.0)  # tb min on v
        conv_tb_a, tb_a_onset2 = \
            data_process.get_onset(npy_list[8][0], npy_list[8][1],
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=k0, type='tb')  # tb asc down
        tb_a_t3_0_cross = conv_tb_a[0][(conv_tb_a[0]>tb_a_onset2) & (conv_tb_a[1] > -0.5)][0]
        conv_tb_d, tb_d_onset2 = \
            data_process.get_onset(npy_list[9][0], npy_list[9][1],
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=k0, type='tb')  # sigma_0
        tb_d_t3_0_cross = conv_tb_d[0][(conv_tb_d[0]>tb_d_onset2) & (conv_tb_d[1] > -0.5)][0]
        # plot t3
        # plot_funcs.plot_subplot([npy_list[8],
        #                      npy_list[9],
        #                      npy_list[2]],
        #                     [conv_tb_a, conv_tb_d, conv_sigma],
        #                     main_label=['NPR PM ($10^{-2})$', 'NPR AM ($10^{-2})$', '$\sigma^0$ (dB)'],
        #                     vline=[[tb_a_onset2, tb_a_t3_0_cross, sup_onset2],
        #                            ['k-', 'b-', 'r-'], ['tb_down', 'tb_min', 'sig_up']],  # tb down, tb min, sigma up
        #                     x_unit='sec', x_lim=npy_list[7],
        #                     figname='result_agu/result_2019/t3/20181202plot_ascat_%d_%d_tb.png' % (id_name, id0))
        # check ascat noise, angular dependency
        angle0 = npy_list[11]
        sate_type = npy_list[10]
        fig0 = plt.figure()
        ax_angle = fig0.add_subplot(1, 1, 1)
        ax_angle.plot(angle0, npy_list[2][1], 'ko')
        plt.savefig('check_angle_remove_%d_%d.png' % (id_name, id0))
        plt.close()
        # split the ascat by different orbits
        i_asc, i_des = sate_type < 2, sate_type > 1
        sigma_a, sigma_d = npy_list[2][:, i_asc], npy_list[2][:, i_des]
        conv_sigma_a, sigma_t3_up_a, sigma_melt_t1_a = \
            data_process.get_onset_new(sigma_a[0], sigma_a[1],  # onset t3, melt t1 based on single orbit
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=3, type='npr', melt_window=npr_a_t1)
        conv_sigma_d, sigma_t3_up_d, sigma_melt_t1_d = \
            data_process.get_onset_new(sigma_d[0], sigma_d[1],  # onset t3, melt t1 based on single orbit
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=3, type='npr', melt_window=npr_a_t1)
        # plot result

        text_example0 = '%.3f $\pm$ %.3f,  %.3f (%.3f)' \
                       % (sigma_melt_t1_a[3], sigma_melt_t1_a[4], sigma_melt_t1_a[2], sigma_melt_t1_a[1])
        text_example1 = '%.3f $\pm$ %.3f,  %.3f (%.3f)' \
                       % (sigma_melt_t1_d[3], sigma_melt_t1_d[4], sigma_melt_t1_d[2], sigma_melt_t1_d[1])
        plot_funcs.plot_subplot([npy_list[2][:, i_asc], npy_list[2][:, i_des], npy_list[2]],
                            [conv_sigma_a, conv_sigma_d, conv_sigma],
                            main_label=['$\sigma^0$ PM ', '$\sigma^0$ AM ', '$\sigma^0$ (dB)'],
                            vline=[[npr_a_t1, sigma_melt_t1[0], npr_a_t2_0_x, sig_t2_0_x],
                                   ['k-', 'r-', 'k:', 'r:'], ['npr_up', 'sig_down', 'npr_max', 'sig_min']],
                            x_unit='sec', x_lim=npy_list[7],
                            figname='result_agu/result_2019/new/ascat_split_%d_%dk0.png' % (id_name, id0),
                            annote=[[0, 1, 2], [text_example0, text_example1, text_example]])

        # save result in tables
        tables[table_write, 0] = id_name
        tables_2 = tables.copy()  # results from descending orbit
        onsets_doy = bxy.time_getlocaltime([npr_a_t1, sigma_melt_t1[0], npr_a_t2_0_x, sig_t2_0_x, tb_a_t3_0_cross,
                                            sigma_t3_up])[3]
        tables[table_write, 1: -1] = onsets_doy
        tables[table_write, -1] = sigma_melt_t1[1]
        table_write += 1
    fmt_list = []
    for item0 in tables[0]:
        fmt_list.append('%d')
    fmt_list[-1] = '%.3f'
    np.savetxt('result_agu/result_2019/table_result.txt', tables,
               delimiter=',', header='station,t1,t1_b,t2,t2_b,t3,t3_b,t1_level', fmt=','.join(fmt_list))


def read_estimate(ids, npys=['npr', 'ascat', 'conv_npr', 'conv_ascat', 'vline',
                     'lim', 'tb',  'sate_type', 'ascat_angle'], orbit='A'):

    return 0


def drawing_parameter_v2(ids0, npy_type=['npr', 'ascat', 'conv_npr', 'conv_ascat', 'vline',
                     'lim', 'tb',  'sate_type', 'ascat_angle'],
                         orbit='A', isplot=True, is_save=True, ks=[7, 7, 7], year0=2016):
    """
    :param ids0:
    :param npy_type:
    :param orbit:
    :param isplot: if true, plot the 'npr + ascat', 'tb + ascat', and 'ascat single orbit + ascat' onset estimation
    :param is_save:
    :param ks:
    :param year0:
    :return: if is_save, saving the 3 timings in a table, else return npy_dict, conv_dict, onset_dict, orbit_ascat
    """
    table_write = 0
    ids = ids0.copy()
    ids.shape = 2, -1
    tables = np.zeros([ids.shape[1], 8]) - 999
    k_ascat = ks[2]
    k0, k1 = ks[0], ks[1]
    for id0, id_name in zip(ids[0], ids[1]):
        npy_list = []
        npy_dict = {}
        conv_dict = {}
        onset_dict = {}
        for file in npy_type:
            if file in ['npr', 'conv_npr', 'tb']:
                if year0 == 2016:
                    npy_name = 'result_agu/npy_station_plot/%s_%s_%d_%d.npy' % (file, orbit, id0, id_name)
                else:
                    npy_name = 'result_agu/npy_station_plot/%s_%s_%d_%d_%d.npy' % (file, orbit, id0, id_name, year0)
            else:
                if year0 == 2016:
                    npy_name = 'result_agu/npy_station_plot/%s_%d_%d.npy' % (file, id0, id_name)
                else:
                    npy_name = 'result_agu/npy_station_plot/%s_%d_%d_%d.npy' % (file, id0, id_name, year0)
            f0_reader = np.load(npy_name)
            npy_list.append(f0_reader)
            npy_dict[file] = f0_reader
        npy_dict['npr'][1]*=100
        # set time window
        thaw_window_12 = [bxy.get_total_sec('%d0101' % year0, reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]]
        thaw_window_0 = [bxy.get_total_sec('%d0101' % year0, reftime=[2000, 1, 1, 0]) +
                                   doy0*3600*24 for doy0 in [60, 150]]
        conv_npr_a, npr_a_t1 = \
            data_process.get_onset(npy_dict['npr'][0], npy_dict['npr'][1],
                                   thaw_window=thaw_window_12,
                                   k=k0, type='npr', year0=year0)
        npy_dict['npr'][1][npy_dict['npr'][1]<-1] = -99
        conv_dict['npr'] = conv_npr_a
        onset_dict['t1_npr'] = npr_a_t1
        # ascat t1 and t3
        melt_st = npr_a_t1 - 7*24*3600
        conv_sigma, sigma_t3_up, sigma_melt_t1 = \
            data_process.get_onset_new(npy_dict['ascat'][0], npy_dict['ascat'][1],
                                   thaw_window=thaw_window_0,
                                   k=k1, type='sigma', melt_window=melt_st, mode=2, year0=year0)
        # np.array([melt_onset0, s_level, conv_edge_onset, winter_noise_std, winter_noise])
        orbit_ascat = 'AD'
        i_asc, i_des = npy_dict['sate_type'] < 2, npy_dict['sate_type'] > 1
        sigma_all = npy_dict['ascat'].copy()
        sigma_a, sigma_d = npy_dict['ascat'][:, i_asc], npy_dict['ascat'][:, i_des]

        # if sigma_melt_t1[1] < 1:  # significant level
        #     # split the ascat by different orbits
        #     conv_sigma_a, sigma_t3_up_a, sigma_melt_t1_a = \
        #         data_process.get_onset_new(sigma_a[0], sigma_a[1],  # onset t3, melt t1 based on single orbit
        #                                thaw_window=thaw_window_0,
        #                                k=k_ascat, type='sigma', melt_window=npr_a_t1, mode=2, year0=year0)
        #     if sigma_melt_t1_a[1] > 0:
        #        conv_sigma, sigma_t3_up, sigma_melt_t1 = conv_sigma_a, sigma_t3_up_a, sigma_melt_t1_a
        #        orbit_ascat = 'A'
        #        npy_dict['ascat'] = sigma_a
        #     else:
        #         conv_sigma_d, sigma_t3_up_d, sigma_melt_t1_d = \
        #             data_process.get_onset_new(sigma_d[0], sigma_d[1],  # onset t3, melt t1 based on single orbit
        #                                    thaw_window=thaw_window_0,
        #                                    k=k_ascat, type='sigma', melt_window=npr_a_t1, mode=2, year0=year0)
        #         if sigma_melt_t1_d[1] > 0:
        #             conv_sigma, sigma_t3_up, sigma_melt_t1 = conv_sigma_d, sigma_t3_up_d, sigma_melt_t1_d
        #             orbit_ascat = 'D'
        #             npy_dict['ascat'] = sigma_d

        npr_a_t2_0_x = data_process.get_onset_zero_x(conv_npr_a, npr_a_t1)
        sig_t2_0_x = data_process.get_onset_zero_x(conv_sigma, sigma_melt_t1[0], zero_x=0)
        conv_dict['ascat'] = conv_sigma
        onset_dict['t1_ascat'] = sigma_melt_t1
        onset_dict['t3_ascat'], onset_dict['t2_npr'], onset_dict['t2_ascat'] = sigma_t3_up, npr_a_t2_0_x, sig_t2_0_x

        # show confidence level in the plot
        text_example = '%.3f,  (%.3f) %.3f %.2f' \
                       % (sigma_melt_t1[4], 2*sigma_melt_t1[3], sigma_melt_t1[2], sigma_melt_t1[1])
        if isplot:
            # plot t1 & t2
            plot_funcs.plot_subplot([npy_dict['npr'],
                                 npy_dict['npr'],
                                 npy_dict['ascat']],
                                [conv_npr_a, npy_list[4], conv_sigma],
                                main_label=['NPR PM ($10^{-2})$', 'NPR AM ($10^{-2})$', '$\sigma^0$ (dB)'],
                                vline=[[npr_a_t1, sigma_melt_t1[0], npr_a_t2_0_x, sig_t2_0_x],
                                       ['k-', 'r-', 'k:', 'r:'], ['npr_up', 'sig_down', 'npr_max', 'sig_min']],
                                x_unit='sec', x_lim=npy_dict['lim'],
                                figname='result_agu/result_2019/t1_and_t2/20181202plot_ascat_%d_%dk0.png' % (id_name, id0),
                                annote=[-1, text_example])

    # timing III based on TB: v-pol and h-pol
        npy_dict['tb'][1:, :][npy_dict['tb'][1:, :] == -9999] = np.nan
        conv_tb_a, tb_a_onset2 = \
            data_process.get_onset(npy_dict['tb'][0], npy_dict['tb'][1],
                                   thaw_window=thaw_window_0,
                                   k=k0, type='tb', year0=year0)  # tb asc down vertical
        print 'current ids of the pixel: ', id0, id_name
        tb_a_t3_0_cross = conv_tb_a[0][(conv_tb_a[0]>tb_a_onset2) & (conv_tb_a[1] > -0.5)][0]
        onset_dict['t3_tbv'] = tb_a_t3_0_cross
        conv_tb_a, tb_a_onset2 = \
            data_process.get_onset(npy_dict['tb'][0], npy_dict['tb'][2],
                                   thaw_window=thaw_window_0,
                                   k=k0, type='tb', year0=year0)  # tb asc down vertical
        tb_a_t3_0_cross = conv_tb_a[0][(conv_tb_a[0]>tb_a_onset2) & (conv_tb_a[1] > -0.5)][0]
        onset_dict['t3_tbh'] = tb_a_t3_0_cross
        # plot t3
        if isplot:
            p = 0
            # plot_funcs.plot_subplot([npy_dict['tb'],
            #                      npy_list[9],
            #                      npy_list[2]],
            #                     [conv_tb_a, conv_tb_d, conv_sigma],
            #                     main_label=['NPR PM ($10^{-2})$', 'NPR AM ($10^{-2})$', '$\sigma^0$ (dB)'],
            #                     vline=[[tb_a_onset2, tb_a_t3_0_cross, sup_onset2],
            #                            ['k-', 'b-', 'r-'], ['tb_down', 'tb_min', 'sig_up']],  # tb down, tb min, sigma up
            #                     x_unit='sec', x_lim=npy_list[5],
            #                     figname='result_agu/result_2019/t3/20181202plot_ascat_%d_%d_tb.png' % (id_name, id0))

        # check ascat noise, angular dependency

        # plot result
        if isplot:
            # np.array([0 melt_onset0, 1 level, 2 conv_edge_onset, 3 winter_noise_std, 4 winter_noise]
            print "the current station is", id0, id_name
            conv_sigma_a, sigma_t3_up_a, sigma_melt_t1_a = \
                data_process.get_onset_new(sigma_a[0], sigma_a[1],  # onset t3, melt t1 based on single orbit
                                       thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                       doy0*3600*24 for doy0 in [60, 150]],
                                       k=k_ascat, type='sigma', melt_window=npr_a_t1, year0=year0)
            conv_sigma_d, sigma_t3_up_d, sigma_melt_t1_d = \
                data_process.get_onset_new(sigma_d[0], sigma_d[1],  # onset t3, melt t1 based on single orbit
                                       thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                       doy0*3600*24 for doy0 in [60, 150]],
                                       k=k_ascat, type='sigma', melt_window=npr_a_t1, year0=year0)
            #  % winter noise % (winter std) % edge value % level/min noise
            text_example0 = 'k: %d, %.3f (%.3f) %.3f %.3f' \
                           % (k_ascat, sigma_melt_t1_a[4], 2*sigma_melt_t1_a[3], sigma_melt_t1_a[2], sigma_melt_t1_a[1])
            text_example1 = 'k: %d, %.3f (%.3f) %.3f %.3f' \
                           % (k_ascat, sigma_melt_t1_d[4], 2*sigma_melt_t1_d[3], sigma_melt_t1_d[2], sigma_melt_t1_d[1])
            h_lines = [[0, 0, 1, 1, 2, 2],
                       [
                        sigma_melt_t1_a[1], sigma_melt_t1_a[4],
                        sigma_melt_t1_d[1], sigma_melt_t1_d[4],
                        sigma_melt_t1[1], sigma_melt_t1[4]
                       ],
                       ['--', ':', '--', ':', '--', ':']]  # -- min noise, : winter mean noise
            plot_funcs.plot_subplot([sigma_a, sigma_d, sigma_all],
                                [conv_sigma_a, conv_sigma_d, conv_sigma],
                                main_label=['$\sigma^0$ PM ', '$\sigma^0$ AM ', '$\sigma^0$ (dB)'],
                                vline=[[npr_a_t1, sigma_melt_t1[0], npr_a_t2_0_x, sig_t2_0_x],
                                       ['k-', 'r-', 'k:', 'r:'], ['npr_up', 'sig_down', 'npr_max', 'sig_min']],
                                x_unit='sec', x_lim=npy_dict['lim'],
                                figname='result_agu/result_2019/new/ascat_split_%d_%dk0.png' % (id_name, id0),
                                annote=[[0, 1, 2], [text_example0, text_example1, text_example]], h_line=h_lines,
                                    y_lim=[[0, 1, 2], [[-15, -8], [-15, -8], [-15, -8]]],
                                    y_lim2=[[0, 1, 2], [[-2, 2], [-2, 2], [-2, 2]]])

        # save result in tables
        tables[table_write, 0] = id_name
        tables_2 = tables.copy()  # results from descending orbit
        onsets_sec = np.array([npr_a_t1, sigma_melt_t1[0], npr_a_t2_0_x, sig_t2_0_x, tb_a_t3_0_cross,
                                            sigma_t3_up])
        tables[table_write, 1: -1] = onsets_sec
        tables[table_write, -1] = sigma_melt_t1[1]
        table_write += 1
        # return without save
        if is_save:
            continue
        else:
            return npy_dict, conv_dict, onset_dict, orbit_ascat
    fmt_list = []
    for item0 in tables[0]:
        fmt_list.append('%d')
    fmt_list[-1] = '%.3f'
    np.savetxt('result_agu/result_2019/table_result_%s_sec.txt' % orbit, tables,
               delimiter=',', header='station,t1,t1_b,t2,t2_b,t3,t3_b,t1_level', fmt=','.join(fmt_list))
    tran2doy('result_agu/result_2019/table_result_%s_sec.txt' % orbit,
             'result_agu/result_2019/table_result_%s_doys.txt' % orbit)


def tran2doy(txtname, savename):
    txt0 = np.loadtxt(txtname, delimiter=',')
    txt0_copy = txt0.copy()
    for i in [2, 4, 6]:
        txt0_copy[:, i] = bxy.time_getlocaltime(txt0[:, i], ref_time=[2016, 1, 1, 0])[3]
    for i in [1, 3, 5]:
        txt0_copy[:, i] = bxy.time_getlocaltime(txt0[:, i], ref_time=[2016, 1, 1, 12])[3]
    txt0_copy[:, -1] = np.abs(txt0[:, -1])
    with open(txtname, 'r') as f0:
        for row in f0:
            heads = row
            break
    np.savetxt(savename, txt0_copy, delimiter=',', header=heads, fmt='%.2f')


def amsr2_l3_comparison():
    '''
    This scripts compare the snow onets based on amsr2 with that from ascat, smap at each station.
    :return:
    '''
    # read data, include the unvalid
    table_ap = np.loadtxt('result_agu/result_2019/table0_smap_ascat.txt', delimiter=',')
    table_amsr2 = np.loadtxt('result_agu/result_2019/table1_amsr2.txt', delimiter=',')
    # get heads
    with open('result_agu/result_2019/table0_smap_ascat.txt', 'r') as f0:
        for row in f0:
            head_ap = row.split(',')
            break
    with open('result_agu/result_2019/table0_smap_ascat.txt', 'r') as f1:
        for row in f1:
            head_amsr2 = row.split(',')
    # scattering plotting
    # xx = table_amsr2[:, 1]
    symbs = ['o', '^', '*', 's', 'D', 'h']
    clrs = ['r', 'g', 'b']
    marks = [a + b for a in clrs for b in symbs]
    site_list = table_ap.T[0]
    for i_y in range(0, table_ap.shape[1]):
        label_y = head_ap[i_y]
        i_site = 0
        ax = plt.subplot2grid((3, 8), (0, 0), colspan=5, rowspan=2)
        # mask the unvalid
        yy0, xx0 = table_ap[:, i_y], table_amsr2[:, 1]
        valid_row = ((yy0<200) | (yy0>900)) & (xx0<200)
        table_ap = table_ap[valid_row]
        table_amsr2 = table_amsr2[valid_row]
        xx = table_amsr2[:, 1]
        yy = table_ap[:, i_y]
        for yy0 in yy:
            # site_label = site_infos.change_site(str(site_list[i_site].astype(int)), names=True)
            site_label = str(site_list[i_site].astype(int))
            ax.plot(xx[i_site], yy0, marks[i_site], label=site_label, markersize=10)
            i_site += 1
        ax.legend(bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0., prop={'size': 17}, numpoints=1)
        ax.set_xlabel('snow melt (amsr2) \n Day of year 2016')
        ax.set_ylabel('%s \n Day of year 2016' % label_y)
        rmse0 = np.sqrt(np.mean((xx - yy)**2))
        rmse0 = np.mean(yy - xx)
        rmse1 = np.std(yy - xx)
        text0 = 'Bias = %d$\pm$%d (days)'\
            % (rmse0.astype(int), rmse1.astype(int))
        ax.text(0.20, 0.85, text0, transform=ax.transAxes, va='top', fontsize=22)
        ax.set_xlim([60, 150])
        ax.set_ylim([60, 150])
        plt.rcParams.update({'font.size': 18})
        plt.savefig('result_08_01/air_npr_onset%s' % label_y, bbox_inches='tight')
        plt.close()
    return 0


def snow_station_amsr2():
    snowmelt_amsr2 = []
    site_nos = ['947', '968', '2213']
    s_info, s_measurements, s_secs = Read_radar.read_amsr2_l3(site_nos, prj='EQMD')
    # dimensions: ('date', 'atts', 'location', 'variables')
    s_info_d, s_measurements_d, s_secs_d = Read_radar.read_amsr2_l3(site_nos, prj='EQMD')
    # test the pass hour
    d_pass = np.abs(s_measurements_d[:, 1, 0, 0])/60
    a_pass = np.abs(s_measurements[:, 1, 0, 0])/60
    s_measurements[s_measurements < -30000] = np.nan
    for i_no, site_no in enumerate(site_nos):
        fig_name = 'result_08_01/%s_snow_amsr2.png' % site_no
        p_doy = bxy.time_getlocaltime(s_secs, ref_time=[2000, 1, 1, 0], t_out='utc')[3]
        air_measure, air_sec = read_site.get_secs_values(site_no, "Air Temperature Observed (degC)",
                                                           p_doy, nan_value=-0.5, pass_hr=13)
        if site_no in ['2065', '2081']:
            air_measure, air_sec = read_site.get_secs_values(site_no, "Air Temperature Average (degC)",
                                                           p_doy, nan_value=-0.5)
        snow_measure, snow_sec = read_site.get_secs_values(site_no, 'snow', p_doy)
        snd_50 = data_process.zero_find([s_secs, s_measurements[:, 0, i_no, 0].ravel()], th=5)
        pass_hr = np.abs(s_measurements[:, 1, i_no, 0]/60)
        # test time zone
        sec00 = bxy.get_total_sec('20160101')
        tuple00 = bxy.time_getlocaltime([sec00], ref_time=[2000, 1, 1, 0], t_out='utc')
        # plotting
        if site_no in ['947', '949', '950', '967', '1089']:
            snow_label = 'SWE (mm)'
        else:
            snow_label = 'SND (mm)'
            snow_measure *= 10
        # value: s_measurements[:, 0, i_no, 0], time: 1
        # mainaxes0 : [x0, y0, x1, y1 ...]
        plot_funcs.plot_subplot([[s_secs, s_measurements[:, 0, i_no, 0].ravel(), snow_sec, snow_measure],
                             [s_secs, s_measurements[:, 0, i_no, 1].ravel(), s_secs, pass_hr]],
                            [[air_sec, air_measure], [air_sec, air_measure]],
                            main_label=['SND (mm) vs %s' % snow_label, 'SWE (mm) vs %s' % snow_label],
                            x_unit='sec', vline=[[snd_50-4*3600*24], ['r-'], ['snow free']],
                            figname='result_08_01/2019_snow_%s_.png' % (site_no),
                            main_syb = ['k-', 'r-', 'b-'])
        snowmelt_amsr2.append(snd_50-4*3600*24)
    melt_date_amsr2 = bxy.time_getlocaltime(snowmelt_amsr2, ref_time=[2000, 1, 1, 0])[3]
    # with open('result_agu/result_2019/result_amsr2.txt', 'w') as f0:
    #     f0.writelines(site_nos)
    #     f0.write('\n')
    #     f0.writelines(melt_date_amsr2)

    np.savetxt('result_agu/result_2019/result_amsr2.txt', np.array([site_nos, melt_date_amsr2]).astype(int).T,
               delimiter=',', fmt='%d', header='id,melt_amsr2')


def compare_onset_insitu(name_situ, fromfile=False):
    path0 = 'result_agu/result_2019'
    if fromfile is not False:
        result_a = fromfile
    else:
        result_a, result_d = np.loadtxt('%s/table_result_A_sec.txt' % path0, delimiter=','), \
                             np.loadtxt('%s/table_result_D_sec.txt' % path0, delimiter=',')
    insitu_array = np.zeros([result_a.shape[0], 5, result_a.shape[1]-2])
    # get in situ measurements, inputs include s_no (int), onset (secs), measurement type (e.g. soil moisture)
    # 3d array (s_no,  statistic, timings)
    for i_site in np.arange(0, result_a.shape[0]):
        insitu_vlaue = data_process.get_insitu(result_a[i_site, 0].astype(int), # mean, max, min, standard deviation
                                               result_a[i_site, 1:-1], name_situ, window=3)

        insitu_array[i_site, 1:, :] = insitu_vlaue
        insitu_array[i_site, 0, :] = result_a[i_site, 0].astype(int)
    np.save('test_get_insitu_%s.npy' % name_situ, insitu_array)
    compare_onset_insitu_plot(name_situ, result_a=result_a, mode='site')
    return insitu_array


def in_situ_during_onsets(name_situ, fromfile=False):
    path0 = 'result_agu/result_2019'
    if fromfile is not False:
        result_a = fromfile
    else:
        result_a, result_d = np.loadtxt('%s/table_result_A_sec.txt' % path0, delimiter=','), \
                             np.loadtxt('%s/table_result_D_sec.txt' % path0, delimiter=',')
    insitu_array = np.zeros([result_a.shape[0], 4, 5])
    # 3d array (s_no,  period, statistic)
    for i_site in np.arange(0, result_a.shape[0]):
        insitu_vlaue = data_process.get_period_insitu(result_a[i_site, 0].astype(int), # mean, max, min, standard deviation
                                                      result_a[i_site, 1:-1], name_situ,
                                                      )

        insitu_array[i_site, :, 1:] = insitu_vlaue
        insitu_array[i_site, :, 0] = result_a[i_site, 0].astype(int)
    # np.save('test_get_insitu_%s.npy' % name_situ, insitu_array)
    # compare_onset_insitu_plot(name_situ, result_a=result_a, mode='site')
    return insitu_array


def compare_onset_insitu_plot(name_situ,
                              result_a=np.loadtxt('result_agu/result_2019/table_result_A_sec.txt', delimiter=','),
                              mode='all'):
    # initials
    insitu_vlaue_a = np.load('test_get_insitu_%s.npy' % name_situ)
    # insitu_vlaue_a[10, :, :] = np.nan
    path0 = 'result_agu/result_2019'
    level_index = np.abs(result_a[:, -1]) > 1.9  # site_no,
    level_index = result_a[:, -1] > 0  # site_no,
    # melt_onset0, level, conv_edge_onset, conv_winter_mean, winter_noise
    if mode == 'all':
        # plotting result: the in situ measures in 6 timings
        mean0 = np.nanmean(insitu_vlaue_a, axis=0)
        max0, min0, std0 = \
            np.nanmax(insitu_vlaue_a, axis=0), np.nanmin(insitu_vlaue_a, axis=0), np.nanstd(insitu_vlaue_a, axis=0)
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(1, 1, 1)
        x = np.arange(1, 7)
        y = mean0[0]
        yerr = np.array([min0[0], max0[0]])
        ax0.errorbar(x, y, yerr=std0[0], fmt='r^')
        # ax0.errorbar(x[level_index], y[level_index], yerr=std0[0][level_index], 'b*')
        ax0.set_xlim([0, 8])
        ax0.set_ylim([-10, 30])
        plt.grid(True)
        plt.savefig('./result_agu/result_2019/test_insitu_onset_%s.png' % name_situ)
        plt.close()

    # plot: the in situmeasurement of each station
    elif mode == 'site':
        s_name = site_infos.get_id()
        plot_timings = insitu_vlaue_a[:, 1, :]  # mean
        # plot_timings = insitu_vlaue_a[:, 2, :] - insitu_vlaue_a[:, 3, :]
        std_timings = insitu_vlaue_a[:, 4, :]
        graph_shape = (3, 2)
        site_no = insitu_vlaue_a[:, 0, 0].astype(int)

        # the x and corresponded tick label
        x_tick_labels = np.chararray(len(s_name), itemsize=4)
        tick_nos = len(s_name)  # the total number of xticks
        x_tick_labels[:] = ''
        x_int = []
        for sno in site_no:
            index0 = s_name.index(str(sno))
            x_int.append(index0)
            x_tick_labels[index0] = s_name[index0]
        x = np.array(x_int)
        # x = np.arange(0, insitu_vlaue_a.shape[0])
        timing_text = ['snowmelt_npr', 'snowmelt_$\sigma^0$','max_npr', 'mi_$\sigma^0$',
                       'snowfree_tb', 'snowfree_$\sigma^0$']
        for i_a in range(0, 6):
            n1 = i_a/graph_shape[1]
            n2 = i_a - n1*graph_shape[1]
            ax = plt.subplot2grid(graph_shape, (n1, n2))
            ax.errorbar(x, plot_timings[:, i_a], yerr=std_timings[:, i_a], fmt='r^')
            ax.errorbar(x[level_index], plot_timings[:, i_a][level_index],
                        yerr=std_timings[:, i_a][level_index], fmt='b*')
            # p0 = ax.bar(x, mean_timings[:, i_a], 2, yerr=std_timings[:, i_a])
            label_ax = timing_text[i_a]
            ax.text(0.55, 0.85, label_ax, transform=ax.transAxes, va='top', fontsize=16)
            ax.set_xlim([0, 20])
            ax.set_ylim([-10, 40])
            plt.xticks(np.arange(tick_nos), x_tick_labels, rotation=90)
            plt.grid(True)
        plt.savefig('./result_agu/result_2019/diff_station_onset_%s.png' % name_situ)
        plt.close()


def period_insitu_plot(insitu_vlaue_a, year0, period_no):
    """

    :param insitu_vlaue_a: the 4d array record (measurement: sm, tsoil, tair, snow), site, period, statistic
    :param year0:
    :return:
    """
    onset_doy = np.loadtxt('result_agu/result_2019/onset_secs_%d.txt' % year0, delimiter=',')
    level_index = onset_doy[:, -1] > 0
    s_name = onset_doy[:, 0]
    valid_index = onset_doy[:, -1] > -99
    if year0 == 2018:
        for unval in [1233, 2211, 2212, 2213]:
            valid_index[s_name==unval] = False
    elif year0 == 2017:
        for unval in [1233, 2211, 2212, 2213]:
            valid_index[s_name==unval] = False
    if period_no == 3:
        valid_index = valid_index & level_index
        valid_index[s_name==949] = False
        valid_index[s_name==1090] = False
    plot_timings = insitu_vlaue_a[:, :, period_no, 1]  # mean
    std_timings = insitu_vlaue_a[:, :, period_no, 4]

    graph_shape = (2, 2)
    site_no = insitu_vlaue_a[0, :, 0, 0].astype(int)

    # the x and corresponded tick label
    x_tick_labels = np.chararray(s_name.size, itemsize=4)
    tick_nos = s_name.size  # the total number of xticks
    x_tick_labels[:] = ''
    x_int = []
    for sno in site_no:
        index0 = np.where(s_name == sno)[0][0]
        x_int.append(index0)
        x_tick_labels[index0] = s_name[index0]
    x = np.array(x_int)
    # x = np.arange(0, insitu_vlaue_a.shape[0])
    timing_text = ['sm', 'tsoil', 't_air', 'snow']
    for i_a in range(0, 4):
        n1 = i_a/graph_shape[1]
        n2 = i_a - n1*graph_shape[1]
        ax = plt.subplot2grid(graph_shape, (n1, n2))
        ax.errorbar(x[valid_index], plot_timings[i_a][valid_index], yerr=std_timings[i_a][valid_index], fmt='r^')
        ax.errorbar(x[level_index & valid_index], plot_timings[i_a][level_index & valid_index],
                    yerr=std_timings[i_a][level_index & valid_index], fmt='b*')
        # p0 = ax.bar(x, mean_timings[:, i_a], 2, yerr=std_timings[:, i_a])
        label_ax = timing_text[i_a]
        ax.text(0.55, 0.85, label_ax, transform=ax.transAxes, va='top', fontsize=16)
        ax.set_xlim([0, 20])
        plt.xticks(np.arange(tick_nos), x_tick_labels, rotation=90)
        plt.grid(True)
    plt.savefig('./result_agu/result_2019/diff_station_period_%d_%d.png' % (period_no, year0))
    plt.close()


def compare_stational(all_site=True, y=2016, id_check=[968]):
    '''
    call drawing_parameter_v2
    plot npr(t1, t2), ascat(t1, t2, t3) and tb(t3), each of which are compared with in situ meausrements
    :return: compare the onset with corresponded measurements
    '''
    land_ids = np.loadtxt('result_agu/result_2019/points_num.txt', delimiter=',')
    bool_ind = []
    for id00 in land_ids[1]:
        if id00 in id_check:
            bool_ind.append(True)
        else:
            bool_ind.append(False)
    l_id = land_ids[:, bool_ind]
    if all_site:
        l_id = land_ids
    onset_out = []
    for sno in l_id.T:
        site_no = sno[1].astype(int)
        if sno[0] == 5858:
            continue
        m_name = site_infos.in_situ_vars(sno[1].astype(int))
        if site_no in [2065, 2081]:
            air_measure = 'Air Temperature Average (degC)'
        else:
            air_measure = 'Air Temperature Observed (degC)'
        secs_insitu_0 = bxy.get_total_sec('%d0101' % y)
        doy_insitu = np.arange(1, 365)
        sec_insitu = secs_insitu_0 + (doy_insitu-1)*24*3600
        check_doy = bxy.time_getlocaltime(sec_insitu, ref_time=[2000, 1, 1, 0], t_out='utc')  # 366+np.arange(1, 365)
        m_value, m_sec = read_site.read_measurements_v2(str(sno[1].astype(int)), m_name, sec_insitu, year0=y, hr=18)
        # get npy_dict, conv_dict, onset_dict, orbit_ascat
        mw_site, conv_site, onset_es, orbit_ascat = \
            drawing_parameter_v2(sno, orbit='A', is_save=False, ks=[7, 7, 7], isplot=False, year0=y)

        # save onset estimates in an
        onset_out0 = np.zeros(8)
        onset_out0[0] = sno[1]
        onset_out0[1:] = np.array([onset_es['t1_npr'], onset_es['t1_ascat'][0],
                                    onset_es['t2_npr'], onset_es['t2_ascat'],
                                    onset_es['t3_tbv'], onset_es['t3_ascat'], onset_es['t1_ascat'][1]])
        onset_out.append(onset_out0)
        print 'the remote sensing measurements are: ', mw_site.keys()
        # compare estimate and in situ
        if 'Snow Water Equivalent (mm)' in m_value.keys():
            snow_text = 'Snow Water Equivalent (mm)'
        else:
            snow_text = 'Snow Depth (cm)'
        window0 = [bxy.get_total_sec('%d0101' % year0) + day_i*24*3600 for day_i in [0, 60]]
        window1 = [bxy.get_total_sec('%d0101' % year0) + day_i*24*3600 for day_i in [150, 210]]
        npr_0, npr_1 = np.nanmean(mw_site['npr'][1][(mw_site['npr'][0]>window0[0]) & (mw_site['npr'][0]<window0[1])]), \
                       np.nanmean(mw_site['npr'][1][(mw_site['npr'][0]>window1[0]) & (mw_site['npr'][0]<window1[1])])
        # npr t1 and t2
        plot_funcs.plot_subplot([mw_site['npr'],
                                 [m_sec, m_value['Soil Temperature Observed -2in (degC)']],
                                 [m_sec, m_value[air_measure]]],
                                [conv_site['npr'],
                                 [m_sec, m_value['Soil Moisture Percent -2in (pct)']],
                                 [m_sec, m_value[snow_text]]], main_label=['npr', 'Tsoil', 'Tair'],
                                figname='result_agu/result_2019/new/%d_estimate_%s.png' % (sno[1], 'npr'),
                                vline=[[onset_es['t1_npr'], onset_es['t2_npr']], ['r:', 'r-'], ['t1_npr', 't2_npr']],
                                h_line2=[[0, 0, 2], [npr_0, npr_1, 0], [':', ':', '--']],
                                y_lim=[[0], [[0, 5]]],
                                x_unit='doy')
        # ascat t1, t2, and t3
        # site melt_onset0, level, conv_edge_onset, conv_winter_mean, winter_noise
        text_example0 = 'orb_%s %.3f (%.3f),  %.3f (%.3f)' \
                       % (orbit_ascat, onset_es['t1_ascat'][-1], 2*onset_es['t1_ascat'][3],
                          onset_es['t1_ascat'][2], onset_es['t1_ascat'][1])
        plot_funcs.plot_subplot([mw_site['ascat'],
                                 [m_sec, m_value['Soil Temperature Observed -2in (degC)']],
                                 [m_sec, m_value[air_measure]]],
                                [conv_site['ascat'],
                                 [m_sec, m_value['Soil Moisture Percent -2in (pct)']],
                                 [m_sec, m_value[snow_text]]], main_label=['ascat', 'Tsoil', 'Tair'],
                                figname='result_agu/result_2019/new/%d_estimate_%s.png' % (sno[1], 'ascat'),
                                vline=[[onset_es['t1_ascat'][0], onset_es['t2_ascat'], onset_es['t3_ascat']],
                                       ['r:', 'b-', 'r-'], ['t1_ascat', 't2_ascat', 't3_ascat']],
                                x_unit='doy',
                                annote=[[0], [text_example0]])
        # tb t3
        plot_funcs.plot_subplot([[mw_site['tb'][0], mw_site['tb'][1], mw_site['tb'][0], mw_site['tb'][2]],
                                 [m_sec, m_value['Soil Temperature Observed -2in (degC)']],
                                 [m_sec, m_value[air_measure]]],
                                [conv_site['ascat'],
                                 [m_sec, m_value['Soil Moisture Percent -2in (pct)']],
                                 [m_sec, m_value[snow_text]]], main_label=['tb', 'Tsoil', 'Tair'],
                                figname='result_agu/result_2019/new/%d_estimate_%s.png' % (sno[1], 'tb'),
                                vline=[[onset_es['t3_tbv'], onset_es['t3_tbh']],
                                       ['r:', 'r-'], ['t3_tbv', 't3_tbh']],
                                x_unit='doy')
    return np.array(onset_out)


def in_situ_tsoil_sm_snd(year0=2016):
    # 0309 read in situ measurement, get onsets based on them
    site_nos = site_infos.get_id(mode='int')
    tsoil_name = "Soil Temperature Observed -2in (degC)"
    snd_name = "snow"
    zero_date_array = np.zeros([site_nos.size, 5])
    i0 = 0
    for sno0 in site_nos:
        sno = str(sno0)
        print 'the %d was processing' % sno0
        t_value, t_date = read_site.read_measurements\
            (sno, tsoil_name, np.arange(1, 365), year0=year0, hr=18, t_unit='sec')
        snow_value, snow_date = read_site.read_measurements\
            (sno, snd_name, np.arange(1, 365), year0=year0, hr=0, t_unit='sec')
        tsoil_zero_day = data_process.zero_find(np.array([t_date, -t_value]), w=5, th=0)
        sm_value, sm_date = read_site.read_measurements\
            (sno, "Soil Moisture Percent -2in (pct)", np.arange(1, 365), year0=year0, hr=18, t_unit='sec')
        # valid_snow_i = bxy.reject_outliers(snow_value, m=4)
        # snow_date, snow_value = snow_date[valid_snow_i], snow_value[valid_snow_i]
        snd_zero_day = data_process.zero_find(np.array([snow_date, snow_value]), w=5, th=5)
        zero_doy = bxy.time_getlocaltime([tsoil_zero_day, snd_zero_day])[-2]
        zero_date_array[i0] = np.array([sno0, tsoil_zero_day, snd_zero_day, zero_doy[0], zero_doy[1]])
        i0 += 1
        # plot test
        if sno0 in site_nos:
            # ts = bxy.get_total_sec('%d0101' % year0) + 24*3600*np.array([t_date, snow_date])
            ts = np.array([t_date, snow_date, sm_date])
            sm_lim = [[2], [[0, 75]]]
            plot_funcs.plot_subplot([[ts[0], t_value], [ts[1], snow_value], [ts[2], sm_value]],
                                    [[ts[0], t_value], [ts[1], snow_value], [ts[2], sm_value]],
                                    main_label=['$T_{soil} (^oC)$', 'SND (cm)', 'VWC (%)'],
                                    figname='result_agu/test_insitu_zero%d.png' % sno0, h_line=[[0], [0], [':']],
                                    vline=[[tsoil_zero_day, snd_zero_day], ['r-', 'r:'], ['t>0', 'snow=0']],
                                    x_unit='sec', y_lim=sm_lim, y_lim2=sm_lim)
    np.savetxt('./result_agu/result_2019/tsoil_snd_all.txt', zero_date_array, delimiter=',', fmt='%d')
    return 0


def in_situ_tair_snd(sno0, year0=2016, npr_date=-1, ascat_date=-1):
    """
    first, determine the date when air temperautre above 0. Then get the snow, air temperature variation after this
    date, to a given date such as npr statrs to increase
    :param year0:
    :param npr_date: the date when npr starts to increase
    :param ascat_date: the date when ascat starts to decrease
    :return: the time series of t_air and snd during two periods: t_air >0 -- npr thawing date, and t_air>0 -- ascat
    snow date
    """
    if npr_date < 0:
        npr_date = 100*24*3600 + get_ascat_sec('%d0101' % year0)
    if ascat_date < 0:
        ascat_date = 100*24*3600 + get_ascat_sec('%d0101' % year0)
    snd_name = "snow"
    print 'the %d was processing' % sno0
    sno = str(sno0)
    tair_name = "Air Temperature Observed (degC)"
    if sno0 in [2065, 2081]:
        if year0 == 2016:
            tair_name = "Air Temperature Average (degC)"
    t_value, t_date = read_site.read_measurements\
        (sno, tair_name, np.arange(1, 365), year0=year0, hr=18, t_unit='sec')


    tair_zero_day2 = data_process.zero_find(np.array([t_date, -t_value]), w=7, th=0)  # in unit of sec
    tair_zero_day1 = data_process.zero_find_gt(np.array([t_date, t_value]), w=7, th=1)
    air_win = 7  # check days during window shown air temperature gt 0 degC
    w, w_valid = data_process.n_convolve3(t_value, air_win)
    air0_index0 = np.where(w>5)
    for ind0 in air0_index0[0]:
        if t_date[ind0] > bxy.get_total_sec('%d0307' % year0):
            tair_zero_day = t_date[ind0] - air_win*24*3600
            break
    # check
    zero_date = bxy.time_getlocaltime([tair_zero_day,tair_zero_day2, npr_date[0], ascat_date[0]],
                                      ref_time=[2000, 1, 1, 0], t_source="US/Alaska")[-2]
    i_zero = np.where(bxy.time_getlocaltime(t_date, ref_time=[2000, 1, 1, 0],
                                            t_source="US/Alaska")[-2] == zero_date[0])[0][0]
    t_check = t_value[i_zero - 3: i_zero + 4]
    air_0, air00 = read_site.read_measurements(sno, tair_name, 366+np.arange(50, 70), hr=18)
    a_extend = np.array([-3600*24, 3600*24])
    period0, period1 = np.array(sorted([tair_zero_day, npr_date])) + a_extend, \
                       np.array(sorted([tair_zero_day, ascat_date])) + a_extend
    snow_value, snow_date = read_site.read_measurements\
        (sno, snd_name, np.arange(1, 365), year0=year0, hr=0, t_unit='sec')
    # get the in situ measurements during a period
    snow2date0 = data_process.measurements_slice(np.array([snow_date, snow_value]),
                                                 peroid=period0)
    snow2date1 = data_process.measurements_slice(np.array([snow_date, snow_value]),
                                                 peroid=period1)
    air2date0, air2date1 = data_process.measurements_slice(np.array([t_date, t_value]),
                                                           peroid=period0),\
                           data_process.measurements_slice(np.array([t_date, t_value]),
                                                           peroid=period1)
    return tair_zero_day, snow2date0, snow2date1, air2date0, air2date1


def rs_series(p_ids=np.array([3770]), t_window=[0, 366]):
    smap_att0 = ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft']
    smap_dict_2016 = data_process.get_smap_dict(np.arange(t_window[0], t_window[1]))
    smap_dict_2017 = data_process.get_smap_dict(np.arange(t_window[0], t_window[1]), y=2017)
    smap_dict_2018 = data_process.get_smap_dict(np.arange(t_window[0], t_window[1]), y=2018)
    smap_dict_all = data_process.dict_concatenate(smap_att0, smap_dict_2016[0], smap_dict_2017[0], smap_dict_2018[0])

    ascat_att0 = ['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore', 'inc_angle_trip_fore',
                  'sigma0_trip_mid', 'inc_angle_trip_mid']
    ascat_dict_2016 = data_process.get_ascat_dict_v2(np.arange(t_window[0], t_window[1]), p_ids=p_ids,
                                                     ascat_atts=ascat_att0, file_path='ascat_resample_all2')
    ascat_dict_2017 = data_process.get_ascat_dict_v2(np.arange(t_window[0], t_window[1]), p_ids=p_ids, y=2017,
                                                     ascat_atts=ascat_att0, file_path='ascat_resample_all2')
    ascat_dict_2018 = data_process.get_ascat_dict_v2(np.arange(t_window[0], t_window[1]), p_ids=p_ids, y=2018,
                                                     ascat_atts=ascat_att0, file_path='ascat_resample_all2')
    ascat_dict_all = data_process.dict_concatenate(ascat_dict_2016.keys(), ascat_dict_2016, ascat_dict_2017, ascat_dict_2018)
    # remove of angular dependency
    return ascat_dict_all, smap_dict_all


def get_smap_series(p_ids=np.array([3770]), t_window=[0, 366], smap_att0=[]):
    """
    the result i sa nd-array, include: all pixels in alaska,
    :param smap_att0: default empty, set in the context
    :param t_window:
    :return:
    """
    smap_att0 = ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft']
    smap_dict_2016 = data_process.get_smap_dict(np.arange(t_window[0], t_window[1]))
    smap_dict_2017 = data_process.get_smap_dict(np.arange(t_window[0], t_window[1]), y=2017)
    smap_dict_2018 = data_process.get_smap_dict(np.arange(t_window[0], t_window[1]), y=2018)
    smap_dict_all = data_process.dict_concatenate(smap_att0, smap_dict_2016[0], smap_dict_2017[0], smap_dict_2018[0])
    np.savez('smap_data0.npz', **smap_dict_all)
    return smap_dict_all


def get_ascat_series(p_ids=np.array([3770]), t_window=[0, 366], ascat_att0=[]):
    '''
    the result is a nd-array, include: number of smap pixels (not all the pixels in alaska due to limited disk space),
    9 nns of the smap pixels, all data from 3 years are saved. the nd-array can be reshaped in -1, 9, time_length
    :param p_ids:
    :param t_window:
    :param ascat_att0: default empty, set in the context
    :return:
    '''
    ascat_att0 = ['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore', 'inc_angle_trip_fore',
                  'sigma0_trip_mid', 'inc_angle_trip_mid']
    ascat_dict_2016 = data_process.get_ascat_dict_v2(np.arange(t_window[0], t_window[1]), p_ids=p_ids,
                                                     ascat_atts=ascat_att0, file_path='ascat_resample_all2')
    ascat_dict_2017 = data_process.get_ascat_dict_v2(np.arange(t_window[0], t_window[1]), p_ids=p_ids, y=2017,
                                                     ascat_atts=ascat_att0, file_path='ascat_resample_all2')
    ascat_dict_2018 = data_process.get_ascat_dict_v2(np.arange(t_window[0], t_window[1]), p_ids=p_ids, y=2018,
                                                     ascat_atts=ascat_att0, file_path='ascat_resample_all2')
    ascat_dict_all = data_process.dict_concatenate\
        (ascat_dict_2016.keys(), ascat_dict_2016, ascat_dict_2017, ascat_dict_2018)
    start0 = bxy.get_time_now()
    # data_process.angular_effect(ascat_dict_all, 'inc_angle_trip_aft', 'sigma0_trip_aft')
    # data_process.angular_effect(ascat_dict_all, 'inc_angle_trip_fore', 'sigma0_trip_fore')
    # data_process.angular_effect(ascat_dict_all, 'inc_angle_trip_mid', 'sigma0_trip_mid')
    start1 = bxy.get_time_now()
    print("----angular correct part: %s seconds ---" % (start1-start0))
    np.savez('ascat_data0.npz', **ascat_dict_all)
    return ascat_dict_all


def in_situ_series(sno, y=2016, air_measure='air'):
    m_name = site_infos.in_situ_vars(sno)
    if air_measure == 'air':
        if sno in [2065, 2081]:
            air_measure = 'Air Temperature Average (degC)'
        else:
            air_measure = 'Air Temperature Observed (degC)'
    secs_insitu_0 = bxy.get_total_sec('%d0101' % y)
    doy_insitu = np.array([np.arange(1, 366), np.arange(1, 366), np.arange(1, 366)]).T.ravel()
    hr_array0 = np.zeros([doy_insitu.size/3, 1])
    hr_array = np.matmul(hr_array0, np.array([[7, 14, 18]]))
    air_value_tr, air_date_tr = \
        read_site.read_measurements(str(sno), air_measure, doy_insitu, hr=hr_array.ravel(), t_unit='sec', year0=y)
    # check x time
    x_time = bxy.time_getlocaltime(air_date_tr, ref_time=[2000, 1, 1, 0], t_source='US/Alaska')
    air_value, air_date = air_value_tr.reshape(-1, 3), air_date_tr.reshape(-1, 3)  # Col 0, 1, 2: hr7, hr14 hr18
    # air_win = 7
    # w, w_valid = data_process.n_convolve3(air_value[0], air_win)
    # air0_index0 = np.where(w>5)
    # for ind0 in air0_index0[0]:
    #     if air_date[ind0] > bxy.get_total_sec('%d0307' % year0):
    #         tair_zero_day = air_date[ind0] - air_win*24*3600
    #         break
    return np.array([air_date, air_value])


def data_prepare(sensor=0):
    if sensor == 0:
        # # data preparing ascat new, updated 2019 03 05
        lat_gd, lon_gd = spt_quick.get_grid()
        # metopB 2017
        year_no = 2017
        doy_startB = bxy.get_doy(['20160101'], year0=year_no)
        doy_startA = bxy.get_doy(['20160101'], year0=year_no)
        doy_endB = bxy.get_doy(['20170525'], year0=year_no)
        for doy0 in np.arange(doy_startB, doy_endB):
            status = Read_radar.read_ascat_alaska(doy0, year0=2017, sate='B', mode='Tibet')
            if status == -1:  # not nc data for this specific date
                continue
            t_str = bxy.doy2date(doy0, fmt="%Y%m%d", year0=year_no)
            spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
            spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
            # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
            # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
        # metopA 2017
        for doy0 in np.arange(doy_startA, doy_endB):
            status = Read_radar.read_ascat_alaska(doy0, year0=year_no, sate='A', mode='Tibet')
            if status == -1:  # not nc data for this specific date
                continue
            t_str = bxy.doy2date(doy0, fmt="%Y%m%d", year0=year_no)
            spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
            spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, sate='A', format_ascat='h5')
            # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
            # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
    elif sensor==1:
        # data preparing smap
        t_window = ['2018.06.09', '2018.09.06']
        doy_array, year_no = bxy.get_doy_array(t_window[0], t_window[1], fmt='%Y.%m.%d')
        Read_radar.radar_read_alaska('_D_', ['alaska'], t_window, 'vv', year=year_no)
        Read_radar.radar_read_alaska('_A_', ['alaska'], t_window, 'vv', year=year_no)
        for doy0 in doy_array:
            t_str=bxy.doy2date(doy0, fmt='%Y%m%d', year0=year_no)
            spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap_area_result', orbit='A')
            spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap_area_result', orbit='D')


def dict2npz():
    return 0


def tri_years_plotting(win=[30, 50], id0=np.array([])):
    """
    read 3 year data, and saved in npz files.
    :param win:
    :return:
    """
    pixel_num = np.loadtxt('result_agu/result_2019/points_num_tp.txt', delimiter=',')
    if id0.size<1:
        id0 = np.array([960, 2065, 968, 1090, 947, 2211, 2213, 1175]).astype(int)
    id_num = np.array([pixel_num[0][pixel_num[1] == i00][0] for i00 in id0]).astype(int)
    start0 = bxy.get_time_now()
    # ascat, smp = rs_series(p_ids=id_num, t_window=win)
    ascat = get_ascat_series(p_ids=id_num, t_window=win)
    smp = get_smap_series(p_ids=id_num, t_window=win)
    start1 = bxy.get_time_now()
    print("----read interested rs data for 3 years: %s seconds ---" % (start1-start0))
    # ascat angular
    # data_process.angular_effect(ascat, 'inc_angle_trip_aft', 'sigma0_trip_aft')
    # data_process.angular_effect(ascat, 'inc_angle_trip_fore', 'sigma0_trip_fore')
    # data_process.angular_effect(ascat, 'inc_angle_trip_mid', 'sigma0_trip_mid')
    # # save dict in npz
    # np.savez('ascat_data0.npz', **ascat)
    i2 = -1
    for t_station in id0:
        i2 += 1
        # ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft']
        # ['sigma0_trip_mid', 'inc_angle_trip_mid', 'utc_line_nodes']
        id01 = id0 == t_station
        print 't_station is %d' % (t_station)
        num01 = id_num[id0 == t_station]
        print 'num01 is %d' % (num01)
        # check overpass timing of 9 pixels
        # ascat['utc_line_nodes'][0: 9]
        dis = ascat['distance'][id01].copy()[:, 0:9][0]  # distance between smap and 9 ascat pixels, small to large
        mean_aft_series = \
            data_process.distance_interpolate(dis, i2, ascat['inc_angle_trip_aft'], ascat['sigma0_trip_aft_40'])
        mean_fore_series = \
            data_process.distance_interpolate(dis, i2, ascat['inc_angle_trip_fore'], ascat['sigma0_trip_fore_40'])
        mean_mid_series = \
            data_process.distance_interpolate(dis, i2, ascat['inc_angle_trip_mid'], ascat['sigma0_trip_mid_40'])

        # mid_sigma, shp0 = ascat['sigma0_trip_aft_40'].copy(), ascat['sigma0_trip_aft_40'].shape
        # inc_af, inc_f = ascat['inc_angle_trip_aft'], ascat['inc_angle_trip_fore']
        # mid_sigma.shape = -1, 9, shp0[1]
        # sigma_01, inc_01, t_01 = mid_sigma[id01, 0:3, :], inc_af[id01, 0:3, :], inc_f[id01, 0:3]

        # mid_sigma[]
        valid_i1 = (mean_aft_series > -900) & (mean_aft_series < -0.1)
        # ascat_value = ascat['sigma0_trip_aft_40'][0+i2*9, :][valid_i1]
        ascat_value = mean_aft_series[valid_i1]
        ascat_secs = ascat['utc_line_nodes'][0+i2*9, :][valid_i1]

        valid_i1 = (mean_mid_series > -900) & (mean_mid_series < -0.1)
        # ascat_value = ascat['sigma0_trip_aft_40'][0+i2*9, :][valid_i1]
        ascat_value_m = mean_mid_series[valid_i1]
        ascat_secs_m = ascat['utc_line_nodes'][0+i2*9, :][valid_i1]

        valid_i1 = (mean_fore_series > -900) & (mean_fore_series < -0.1)
        # ascat_value = ascat['sigma0_trip_aft_40'][0+i2*9, :][valid_i1]
        ascat_value_f = mean_fore_series[valid_i1]
        ascat_secs_f = ascat['utc_line_nodes'][0+i2*9, :][valid_i1]

        # get smap_value
        valid_i0 = smp['cell_tb_v_aft'][num01] > -90
        t_valid = smp['cell_tb_time_seconds_aft'][num01][valid_i0] + 12*3600
        npr_value = (smp['cell_tb_v_aft'][num01][valid_i0] - smp['cell_tb_h_aft'][num01][valid_i0])/\
                    (smp['cell_tb_v_aft'][num01][valid_i0] + smp['cell_tb_h_aft'][num01][valid_i0])
        # get in situ measurement
        tair0, tair1, tair2 = in_situ_series(t_station), in_situ_series(t_station, y=2017), in_situ_series(t_station, y=2018)
        t_air_all = np.concatenate((tair0, tair1, tair2), axis=1)  # dimension: att(x y), date, pass hr(7 14 18)
        # plot 3 year data

        air_plot = \
            [t_air_all[0, :, 0], t_air_all[1, :, 0],
             t_air_all[0, :, 1], t_air_all[1, :, 1],
             t_air_all[0, :, 2], t_air_all[1, :, 2]]
        # save plotting results
        np.savetxt('%d_npr_%d.txt' % (num01, t_station), [t_valid, npr_value])
        np.savetxt('%d_ascat_a_%d.txt' % (num01, t_station), [ascat_secs, ascat_value])
        np.savetxt('%d_ascat_f_%d.txt' % (num01, t_station), [ascat_secs_f, ascat_value_f])
        np.savetxt('%d_ascat_m_%d.txt' % (num01, t_station), [ascat_secs_m, ascat_value_m])
        np.savetxt('%d_air_%d.txt' % (num01, t_station), air_plot)


def get_3_year_insitu(t_station, m_name='air'):
    """

    :param t_station:
    :param m_name:
    :return: ndarray with shape equals (period length , 2*number of period). For every two elements, the 1st is the time in unit of secs,
    the other is the value.
    """
    # get in situ measurement
    # pixel_num = np.loadtxt('result_agu/result_2019/points_num_tp.txt', delimiter=',')
    # num01 = pixel_num[0][pixel_num[1] == t_station][0].astype(int)
    tair0, tair1, tair2 = in_situ_series(t_station, air_measure=m_name), \
                          in_situ_series(t_station, y=2017, air_measure=m_name), \
                          in_situ_series(t_station, y=2018, air_measure=m_name)
    t_air_all = np.concatenate((tair0, tair1, tair2), axis=1)  # dimension: att(x y), date, pass hr(7 14 18)
    # plot 3 year data
    air_plot = \
        [t_air_all[0, :, 0], t_air_all[1, :, 0],
         t_air_all[0, :, 1], t_air_all[1, :, 1],
         t_air_all[0, :, 2], t_air_all[1, :, 2]]
    # np.savetxt('result_08_01/plot_data/%d_%s_%d.txt' % (num01, m_name, t_station), air_plot)
    return np.array(air_plot)


def read_smap_series(num0s=[4547]):
    num01 = num0s[0]
    smp = np.load('smap_data0.npz')
    tbv, tbh = smp['cell_tb_v_aft'][num01], smp['cell_tb_h_aft'][num01]
    valid_i0 = (tbv > -90) & (tbv > tbh)
    t_valid = smp['cell_tb_time_seconds_aft'][num01][valid_i0] + 12*3600
    npr_value = (tbv[valid_i0] - tbh[valid_i0])/\
                (tbv[valid_i0] + tbh[valid_i0])
    return np.array([t_valid, npr_value])


def get_regional_npr(num01=[4547]):
    smp = np.load('smap_data0.npz')
    tbv, tbh = smp['cell_tb_v_aft'][num01], smp['cell_tb_h_aft'][num01]
    t_valid = smp['cell_tb_time_seconds_aft'][num01] + 12*3600


def data_prepare_ascat(p0=['20170101', '20171231'], un_grid=True, grid=True):
    '''
    :param p0:
    :param un_grid: Default True, indicate the un_grid data for a region/pixel has been saved
    :param grid: Default True, indicate the gridded data for a region on a given date and passing hour is saved
    :return:
    '''
    # # data preparing ascat new, updated 2019 03 05
    lat_gd, lon_gd = spt_quick.get_grid()
    # metopB 2017
    doy_list, year_list = bxy.get_doy_v2(p0)
    doy_st, doy_en, year_no = doy_list[0], doy_list[1], year_list[0]
    # check for a specified date
    if un_grid:
        for doy0 in np.arange(doy_st, doy_en):
            status = Read_radar.read_ascat_alaska(doy0, year0=year_no, sate='B')
            status = Read_radar.read_ascat_alaska(doy0, year0=year_no, sate='A')
    if grid:
        for doy0 in np.arange(doy_st, doy_en):
            # if status == -1:  # not nc data for this specific date
            #     continue
            t_str = bxy.doy2date(doy0, fmt="%Y%m%d", year0=year_no)
            spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')  # as, metopB
            spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')  # des, metopB
            spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, sate='A', format_ascat='h5')  # ascending
            spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')  # descending

            # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
            # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
        # metopA 2017
        # for doy0 in np.arange(doy_st, doy_en):
        #     if status == -1:  # not nc data for this specific date
        #         continue
        #     t_str = bxy.doy2date(doy0, fmt="%Y%m%d", year0=year_no)
        #     spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
        #     spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, sate='A', format_ascat='h5')
            # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
            # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')


def data_prepare_ak(p0=['20170101', '20171231']):
    # metopB 2017
    doy_list, year_list = bxy.get_doy_v2(p0)
    doy_st, doy_en, year_no = doy_list[0], doy_list[1], year_list[0]
    # for doy0 in np.arange(doy_st, doy_en):
    #     status = Read_radar.read_ascat_alaska(doy0, year0=year_no, sate='B')
    #     if status == -1:  # not nc data for this specific date
    #         continue
    for doy0 in np.arange(doy_st, doy_en):
        status = Read_radar.read_ascat_alaska(doy0, year0=year_no, sate='A')
        if status == -1:  # not nc data for this specific date
            continue
    return 0


def data_prepare_grid(p0=['20170101', '20171231']):
    lat_gd, lon_gd = spt_quick.get_grid()
    doy_list, year_list = bxy.get_doy_v2(p0)
    doy_st, doy_en, year_no = doy_list[0], doy_list[1], year_list[0]
    for doy0 in np.arange(doy_st, doy_en):
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d", year0=year_no)
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, sate='A', format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
    return 0


def ascat_all_mode_series(sno=947):
    path = '/home/xiyu/PycharmProjects/R3/result_08_01/plot_data'
    pixel_num = np.loadtxt('result_agu/result_2019/points_num.txt', delimiter=',').astype(int)
    id0 = np.array([960, 2065, 968, 1090, 947, 2211, 2213, 1175]).astype(int)
    id_num = pixel_num[0][np.in1d(pixel_num[1], id0)].astype(int)

    pixel_no = pixel_num[0][pixel_num[1] == sno][0]
    air = np.loadtxt('%s/%d_air_%d.txt' % (path, pixel_no, sno))
    snow = np.loadtxt('%s/%d_snow_%d.txt' % (path, pixel_no, sno))
    # npr = np.loadtxt('%s/%d_npr_%d.txt' % (path, pixel_no, sno))
    npr = read_smap_series([pixel_no])
    sigma_f, sigma_m, sigma_af = np.loadtxt('%s/%d_ascat_f_%d.txt' % (path, pixel_no, sno)), \
                                 np.loadtxt('%s/%d_ascat_m_%d.txt' % (path, pixel_no, sno)), \
                                 np.loadtxt('%s/%d_ascat_a_%d.txt' % (path, pixel_no, sno))
    plot_funcs.plot_subplot([npr,
                             [sigma_f[0], sigma_f[1], sigma_m[0], sigma_m[1], sigma_af[0], sigma_af[1]],
                             air],
                         [[], [], snow], main_label=['npr', '$\sigma^0$ \n (r: fore, b: mid, k: aft)', '$T_{air}$'],
                        figname='result_agu/result_2019/new/%d_estimate_%d.png' % (pixel_no, sno),
                        h_line=[[2], [0], ['--']],
                        x_unit='mmdd')
    return 0


def combine_alaska_melt():
    s_infos = [['947', 65.12422, -146.73390], ['949', 65.07833, -145.87067],
             ['950', 64.85033, -146.20945], ['1090', 65.36710, -146.59200], ['960', 65.48, -145.42],
             ['962', 66.74500, -150.66750], ['1233', 59.82001, -156.99064], ['2213', 65.40300, -164.71100],
             ['2081', 64.68582, -148.91130], ['2210', 65.19800, -156.63500], ['2211', 63.63900, -158.03010],
             ['2212', 66.17700, -151.74200], ['1175', 67.93333, -162.28333],
             ['2065', 61.58337, -159.57708], ['967', 62.13333, -150.04167], ['968', 68.61683, -149.30017],
             ['1177', 70.26666, -148.56666]]
    s_infos = [[2563, 61.22422, -162.63390], [1, 60.90, -162.01]]  # [0, 66.22422, -146.73390]
    # s_info=[4265, -64.3, -155.80]
    ## 20181104, the level, conv, two time series in regions where no melt event was detected
    site_nos = ['947', '949', '950', '960', '962', '967', '968','1090','1175',
                '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'] #'1089'
    # site_nos = ['947']
    site_nos_int = [int(str0) for str0 in site_nos]
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    points_info = []
    points_index = []
    for sno in site_nos:
        s_info = site_infos.change_site(sno)
        points_info.append(s_info)
        dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
        p_index = np.argmin(dis_1d)  # nearest
        # temp check the distance of neighbor pixels
        nn_index = np.argsort(dis_1d)[0:9]
        # if sno == '947':
        #     p_index = 4547
        points_index.append(p_index)
    odd_plot_in_map([-1], s_info=['947', -65.12422, -146.73390])  # plot the map
    land_id = points_index
    land_ids=np.array(points_index)
    # np.savetxt('result_agu/result_2019/points_num.txt', np.array([land_ids, site_nos_int]), delimiter=',', fmt='%d')
    combining2(np.arange(0, 360), pixel_plot=False, onset_save=True, land_id=land_ids, id_name=np.array(site_nos_int),
               ascat_atts=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes'])


def get_yearly_ascat(p_ids=np.array([3770]), t_window=[0, 210], path0='./result_08_01/series/ascat', year0=2016,
                      ascat_att0=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore',
                                  'inc_angle_trip_fore', 'sigma0_trip_mid', 'inc_angle_trip_mid']):
    """
    get ascat time series with specified pixels and attributes. shape of each keys (number of pixels * 9, time_period)
    :param p_ids:
    :param t_window:
    :param path0:
    :param y:
    :param ascat_att0:
    :return:
    """
    doy_array = np.arange(t_window[0], t_window[1])
    file_path='ascat_resample_all2'
    path_ascat = []
    for doy0 in doy_array:
        time_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=year0)
        match_name = 'result_08_01/%s/ascat_*%s*.h5' % (file_path, time_str0)
        path_ascat += glob.glob(match_name)
    # read ascat_data into dictionary. each key0 corresponded to the key0 of the ascat h5 file (300, 300, time)
    ascat_dict = data_process.ascat_alaska_grid_v2(ascat_att0, path_ascat,  pid=p_ids)  # keys contain 'sate_type'
    # np.savez('%s/ascat_data0.npz' % path0, **ascat_dict)
    return ascat_dict


def get_yearly_files(t_window=[0, 210], year0=2016):
    # ascat_att0=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore',
    #                               'inc_angle_trip_fore', 'sigma0_trip_mid', 'inc_angle_trip_mid']
    # path0='./result_08_01/series/ascat'
    doy_array = np.arange(t_window[0], t_window[1])
    file_path = 'ascat_resample_all3'
    path_ascat = []
    for doy0 in doy_array:
        time_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=year0)
        match_name = 'result_08_01/%s/ascat_*%s*.h5' % (file_path, time_str0)
        path_ascat += glob.glob(match_name)
    return path_ascat


def get_yearly_smap(t_window=[0, 366], year0=2016):
    smap_dict_2016 = data_process.get_smap_dict(np.arange(t_window[0], t_window[1]), y=year0)
    return smap_dict_2016


def get_regional_ascat(win=[0, 210], year0=2016, pid=False):
    # get yearly ascat data, interpolation based on distance, correct the angular dependency. Save in forms of npz file.
    if pid is False:
        mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
        mask_1d = mask.reshape(1, -1)[0]
        land_id = np.where(mask_1d != 0)[0]  # all land id in alaska
        region_name = 'ak'
    else:
        land_id = pid
        region_name = 'stations'
    save_keys = ['sigma0_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore',
                                  'sigma0_trip_mid']
    ascat_dict0 = get_yearly_ascat(land_id, t_window=win, year0=year0)  # get ascat time series
    dis = ascat_dict0['distance'].copy()  # distance between smap and 9 ascat pixels, small to large
    mean_mid_series = \
    data_process.distance_interpolate_v2(dis, ascat_dict0['inc_angle_trip_mid'], ascat_dict0['sigma0_trip_mid'],
                                         ascat_dict0['utc_line_nodes'])
    mean_aft_series = \
    data_process.distance_interpolate_v2(dis, ascat_dict0['inc_angle_trip_aft'], ascat_dict0['sigma0_trip_aft'],
                                         ascat_dict0['utc_line_nodes'])
    mean_fore_series = \
    data_process.distance_interpolate_v2(dis, ascat_dict0['inc_angle_trip_fore'], ascat_dict0['sigma0_trip_fore'],
                                         ascat_dict0['utc_line_nodes'])
    save_dict = {key0: means for key0, means in zip(['sigma0_trip_aft', 'sigma0_trip_fore',
                                                     'sigma0_trip_mid', 'inc_angle_trip_aft',
                                                     'inc_angle_trip_fore', 'inc_angle_trip_mid', 'utc_line_nodes'],
                                                    [mean_aft_series[0], mean_fore_series[0], mean_mid_series[0],
                                                     mean_aft_series[1], mean_fore_series[1], mean_mid_series[1],
                                                     mean_aft_series[2]])}
    save_dict['sate_type'] = ascat_dict0['sate_type']
    # save check dictionary
    check_dict = {}
    check_dict['inc_angle_trip_aft'], check_dict['sigma0_trip_aft'], check_dict['utc_line_nodes'] = \
        ascat_dict0['inc_angle_trip_aft'], ascat_dict0['sigma0_trip_aft'], ascat_dict0['utc_line_nodes']
    check_dict['inc_angle_trip_mid'], check_dict['sigma0_trip_mid'] = \
        ascat_dict0['inc_angle_trip_mid'], ascat_dict0['sigma0_trip_mid']
    check_dict['distance'] = ascat_dict0['distance']

    # data_process.angular_effect_v2(save_dict, 'inc_angle_trip_aft', 'sigma0_trip_aft')
    # data_process.angular_effect_v2(save_dict, 'inc_angle_trip_fore', 'sigma0_trip_fore')
    # data_process.angular_effect_v2(save_dict, 'inc_angle_trip_mid', 'sigma0_trip_mid')
    fname0 = 'ascat_year_correct_interpolate_%s_%d.npz' % (region_name, year0)
    np.savez(fname0, **save_dict)
    # check0 = 'ascat_check_%s_%d.npz'  % (region_name, year0)
    # np.savez(check0, **check_dict)


def get_regional_onsets(series0, y=2016, k0=7):
    """
    :param series0: 0: time in secs, 1: values
    :param y:
    :param k0: the standard deviation of gaussian distribution
    :return:
    """
    conv_npr, thaw_secs_npr = data_process.get_onset(series0[0], series0[1], year0=y,   # npr_a_t1
                           thaw_window=[bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12]) +
                           doy0*3600*24 for doy0 in [60, 150]],
                           k=k0, type='npr')  # npr asc up
    return 0


def get_alaska_onset(year0=2016, input_ascat=False):
    orb_no = 0
    y = year0
    secs_0101 = bxy.get_total_sec('%d0101' % y)
    secs_0601 = bxy.get_total_sec('%d0601' % y)
    melt_zone0 = np.array([bxy.get_total_sec(str0) for str0 in ['%d0301' % y, '%d0601' % y]])  # in unit of secs
    thaw_window = melt_zone0.copy()
    secs_winter = np.array([bxy.get_total_sec(str0) for str0 in ['%d0101' % y, '%d0301' % y]])
    # smap series
    smp = get_yearly_smap(t_window=[0, 366], year0=year0)  # asc and des orbits smap measurements
    mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]  # all land id in alaska
    tbv, tbh = smp[orb_no]['cell_tb_v_aft'][land_id], smp[orb_no]['cell_tb_h_aft'][land_id]
    t_tb = smp[orb_no]['cell_tb_time_seconds_aft'][land_id] + 12*3600
    npr = (tbv - tbh)/(tbv + tbh)
    # valid smap measurements
    valid_i0 = (tbv > -90) & (tbv > tbh)
    npr[~valid_i0] = -9999.0
    # ascat series
    if input_ascat is not False:
        ascat_series = input_ascat
    else:
        ascat_series = np.load('ascat_year_correct_interpolate_%d.npz' % year0)
    un_v_pixel = 0
    # melt array
    onset_array_smap = np.zeros(tbv.shape[0]) - 999
    conv_array = np.zeros(ascat_series['utc_line_nodes'].shape[0]) - 999
    lvl_array = conv_array.copy()
    for l0 in np.arange(0, tbv.shape[0]):  # tbv.shapef
        # l0 = 22
        # print 'the no. is', l0
        melt_onset0 = 0
        melt_conv = -999
        melt_lvl0 = 0
        conv_npr, thaw_secs_npr = data_process.get_onset(t_tb[l0], npr[l0], year0=y,   # npr_a_t1
                       thaw_window=[bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12]) +
                       doy0*3600*24 for doy0 in [60, 150]],
                       k=7, type='npr')  # npr asc up
        sigma0, inc0, times0 = ascat_series['sigma0_trip_mid'][l0], \
                               ascat_series['inc_angle_trip_mid'][l0], \
                               ascat_series['utc_line_nodes'][l0]
        unvalid = (ascat_series['sigma0_trip_mid'][l0] == 0) | (ascat_series['sigma0_trip_mid'][l0] < -90)
        # check the numbers of measurements for each pixel: pixel number: valid measurements
        with open('valid_number_%d.txt' % year0, 'a') as f00:
            pause = 0
            f00.write('%d, %d \n' % (land_id[l0], sum(~unvalid)))
        if sum(~unvalid) < 20:
            print 'no time series at this pixel'
            un_v_pixel += 1
            print un_v_pixel
            continue

        sigma0_correct = data_process.angular_correct(sigma0, inc0, times0, inc_c=40)
        max_value_ascat, min_value_ascat, conv_ascat \
                    = test_def.edge_detect(times0, sigma0_correct,
                                           7, seriestype='sig', is_sort=False)
        if max_value_ascat.size < 1:
            max_value_ascat = np.array([[0., secs_0101, 0.]])
        if min_value_ascat.size < 1:
            min_value_ascat = np.array([[0., secs_0101, 0.]])
        thaw_ascat = max_value_ascat[(max_value_ascat[:, 1] > thaw_window[0]) &
                                         (max_value_ascat[:, 1] < thaw_window[1])]
        if thaw_ascat.size < 1:
            thaw_onset0 = secs_0601
        else:
            thaw_onset0 = thaw_ascat[:, 1][thaw_ascat[:, -1].argmax()]
        # re-set melt_zone
        melt_zone0[0] = thaw_secs_npr
        melt_zone0[1] = thaw_onset0
        min_detect_winter = min_value_ascat[(min_value_ascat[:, 1] > secs_winter[0]) &
                               (min_value_ascat[:, 1] < secs_winter[1])]
        min_conv_winter_mean = np.nanmean(min_detect_winter[:, -1])
        min_detect_snowmelt = min_value_ascat[(min_value_ascat[:, 1] > melt_zone0[0]) &
                                 (min_value_ascat[:, 1] < melt_zone0[1])]
        if min_detect_snowmelt[:, -1].size < 1:
            melt_onset0 = 0
            melt_conv = -999
            melt_lvl0 = 0
        else:
            # consider the significant snow melt event
            levels = np.abs(min_detect_snowmelt[:, -1]/min_conv_winter_mean)
            valid_index_melt = levels>2.5
            if sum(valid_index_melt) > 0:
                # if sum(valid_index_melt) > 1:
                melt_onset0 = min_detect_snowmelt[:, 1][valid_index_melt][0]
                # else:
                #     melt_onset0 = min_sec_snowmelt[:, 1][valid_index_melt][0]
                melt_lvl0 = levels[valid_index_melt][0]
            else:
                melt_onset0 = min_detect_snowmelt[:, 1][min_detect_snowmelt[:, -1].argmin()]
                melt_lvl0 = levels[min_detect_snowmelt[:, -1].argmin()]
            melt_conv = min_detect_snowmelt[:, -1][min_detect_snowmelt[:, -1].argmin()]
        onset_array_smap[l0], conv_array[l0], lvl_array[l0] = melt_onset0, melt_conv, melt_lvl0
    np.savez('onset_%d.npz' % year0, *[onset_array_smap, conv_array, lvl_array])


def get_smap_onset(orb_no=0):
    # smap series
    smp = get_yearly_smap(t_window=[0, 366], year0=year0)  # asc and des orbits smap measurements
    mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]  # all land id in alaska within the AK (360N grid system)
    tbv, tbh = smp[orb_no]['cell_tb_v_aft'][land_id], smp[orb_no]['cell_tb_h_aft'][land_id]
    t_tb = smp[orb_no]['cell_tb_time_seconds_aft'][land_id] + 12*3600
    npr = (tbv - tbh)/(tbv + tbh)
    valid_i0 = (tbv > -90) & (tbv > tbh)  # npr
    npr[~valid_i0] = -9999.0
    onset_array_smap = np.zeros(tbv.shape[0]) - 999
    # n125_n360 = np.load('n12_n36_array.npy').astype(int)
    for l0 in np.arange(0, tbv.shape[0]):  # tbv.shape[0]
        # print 'the no. is', l0
        conv_npr, thaw_secs_npr = data_process.get_onset(t_tb[l0], npr[l0], year0=y,   # npr_a_t1
                       thaw_window=[bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12]) +
                       doy0*3600*24 for doy0 in [60, 150]],
                       k=7, type='npr')  # npr asc up
        onset_array_smap[l0] = thaw_secs_npr
    return onset_array_smap


def get_finer_onset(year0=2016, input_ascat=False, savename=False, ascat_index=False):
    orb_no = 0
    y = year0
    secs_0101 = bxy.get_total_sec('%d0101' % y)
    secs_0601 = bxy.get_total_sec('%d0601' % y)
    melt_zone0 = np.array([bxy.get_total_sec(str0) for str0 in ['%d0301' % y, '%d0601' % y]])  # in unit of secs
    thaw_window = melt_zone0.copy()
    secs_winter = np.array([bxy.get_total_sec(str0) for str0 in ['%d0101' % y, '%d0301' % y]])

    # smap series
    smp = get_yearly_smap(t_window=[0, 366], year0=year0)  # asc and des orbits smap measurements
    mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]  # all land id in alaska
    tbv, tbh = smp[orb_no]['cell_tb_v_aft'][land_id], smp[orb_no]['cell_tb_h_aft'][land_id]
    t_tb = smp[orb_no]['cell_tb_time_seconds_aft'][land_id]
    t_tb[t_tb>1e7] += 12*3600
    npr = (tbv - tbh)/(tbv + tbh)
    # valid smap measurements
    valid_i0 = (tbv > -90) & (tbv > tbh)
    npr[~valid_i0] = -9999.0
    # ascat series
    if input_ascat is not False:
        ascat_series = input_ascat
    else:
        ascat_series = np.load('ascat_year_correct_interpolate_%d.npz' % year0)
    # connect ascat pixel with smap pixel
    n125_n360_all = np.load('n12_n36_array.npy').astype(int)
    if ascat_series['sigma0_trip_mid'].shape[0] > 1e3:
        # n125_n360: 1d index of 0: sigma0, 1: smap
          n125_n360 = n125_n360_all
    else:
        n125_n360_id = np.array([np.where(n125_n360_all[0] == ascat_index0)[0][0] for ascat_index0 in ascat_index])
        n125_n360 = n125_n360_all[:, n125_n360_id]
    un_v_pixel = 0
    # smap series
    # onset based on smap, all pixels
    onset_array_smap = np.zeros(tbv.shape[0]) - 999
    conv_array = np.zeros(ascat_series['utc_line_nodes'].shape[0]) - 999
    lvl_array = conv_array.copy()
    onset_array_ascat = conv_array.copy()

    for l0 in np.arange(0, tbv.shape[0]):  # tbv.shape[0]
        # l0 = 22
        # print 'the no. is', l0
        conv_npr, thaw_secs_npr = data_process.get_onset(t_tb[l0], npr[l0], year0=y,   # npr_a_t1
                       thaw_window=[bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12]) +
                       doy0*3600*24 for doy0 in [60, 150]],
                       k=7, type='npr')  # npr asc up
        onset_array_smap[l0] = thaw_secs_npr
    # onsets from ascat series
    sigma0_pixel = []
    smap_pixel = []
    for l0 in np.arange(0, ascat_series['sigma0_trip_mid'].shape[0]):
        smap_pid = n125_n360[1, l0]
        smap_land_index = np.where(land_id == smap_pid)[0][0]  # index out of the land data
        melt_onset0 = 0
        melt_conv = -999
        melt_lvl0 = 0
        thaw_secs_npr = onset_array_smap[smap_land_index]
        sigma0, inc0, times0 = ascat_series['sigma0_trip_mid'][l0], \
                               ascat_series['inc_angle_trip_mid'][l0], \
                               ascat_series['utc_line_nodes'][l0]
        unvalid = (ascat_series['sigma0_trip_mid'][l0] == 0) | (ascat_series['sigma0_trip_mid'][l0] < -90)
        if sum(~unvalid) < 20:
            print 'no time series at this pixel'
            un_v_pixel += 1
            print un_v_pixel
        # time series and convolution of smap series
        t_pixel,  npr_pixel = t_tb[smap_land_index], npr[smap_land_index]
        conv_npr_pixel, thaw_secs_npr_pixel = data_process.get_onset(t_pixel, npr_pixel, year0=y,   # npr_a_t1
                       thaw_window=[bxy.get_total_sec('%d0101' % y, reftime=[2000, 1, 1, 12]) +
                       doy0*3600*24 for doy0 in [60, 150]],
                       k=7, type='npr')
        smap_pack = np.array([t_pixel, npr_pixel, conv_npr_pixel])
        smap_pixel.append(smap_pack)
        sigma0_correct = data_process.angular_correct(sigma0, inc0, times0, inc_c=40)
        max_value_ascat, min_value_ascat, conv_ascat \
                    = test_def.edge_detect(times0, sigma0_correct,
                                           7, seriestype='sig', is_sort=False)
        sigma0_pack = np.array([times0, sigma0_correct, conv_ascat])
        sigma0_pixel.append(sigma0_pack)
        if max_value_ascat.size < 1:
            max_value_ascat = np.array([[0., secs_0101, 0.]])
        if min_value_ascat.size < 1:
            min_value_ascat = np.array([[0., secs_0101, 0.]])
        thaw_ascat = max_value_ascat[(max_value_ascat[:, 1] > thaw_window[0]) &
                                         (max_value_ascat[:, 1] < thaw_window[1])]
        if thaw_ascat.size < 1:
            thaw_onset0 = secs_0601
        else:
            thaw_onset0 = thaw_ascat[:, 1][thaw_ascat[:, -1].argmax()]
        # re-set melt_zone
        melt_zone0[0] = thaw_secs_npr
        melt_zone0[1] = thaw_onset0
        min_detect_winter = min_value_ascat[(min_value_ascat[:, 1] > secs_winter[0]) &
                               (min_value_ascat[:, 1] < secs_winter[1])]
        min_conv_winter_mean = np.nanmean(min_detect_winter[:, -1])
        min_detect_snowmelt = min_value_ascat[(min_value_ascat[:, 1] > melt_zone0[0]) &
                                 (min_value_ascat[:, 1] < melt_zone0[1])]
        if min_detect_snowmelt[:, -1].size < 1:
            melt_onset0 = 0
            melt_conv = -999
            melt_lvl0 = 0
        else:
            # consider the significant snow melt event
            levels = np.abs(min_detect_snowmelt[:, -1]/min_conv_winter_mean)
            valid_index_melt = levels>2.5
            if sum(valid_index_melt) > 0:
                # if sum(valid_index_melt) > 1:
                melt_onset0 = min_detect_snowmelt[:, 1][valid_index_melt][0]
                # else:
                #     melt_onset0 = min_sec_snowmelt[:, 1][valid_index_melt][0]
                melt_lvl0 = levels[valid_index_melt][0]
            else:
                melt_onset0 = min_detect_snowmelt[:, 1][min_detect_snowmelt[:, -1].argmin()]
                melt_lvl0 = levels[min_detect_snowmelt[:, -1].argmin()]
            melt_conv = min_detect_snowmelt[:, -1][min_detect_snowmelt[:, -1].argmin()]
            onset_array_ascat[l0], conv_array[l0], lvl_array[l0] = melt_onset0, melt_conv, melt_lvl0
    if savename is not False:
        # onset value, time series of npr and ascat; convolution of npr and ascat
        return [onset_array_smap, onset_array_ascat, conv_array, lvl_array], [smap_pixel, sigma0_pixel]
    else:
        np.savez('onset_%d.npz' % year0, *[onset_array_smap, onset_array_ascat, conv_array, lvl_array])


def prepare_smap_series(land_id, input_ascat=False, input_smap=False, ascat_index=False):
    '''
    Obtain smap series (e.g., NPR) based on the input array (dict type, input_smap)
    :param land_id: the 1d id of interested smap pixels
    :param input_ascat:
    :param input_smap: 2 orbits, with keys(), example: ['cell_tb_v_aft']: 9000 * number of days
    :param ascat_index: the 1d indices of input ascat, which are used to find corresponded smap pixels
    :return:  npr series: 3 dimensions: (times/values x total pixels number x time series)
    '''
    orb_no = 0
    # connect ascat pixel with smap pixel
    un_v_pixel = 0
    # smap series
    tbv, tbh = input_smap[orb_no]['cell_tb_v_aft'][land_id], input_smap[orb_no]['cell_tb_h_aft'][land_id]
    t_tb = input_smap[orb_no]['cell_tb_time_seconds_aft'][land_id]
    t_tb[t_tb>1e7] += 12*3600
    npr = (tbv - tbh)/(tbv + tbh)
    valid_i0 = (tbv > -90) & (tbv > tbh)
    npr[~valid_i0] = -9999.0
    # quick cheeck station 947 time series in the specific year
    # test_plot = np.array([t_tb[0], npr[0]])  # land_id==4547
    # test_plot[test_plot<0] = np.nan
    # plot_funcs.plot_subplot([test_plot, test_plot],
    #                     [test_plot, test_plot],
    #                     main_label=['npr', 'npr'],
    #                     figname='ms_pixel_check_%s' % ('947_1'), x_unit='doy',
    #                     )
    return np.array([t_tb, npr]), land_id


def combine_detect_v2(id_array, year0, ascat_series, smap_series, all_region=True, npz_name='interest', gk=[7, 10, 10],
                      pid_smap=np.array([4547, 3770]), npz_doc='.'):
    """
    :param id_array: the 1_d indices of ascat pixels, and smap pixels. note that the max number of smap pixels is 1603.
    :param year0:
    :param ascat_series: dictionary whose keys() include attributes of the interested ascat pixels.
                         Shape & dimension: Pixels X time
    :param smap_series: ndarray save the smap measurements. Shape/dimension: orbits X pixels X time
    :param all_region: True or False
    :return:
    if there are only specified pixels (save_sp is true), return that:
        0 smap ,ascat onset, negative edge value, and significant lvl on ascat onset.
        1 ascat_pixel and smap pixel. Each pixel include:
            [0 time of measurements, 1 value of measurements, 2 time & value of convolution]
        i.e., [onset_array_smap, onset_array_ascat, conv_array, lvl_array], [smap_pixel, ascat_pixel]
    if not, we investigate all pixels in Alaska, save includes:
        onset_array_smap, onset_array_ascat, conv_array, lvl_array. UPDATED 201905
    winter ()
    """
    # saved array: ascat, convolution, level, onset, mean_winter, mean_summer, melt_signal
    # saved array: smap, onset, mean_winter, mean_summer, melt_signal
    '''
    ------------------------------- start line ------------------------------------------------------------------------
    '''
    # from variables
    # the keys of ascat dictionary
    t_tb, npr = smap_series[0], smap_series[1]
    # check sigma type
    # index_asc = ascat_series['sate_type'] < 2
    sigma0_type = ascat_series['sate_type']
    sigma0_all, inc0_all, times0_all = ascat_series['sigma0_trip_mid'], \
                       ascat_series['inc_angle_trip_mid'], \
                       ascat_series['utc_line_nodes']
    # initials
    melt_zone0 = np.array([bxy.get_total_sec(str0) for str0 in ['%d0301' % year0, '%d0601' % year0]])  # in unit of secs
    secs_summer = np.array([bxy.get_total_sec(str0) for str0 in ['%d0701' % year0, '%d0901' % year0]])
    thaw_window = melt_zone0.copy()
    secs_winter = np.array([bxy.get_total_sec(str0) for str0 in ['%d0101' % year0, '%d0301' % year0]])

    start0 = bxy.get_time_now()
    smap_out, ascat_out = data_process.two_series_detect_v2(id_array, [npr, t_tb],
                                                                    [sigma0_all, inc0_all, times0_all],
                                                                    year0, pid_smap=pid_smap, gk=gk)
    print 'two_series_detect_v2 for year %d takes %s seconds' % (year0, bxy.get_time_now()-start0)
    # return 0, 0
    # origin out list
    # [0 onset_array_ascat, 1 melt_end_ascat, 2 conv_ascat_array, 3 lvl_array,
    #  4 mean_winter,       5 mean_summer,    6 smap_melt_signal, 7 mean_melt_a,
    #  8 std_winter_a,      9 std_summer_a,   10 std_melt_a,      11 coef_a,      12 coef_b, 13 time_zero_conv
    #  14 winter_edge 15 sigma0_on_edge, 16 a list contains stat of winter convolutions
    #  17,            18,                19 sigma0_5d_after_onset], \

    # new out list of ascat
    # 0 l0, 1 coef_a, 2 coef_b, 3 pixel_kernels, 4 mean_winter, 5 mean_summer, 6 mean_melt_a,
    # 7 std_winter_a, 8 std_summer_a/sigma_std_summer, 9 std_melt_a, 10 onset_array_ascat,
    # 11 conv_ascat_array, 12 lvl_array, 13 melt_end_ascat, 14 time_zero_conv, \
    # 15 sigma0_on_melt_date, 16 sigma0_min_melt_zone, 17 sigma0_5d_after_onset, 18 winter_edge, 19 winter_conv_mean, \
    # 20 winter_conv_min, 21 winter_conv_std, 22 min_melt_a, \
    # 23 ascat_pixel, 24 melt_events_time_list, 25 melt_events_conv_list
    # find winter_conv_mean
    onset_array_smap = smap_out[0]
    smap_pixel = smap_out[4]
    npr_on_smap_melt_date_ak = smap_out[5]  # smap
    smap_seasonal = [smap_out[1], smap_out[2], smap_out[3]]
    if sigma0_all.shape[0] > 2e3:
        all_region = True
    if all_region:
        time_prefix = bxy.get_time_now()
        time_array = np.array([time_prefix.month, time_prefix.day, time_prefix.hour, time_prefix.minute])
        time_str_array = [time_array[l0].astype(str) for l0 in range(time_array.size)]
        time_str_array_formated = ['0'+item if len(item) < 2 else item for item in time_str_array]
        np.savez('%s/smap_onset_%s_%d_%s%s%s%s.npz' %
                 (npz_doc, npz_name, year0,
                  time_str_array_formated[0], time_str_array_formated[1], time_str_array_formated[2],
                  time_str_array_formated[3]),
                 **{'smap_onset': onset_array_smap,
                    'npr_on_smap_melt_date_ak': npr_on_smap_melt_date_ak, 'smap_winter': smap_seasonal[0],
                    'smap_summer': smap_seasonal[1], 'smap_peak': smap_seasonal[2],
                    })
        return 0, 0
        #      **{'smap_onset': onset_array_smap, 'ascat_onset': onset_array_ascat,
        # 'ascat_end': melt_end_ascat, 'ascat_edge': conv_ascat_array, 'ascat_lvl': lvl_array,
        # 'sigma0_mean_winter': mean_winter, 'sigma0_mean_summer': mean_summer,
        # 'sigma0_mean_melt_zone': sigma0_mean_melt,
        # 'sigma0_std_winter': sigma0_std_winter, 'sigma_std_summer': sigma_std_summer,
        # 'simga0_std_melt_zone': sigma0_std_meltseason,
        # 'winter_edge': winter_edge, 'time_zero_conv': time_zero_conv, 'sigma0_on_melt_date': sigma0_on_melt_date,
        # 'sigma0_min_melt_zone': sigma0_min_melt_zone,
        # 'npr_on_smap_melt_date_ak': npr_on_smap_melt_date_ak,
        # 'coef_a': coef_a, 'coef_b': coef_b, 'smap_winter': smap_seasonal[0],
        # 'smap_summer': smap_seasonal[1], 'smap_peak': smap_seasonal[2],
        # 'winter_conv_mean': winter_conv_mean,
        # 'winter_conv_min': winter_conv_min,
        # 'winter_conv_std}': winter_conv_std,
        # 'sigma0_5dmean_after_onset': sigma0_5d_after_onset,
        # 'sigma0_kernels': sigma0_kernels, 'melt_events_time': melt_events_time,
        # 'melt_events_conv': melt_events_conv
        # # older version 20190812
        # onset value, time series of npr and ascat; convolution of npr and ascat
        # if npz_name == 'no_name':
        #     print 'the npz file for interest/outlier pixels has already been saved'
        #     return [onset_array_smap, onset_array_ascat, melt_end_ascat, conv_ascat_array, lvl_array], \
        #            [smap_pixel, ascat_pixel]
        # print 'saved in onset_%s_%d.npz' % (npz_name, year0)
        # np.savez('onset_%s_%d.npz' % (npz_name, year0),
        #          **{'output_array': np.array([onset_array_smap, onset_array_ascat,
        #                                       melt_end_ascat, conv_ascat_array, lvl_array]),
        #             'smap_pixel': smap_pixel, 'ascat_pixel': ascat_pixel, 'sate_type': sigma0_type,
        #             'winter_conv_mean': winter_conv_mean,
        #             'winter_conv_min': winter_conv_min,
        #             'winter_conv_std}': winter_conv_std})
        # return [onset_array_smap, onset_array_ascat, melt_end_ascat, conv_ascat_array, lvl_array], [smap_pixel, ascat_pixel]
    else:
        # np.savez('onset_%d.npz' % year0, *[onset_array_smap, onset_array_ascat, conv_array, lvl_array,
        #                                    mean_winter, mean_summer])
        time_prefix = bxy.get_time_now()
        np.savez('%s/onset_%s_%d_%d%d%d.npz' %
                 (npz_doc, npz_name, year0, time_prefix.day, time_prefix.hour, time_prefix.minute),
                 **{'smap_onset': onset_array_smap, 'ascat_onset': onset_array_ascat,
                    'ascat_end': melt_end_ascat, 'ascat_edge': conv_ascat_array, 'ascat_lvl': lvl_array,
                    'sigma0_mean_winter': mean_winter, 'sigma0_mean_summer': mean_summer,
                    'sigma0_mean_melt_zone': sigma0_mean_melt,
                    'sigma0_std_winter': sigma0_std_winter, 'sigma_std_summer': sigma_std_summer,
                    'simga0_std_melt_zone': sigma0_std_meltseason,
                    'winter_edge': winter_edge, 'time_zero_conv': time_zero_conv, 'sigma0_on_melt_date': sigma0_on_melt_date,
                    'sigma0_min_melt_zone': sigma0_min_melt_zone,
                    'npr_on_smap_melt_date_ak': npr_on_smap_melt_date_ak,
                    'coef_a': coef_a, 'coef_b': coef_b, 'smap_winter': smap_seasonal[0],
                    'smap_summer': smap_seasonal[1], 'smap_peak': smap_seasonal[2],
                    'winter_conv_mean': winter_conv_mean,
                    'winter_conv_min': winter_conv_min,
                    'winter_conv_std}': winter_conv_std,
                    'sigma0_5dmean_after_onset': sigma0_5d_after_onset,
                    'sigma0_kernels': sigma0_kernels, 'melt_events_time': melt_events_time,
                    'melt_events_conv': melt_events_conv
                    })
        print 'the result saved in onset_%d_%d%d%d.npz' % (year0, time_prefix.day, time_prefix.hour, time_prefix.minute)
        return 0, 0


def ms_read_ascat(year0=2016, pixel_id=False, t_window=[0, 210], ascat_att0=[]):
    '''

    :param year0:
    :param pixel_id: the 2d id based on the (300, 300) grid system
    :param ascat_att0:
    :return:
        dictionary that saves the ascat measurements, keys's given in ascat_att0.
        dimension number of pixels X number of measurements
    '''
    path_files = get_yearly_files(t_window=t_window, year0=year0)
    # path_files = get_yearly_files(t_window=[30, 40], year0=year0)
    if len(ascat_att0) < 1:
        ascat_att0=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore',
                                      'inc_angle_trip_fore', 'sigma0_trip_mid', 'inc_angle_trip_mid']
    # if pixel_id is not False:
    #     if pixel_id.size > 1e4:
    #         print 'all pixels in alaska are read'
    #     else:
    #         print 'the pixels of interest have been spcified'
    #     pasue = 0
    # else:
    #     print 'all pixels in alaska are read'
    #     # pixel id is the pixel with land_mask >  0
    #     pixel_id = np.where(mask0>0)

    dict_2016 = data_process.ascat_alaska_grid_v3(ascat_att0, path_files, pid=pixel_id)
    return dict_2016


def ms_map(year0=2016):
    value_npz = np.load('onset_%d.npz' % year0)  # onset_smap, onset_ascat, conv_array, lvl_array
    onset = value_npz['arr_1']
    v_valid = (onset!=0)&(onset!=-999)
    onset[v_valid] = bxy.time_getlocaltime(onset[v_valid], ref_time=[2000, 1, 1, 0])[-2]
    value_array = data_process.make_mask(onset)
    quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160], fig_name='finer_onset_%d' % year0)
    value_array2 = data_process.make_mask(value_npz['arr_3'])
    quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_lvl_%d' % year0, z_value=[-2, 10])
    return 0


def ms_ascat_target(p_coord, win=[0, 210]):
    """

    :param p_coord: longitude and latitude of each pixel
    :param win:
    :return:
    """
    path_files = get_yearly_files(t_window=[0, 210], year0=year0)
    ascat_att0=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore',
                              'inc_angle_trip_fore', 'sigma0_trip_mid', 'inc_angle_trip_mid']
    lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                    np.load('./result_05_01/other_product/lat_ease_grid.npy')
    mask_a = np.load('/home/xiyu/PycharmProjects/R3/result_05_01/other_product/mask_ease2_125N.npy')
    lon_land_a, lat_land_a = lons_grid[mask_a>0], lats_grid[mask_a>0]
    # turn p_coord to index
    pid = np.zeros(p_coord.shape[0])
    i_pid = 0
    for coord0 in p_coord:
        dis = bxy.cal_dis(coord0[1], coord0[0], lat_land_a, lon_land_a)
        pid[i_pid] = dis.argmin()
        i_pid += 1
    dict_ascat = data_process.ascat_alaska_grid_v3(ascat_att0, path_files, pid=np.where(mask0>0))


def ms_station_prepare(win=[0, 210]):
    pixel_num = np.loadtxt('result_agu/result_2019/points_num_tp.txt', delimiter=',')
    pid = pixel_num[0].astype(int)
    print 'id of pixels: ', pid, 'are reading'
    ascat = get_ascat_series(p_ids=pid, t_window=win)
    smp = get_smap_series(p_ids=pid, t_window=win)
    print 'id of pixels: ', pid, 'have been read and saved'
    return ascat, smp


def ms_station(win=[0, 210], year0=2016, insitu=False):
    """
    read the npz files saving ascat/smap dictionary, for each station, ascat dict contains 9 nns.
    :param win:
    :param year0:
    :return:
    """
    sig_level = 2.5
    year_end = [bxy.get_total_sec(t0) for t0 in ['20160101', '20170101', '20180101', '20190101']]
    dict_a, dict_s = np.load('ascat_data0.npz'), np.load('smap_data0.npz')
    pixel_num = np.loadtxt('result_agu/result_2019/points_num_tp.txt', delimiter=',')
    sid = pixel_num[1].astype(int)  # station id
    pid = pixel_num[0].astype(int)  # pixel id
    # melt_zone0 = np.array([bxy.get_total_sec(str0) for str0 in ['%d0301' % year0, '%d0601' % year0]])
    # onset_array_smap = np.zeros(tbv.shape[0]) - 999
    dis = dict_a['distance'].copy()
    mid_angle, mid_sig, mid_t = dict_a['inc_angle_trip_mid'], dict_a['sigma0_trip_mid'], dict_a['utc_line_nodes']
    mean_mid_series = data_process.distance_interpolate_v2(dis, mid_angle, mid_sig, mid_t)  # value, angle, time
    for l0 in np.arange(0, pid.size):
        # initials
        conv_series = np.array([])
        conv_t = conv_series.copy()

        # obtain smap time seris
        tbv, tbh, t_smap = dict_s['cell_tb_v_aft'][pid[l0]], dict_s['cell_tb_h_aft'][pid[l0]], \
                           dict_s['cell_tb_time_seconds_aft'][pid[l0]]
        unvalid = (tbv < -100) | (tbh < -100) | (tbv < tbh)
        npr_smap = (tbv-tbh)/(tbv+tbh)
        npr_smap[unvalid] = -999
        # obtain scat time series, interpolation and correction
        t_ascat = mean_mid_series[2, l0, :]
        v_ascat, angle_ascat = mean_mid_series[0, l0, :], mean_mid_series[1, l0, :]
        sigma0_correct_all = data_process.angular_correct(v_ascat.copy(), angle_ascat, t_ascat, inc_c=40)
        # sigma0_correct_all = v_ascat
        # each year conv, onset of npr; conv, onset, and lvl of ascat
        c_npr, c_sigma, t_sigma = [], [], []
        o_npr, o_sigma, lv_sigma = [], [], []
        v_sigma_c = []
        for year1 in [2016, 2017, 2018]:
            year_sec = [bxy.get_total_sec(str0) for str0 in ['%d0101' % year1, '%d1231' % year1]]
            thaw_window = [bxy.get_total_sec(str0) for str0 in ['%d0301' % year1, '%d0531' % year1]]
            melt_zone0 = np.array([bxy.get_total_sec(str0) for str0 in ['%d0301' % year1, '%d0601' % year1]])

            p0 = (t_smap > year_sec[0]) & (t_smap < year_sec[1])
            # detect npr and ascat edge
            conv_npr, thaw_secs_npr = data_process.get_onset(t_smap[p0], npr_smap[p0], year0=year1,  # smap
                               thaw_window=[bxy.get_total_sec('%d0101' % year1, reftime=[2000, 1, 1, 12]) +
                               doy0*3600*24 for doy0 in [60, 150]], k=7, type='npr')
            c_npr.append(conv_npr)
            o_npr.append(thaw_secs_npr)
            p00 = (t_ascat > year_sec[0]) & (t_ascat < year_sec[1])  # ascat
            sigma0_correct = data_process.angular_correct(v_ascat[p00], angle_ascat[p00], t_ascat[p00], inc_c=40)
            v_sigma_c.append(sigma0_correct), t_sigma.append(t_ascat[p00])
            max_value_ascat, min_value_ascat, conv_ascat = test_def.edge_detect\
                (t_ascat[p00], sigma0_correct, 7, seriestype='sig', is_sort=False)
            c_sigma.append(conv_ascat)
            thaw_ascat = max_value_ascat[(max_value_ascat[:, 1] > thaw_window[0]) &
                                             (max_value_ascat[:, 1] < thaw_window[1])]
            if thaw_ascat.size < 1:
                thaw_onset0 = thaw_window[0]
            else:
                thaw_onset0 = thaw_ascat[:, 1][thaw_ascat[:, -1].argmax()]
            # re-set melt_zone
            melt_zone0[0] = thaw_secs_npr
            melt_zone0[1] = thaw_onset0
            min_detect_winter = min_value_ascat[(min_value_ascat[:, 1] > year_sec[0]) &
                                   (min_value_ascat[:, 1] < thaw_window[1])]
            min_conv_winter_mean = np.nanmean(min_detect_winter[:, -1])
            min_detect_snowmelt = min_value_ascat[(min_value_ascat[:, 1] > melt_zone0[0]) &
                                     (min_value_ascat[:, 1] < melt_zone0[1])]
            if min_detect_snowmelt[:, -1].size < 1:
                melt_onset0 = 0
                melt_conv = -999
                melt_lvl0 = 0
                o_sigma.append(melt_onset0)
                lv_sigma.append(melt_lvl0)
            else:
                # consider the significant snow melt event
                levels = np.abs(min_detect_snowmelt[:, -1]/min_conv_winter_mean)
                valid_index_melt = levels>sig_level
                if sum(valid_index_melt) > 0:
                    # if sum(valid_index_melt) > 1:
                    melt_onset0 = min_detect_snowmelt[:, 1][valid_index_melt][0]
                    # else:
                    #     melt_onset0 = min_sec_snowmelt[:, 1][valid_index_melt][0]
                    melt_lvl0 = levels[valid_index_melt][0]
                else:
                    melt_onset0 = min_detect_snowmelt[:, 1][min_detect_snowmelt[:, -1].argmin()]
                    melt_lvl0 = levels[min_detect_snowmelt[:, -1].argmin()]
                melt_conv = min_detect_snowmelt[:, -1][min_detect_snowmelt[:, -1].argmin()]
                o_sigma.append(melt_onset0)
                lv_sigma.append(melt_lvl0)
        # c_npr, c_sigma = [], []
        # o_npr, o_sigma, lv_sigma = [], [], []
        # plot
        npr_conv_array = np.concatenate((c_npr[0], c_npr[1], c_npr[2]), axis=1)
        sig_conv_array = np.concatenate((c_sigma[0], c_sigma[1], c_sigma[2]), axis=1)
        v_sigma_c_all = np.concatenate((v_sigma_c[0], v_sigma_c[1], v_sigma_c[2]))
        t_sigma_all = np.concatenate((t_sigma[0], t_sigma[1], t_sigma[2]))
        pause = 0
        if insitu:
            pause = 0
            path = '/home/xiyu/PycharmProjects/R3/result_08_01/plot_data'
            pixel_num = np.loadtxt('result_agu/result_2019/points_num.txt', delimiter=',').astype(int)
            pixel_no = pixel_num[0][pixel_num[1] == sid[l0]][0]
            snow = np.loadtxt('%s/%d_snow_%d.txt' % (path, pixel_no, sid[l0]))
            insitu_plot = get_3_year_insitu(sid[l0], m_name=insitu)
            insitu_plot2 = get_3_year_insitu(sid[l0], m_name="Soil Moisture Percent -8in (pct)")
            insitu_plot3 = get_3_year_insitu(sid[l0], m_name="Soil Moisture Percent -20in (pct)")
            plot_funcs.plot_subplot([[t_smap, npr_smap],
                                    [t_ascat, sigma0_correct_all], [insitu_plot[0], insitu_plot[1], insitu_plot2[0],
                                    insitu_plot2[1], insitu_plot3[0], insitu_plot3[1]]],
                                    [npr_conv_array, sig_conv_array, snow],
                                    main_label=['npr', '$\sigma^0$ mid', '$\sigma^0$ all_c'],
                                    figname='ms_station_test%d' % sid[l0], x_unit='mmdd',
                                    main_syb=['k.', 'r--', 'b--', 'r--', 'b--'],
                                    vline=[[], [], []])
        else:
            plot_funcs.plot_subplot([[t_smap, npr_smap],
                             [t_ascat, sigma0_correct_all], [t_sigma_all, v_sigma_c_all]],
                         [npr_conv_array, sig_conv_array], main_label=['npr', '$\sigma^0$ mid', '$\sigma^0$ all_c'],
                        figname='ms_station_test%d' % sid[l0], x_unit='mmdd')


def station_series(year0=2017):
    p_latlon = np.array([[-156.8, 62.3], [-159.8, 62.3]])
    N125_2d, N125_1d = bxy.latlon2index(p_latlon)
    ascat_dict_yearly = ms_read_ascat(year0)
    smap_yearly = get_yearly_smap(t_window=[0, 366], year0=year0)
    n125_n360, ascat_series, npr_series, smp_id = prepare_smap_series(input_ascat=ascat_dict_yearly, input_smap=smap_yearly)
    onsets, pixels = combine_detect(n125_n360, year0, ascat_series, npr_series)
    quit0()
    return 0


def quick_alaska_onset(year0):
    ascat_dict_yearly = ms_read_ascat(year0)
    smap_yearly = get_yearly_smap(t_window=[0, 366], year0=year0)
    n125_n360, ascat_series, npr_series, smp_id = prepare_smap_series(input_ascat=ascat_dict_yearly, input_smap=smap_yearly)
    nsets, pixels = combine_detect(n125_n360, year0, ascat_series, npr_series, save_sp=False)


def map_plot2(year0, p_latlon=np.array([[-99, -99.], [-99., 99.]]),
              p_index=[np.array([-1, -1]), np.array([-1, -1])],
              p_name=['test0', 'test1'], threshold_sigma=-3, onset_name=False, mode='normal',
              ):
    '''
    the mask include: sigma0_min, the minimum of sigma0 during melt_zone0
    :param year0:
    :param p_latlon:
    :param p_index:
    :param p_name:
    :param threshold_sigma:
    :param onset_name:
    :param mode:
    :return:
    '''
    if mode == 'quick':
        isplot=[True, True, False, False, False, False, False, False, True]
    else:
        isplot=[True, True, True, True, True, True, True, True, True]
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    if len(p_name) == 0:
        for ix in np.arange(0, p_latlon.shape[0]):
            p_name.append('test')
    # plot them in the map
    # ['smap_onset', 'coef_b', 'sigma0_summer', 'sigma0_winter', 'ascat_edge',
    # 'ascat_lvl', 'ascat_onset', 'ascat_melt_signal', 'smap_melt_signal', 'coef_a']
    value_npz = np.load('onset_all_%d.npz' % year0)
    if onset_name:
        value_npz = np.load(onset_name)
        fig_prefix = onset_name.split('_')[-1][0: -4] + mode
    else:
        fig_prefix = ''
    onset = value_npz['ascat_onset']  # arr_1
    ascat_winter = value_npz['sigma0_winter']
    v_valid = (onset!=0)&(onset!=-999)  # plot onset
    onset[v_valid] = bxy.time_getlocaltime(onset[v_valid], ref_time=[2000, 1, 1, 0])[-2]

    year0 = str(year0)
    winter, melt = value_npz['sigma0_winter'], value_npz['sigma0_melt_mean']
    winter_std, melt_std = value_npz['sigma0_winter_std'], value_npz['sigma0_melt_std']

    # ascat lvl
    lvl_value = value_npz['ascat_lvl'].copy()
    lvl_value[value_npz['ascat_lvl']<2] = -999
    lvl_array2 = data_process.make_mask(lvl_value)  # arr_3, plot level
    quick_plot_map_v2(lvl_array2, resolution=12.5, fig_name='finer_lvl_%s' % year0 + fig_prefix, z_value=[-2, 10],
                      points=p_latlon, points_index=p_index)

    # winter/summer std
    if isplot[0]:
        value_array2 = data_process.make_mask(value_npz['sigma0_winter_std'])  # arr_3, plot level
        quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_winter_std_%s' % year0 + fig_prefix,
                          z_value=[0, 1], points=p_latlon, points_index=p_index)
        value_array2 = data_process.make_mask(value_npz['sigma0_summer_std'])  # arr_3, plot level
        quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_summer_std_%s' % year0 + fig_prefix,
                          z_value=[0, 1], points=p_latlon, points_index=p_index)
        value_array2 = data_process.make_mask(0.6*value_npz['sigma0_summer_std'] + 0.4*value_npz['sigma0_winter_std'])
        quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_mean_std_%s' % year0 + fig_prefix,
                          z_value=[0, 1], points=p_latlon, points_index=p_index)


    # ascat (mean winter - min sigma0 on melt)
    if isplot[1]:
        compare_sigma = value_npz['sigma0_winter'] - value_npz['sigma0_min']
        compare_sigma[value_npz['sigma0_min'] < -9000] = -999
        diff_mask = compare_sigma.copy()
        mask_sigma_variation = data_process.make_mask(compare_sigma)
        quick_plot_map_v2(mask_sigma_variation, resolution=12.5, fig_name='finer_sigma0_drop_%s' % year0 + fig_prefix,
                          z_value=[-4, 8], points=p_latlon, points_index=p_index)

    # ascat (winter-2std - melt_min)
    if isplot[2]:
        compare_sigma = value_npz['sigma0_winter'] - value_npz['sigma0_winter_std'] - value_npz['sigma0_min']
        winter_melt_difference = compare_sigma.copy()
        compare_sigma[value_npz['sigma0_min'] < -9000] = -999
        std_mask = compare_sigma.copy()
        mask_sigma_variation = data_process.make_mask(compare_sigma)
        quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                          fig_name='finer_difference_double_std_%s' % year0 + fig_prefix,
                          z_value=[-4, 4], points=p_latlon, points_index=p_index)

    # ascat (melt-std - winter)/winter_std
    if isplot[3]:
        compare_sigma = (value_npz['sigma0_winter'] + value_npz['sigma0_melt_std'] - value_npz['sigma0_melt_mean']) / \
                         value_npz['sigma0_winter_std']
        winter_melt_difference = compare_sigma.copy()
        compare_sigma[value_npz['sigma0_min'] < -9000] = -999
        std_mask = compare_sigma.copy()
        mask_sigma_variation = data_process.make_mask(compare_sigma)
        quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                          fig_name='finer_winter_and_melt_std02snr_%s' % year0 + fig_prefix,
                          z_value=[-4, 4], points=p_latlon, points_index=p_index)

    # ascat 0.5 (winter_std + summer_std)
    if isplot[4]:
        compare_sigma = 0.5*(value_npz['sigma0_winter_std'] + value_npz['sigma0_summer_std'])
        compare_sigma[value_npz['sigma0_min'] < -9000] = -999
        mask_sigma_variation = data_process.make_mask(compare_sigma)
        quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                          fig_name='finer_winter_and_melt_std03_mean_%s' % year0 + fig_prefix,
                          z_value=[0, 1], points=p_latlon, points_index=p_index)

    # ascat difference (melt - winter)
    if isplot[5]:
        compare_sigma = value_npz['sigma0_winter'] - value_npz['sigma0_melt_mean']
        compare_sigma[value_npz['sigma0_min'] < -9000] = -999
        mask_sigma_variation = data_process.make_mask(compare_sigma)
        quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                          fig_name='finer_winter_and_melt_difference_%s' % year0 + fig_prefix,
                          z_value=[-4, 4], points=p_latlon, points_index=p_index)

    # ascat onset
    onset[value_npz['ascat_lvl'] < 1.5] = -999  # onset with lvl < 1.5 are masked
    mask_01 = [(value_npz['ascat_lvl'] < 2) |
               (value_npz['sigma0_min'] < -9000) |
               ((winter_std > 0.5) & (diff_mask < 3*winter_std))|
               (winter_std > 0.75)]  # < 6*winter_std
    mask_02 = [(diff_mask < 1) |
               (value_npz['sigma0_min'] < -9000)]
    mask_03 = [(value_npz['ascat_lvl'] < 2) |
               (value_npz['sigma0_min'] < -9000)]
    onset[mask_01] = -999
    # onset[compare_sigma > threshold_sigma] = -999
    value_array = data_process.make_mask(onset)
    quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160],
                      fig_name='finer_ascat_%s_%d' % (year0 + fig_prefix, 0),
                      points=p_latlon, p_name=p_name, points_index=p_index)

    # ascat winter
    if isplot[6]:
        winter_array = data_process.make_mask(ascat_winter)
        quick_plot_map_v2(winter_array, resolution=12.5, fig_name='winter_%s' % year0 + fig_prefix,
                          z_value=[-20, -5], points=p_latlon, points_index=p_index, p_name=p_name)

    # ascat kernels
    if isplot[7]:
        kernel1, kernel2 = value_npz['sigma0_kernels'][0], value_npz['sigma0_kernels'][1]
        kernel1_grid = data_process.make_mask(kernel1)
        kernel2_grid = data_process.make_mask(kernel2)
        quick_plot_map_v2(kernel1_grid, resolution=12.5, fig_name='kernels_1_%s' % year0 + fig_prefix,
                          z_value=[0, 15], points=p_latlon, points_index=p_index, p_name=p_name)
        quick_plot_map_v2(kernel2_grid, resolution=12.5, fig_name='kernels_2_%s' % year0 + fig_prefix,
                          z_value=[0, 15], points=p_latlon, points_index=p_index, p_name=p_name)
    if isplot[8]:
        onset_correct = bxy.time_getlocaltime(value_npz['melt_events_time'][:, 0])[-2]
        onset_correct[value_npz['ascat_lvl'] < 1.5] = -999  # onset with lvl < 1.5 are masked
        onset_correct[value_npz['melt_events_time'][:, 0] == -9999] = -999
        # onset_correct[mask_01] = -999  #
        # onset[compare_sigma > threshold_sigma] = -999
        value_array = data_process.make_mask(onset_correct)
        quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160],
                          fig_name='finer_ascat_correct_%s_%d' % (year0 + fig_prefix, 0),
                          points=p_latlon, p_name=p_name, points_index=p_index)
    if mode == 'quick':
        print 'quick location has been done'
        return 0

    # other masks
    compare_sigma = (value_npz['sigma0_winter'] - value_npz['sigma0_melt_mean']) / \
             np.sqrt(value_npz['sigma0_winter_std']**2 + value_npz['sigma0_melt_std']**2)
    winter_melt_difference = compare_sigma.copy()
    compare_sigma[value_npz['sigma0_min'] < -9000] = -999
    mask_sigma_variation = data_process.make_mask(compare_sigma)
    # fig_name='finer_winter_std_sigma0_%s'
    # 'finer_winter_and_melt_%s'
    quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                      fig_name='finer_winter_and_melt_std03root_%s' % year0 + fig_prefix,
                      z_value=[-4, 4], points=p_latlon, points_index=p_index)

    compare_sigma = value_npz['sigma0_winter'] + value_npz['sigma0_melt_std'] - value_npz['sigma0_melt_mean']
    winter_melt_difference = compare_sigma.copy()
    compare_sigma[value_npz['sigma0_min'] < -9000] = -999
    mask_sigma_variation = data_process.make_mask(compare_sigma)
    # fig_name='finer_winter_std_sigma0_%s'
    # 'finer_winter_and_melt_%s'
    quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                      fig_name='finer_winter_and_melt_diff_%s' % year0 + fig_prefix,
                      z_value=[-4, 4], points=p_latlon, points_index=p_index)

    # compare winter and summmer sigma0 variation
    summer = value_npz['sigma0_winter']
    # compare_sigma = melt-melt_std - (winter - winter_std)
    # compare_sigma = (melt- melt_std -winter)/winter_std
    compare_sigma = summer-winter
    compare_sigma[value_npz['sigma0_min'] < -9000] = -999
    mask_sigma_variation = data_process.make_mask(compare_sigma)
    quick_plot_map_v2(mask_sigma_variation, resolution=12.5, fig_name='fine_sigma0_difference_%s' % year0 + fig_prefix,
                      z_value=[-4, 4], points=p_latlon, points_index=p_index)
    # smap onset
    h0 = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102')
    lons_grid = h0['cell_lon'].value
    onset_array = np.zeros(lons_grid.shape)
    onset_s = value_npz['smap_onset']
    onset_s_doy = np.zeros(onset_s.shape)
    valid = onset_s > 1e5
    onset_s_doy[valid] = bxy.time_getlocaltime(onset_s[valid])[-2]
    mask = np.load(('./result_05_01/other_product/mask_ease2_360N.npy'))
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]
    land_2d = np.unravel_index(land_id, mask.shape)
    onset_array[land_2d] = onset_s_doy
    quick_plot_map_v2(onset_array, resolution=36, fig_name='smap_%s' % year0 + fig_prefix,
                      z_value=[30, 160], points=p_latlon, p_name=p_name, points_index=p_index)
    p_name=0
    return 0


def map_plot1(year0, p_latlon=np.array([[-99, -99.], [-99., 99.]]),
              p_index=[np.array([-1, -1]), np.array([-1, -1])],
              p_name=['test0', 'test1'], threshold_sigma=-3, onset_name=False, mode='normal',
              ):
    '''
    the mask include: sigma0_min, the minimum of sigma0 during melt_zone0
    :param year0:
    :param p_latlon:
    :param p_index:
    :param p_name:
    :param threshold_sigma:
    :param onset_name:
    :param mode:
    :return:
    '''
    if mode == 'quick':
        isplot=[False, False, False, False, False, False, False, False, True]
    else:
        isplot=[True, True, True, True, True, True, True, True, True]
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    if len(p_name) == 0:
        for ix in np.arange(0, p_latlon.shape[0]):
            p_name.append('test')
    # plot them in the map
    # ['smap_onset', 'coef_b', 'sigma0_summer', 'sigma0_winter', 'ascat_edge',
    # 'ascat_lvl', 'ascat_onset', 'ascat_melt_signal', 'smap_melt_signal', 'coef_a']
    value_npz = np.load('onset_all_%d.npz' % year0)
    if onset_name:
        value_npz = np.load(onset_name)
        fig_prefix = onset_name.split('_')[-1][0: -4] + mode
    else:
        fig_prefix = ''
    onset_secs = value_npz['ascat_onset']  # arr_1
    onset = np.zeros(onset_secs.size) - 999
    ascat_winter = value_npz['sigma0_mean_winter']
    ascat_mean_melt = value_npz['sigma0_mean_melt_zone']
    ascat_min_melt = value_npz['sigma0_min_melt_zone']
    conv_min_winter = value_npz['winter_edge']
    v_valid = (onset_secs!=0)&(onset_secs!=-999)  # plot onset
    onset[v_valid] = bxy.time_getlocaltime(onset_secs[v_valid], ref_time=[2000, 1, 1, 0])[-2]

    year0 = str(year0)
    # winter, melt = ascat_winter, value_npz['sigma0_melt_mean']
    # winter_std, melt_std = value_npz['sigma0_winter_std'], value_npz['sigma0_melt_std']
    winter, melt = ascat_winter, value_npz['sigma0_mean_melt_zone']
    winter_std, melt_std, summer_std = value_npz['sigma0_std_winter'], value_npz['simga0_std_melt_zone'], \
                                       value_npz['sigma_std_summer']

    # quick
    # ascat lvl
    # lvl_value = value_npz['ascat_lvl'].copy()
    # lvl_value = value_npz['lvl_on_melt_date'].copy()
    lvl_value_ratio = value_npz['conv_on_melt_date']/value_npz['winter_conv_std']
    lvl_value_ab = value_npz['conv_on_melt_date']
    lvl_value = lvl_value_ab
    # lvl_value[value_npz['lvl_on_melt_date']<2] = -999
    lvl_array2 = data_process.make_mask(lvl_value)  # arr_3, plot level
    quick_plot_map_v2(lvl_array2, resolution=12.5, fig_name='finer_lvl_%s' % year0 + fig_prefix, z_value=[-2, 10],
                      points=p_latlon, points_index=p_index)

    # winter/summer std
    if isplot[0]:
        value_array2 = data_process.make_mask(winter_std)  # arr_3, plot level
        quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_winter_std_%s' % year0 + fig_prefix,
                          z_value=[0, 1], points=p_latlon, points_index=p_index)
        value_array2 = data_process.make_mask(summer_std)  # arr_3, plot level
        quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_summer_std_%s' % year0 + fig_prefix,
                          z_value=[0, 1], points=p_latlon, points_index=p_index)
        value_array2 = data_process.make_mask(0.6*summer_std + 0.4*winter_std)
        quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_mean_std_%s' % year0 + fig_prefix,
                          z_value=[0, 1], points=p_latlon, points_index=p_index)


    # ascat (mean winter - min sigma0 on melt)
    if isplot[1]:
        compare_sigma = ascat_winter - ascat_min_melt
        compare_sigma[ascat_min_melt < -9000] = -999
        diff_mask = compare_sigma.copy()
        mask_sigma_variation = data_process.make_mask(compare_sigma)
        quick_plot_map_v2(mask_sigma_variation, resolution=12.5, fig_name='finer_sigma0_drop_%s' % year0 + fig_prefix,
                          z_value=[-4, 8], points=p_latlon, points_index=p_index)

    # ascat (winter-2std - melt_min)
    if isplot[2]:
        compare_sigma = ascat_winter - winter_std - ascat_min_melt
        winter_melt_difference = compare_sigma.copy()
        compare_sigma[ascat_min_melt < -9000] = -999
        std_mask = compare_sigma.copy()
        mask_sigma_variation = data_process.make_mask(compare_sigma)
        quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                          fig_name='finer_difference_double_std_%s' % year0 + fig_prefix,
                          z_value=[-4, 4], points=p_latlon, points_index=p_index)

    # ascat (melt-std - winter)/winter_std
    # if isplot[3]:
    #     compare_sigma = (value_npz['sigma0_winter'] + value_npz['sigma0_melt_std'] - value_npz['sigma0_melt_mean']) / \
    #                      value_npz['sigma0_winter_std']
    #     winter_melt_difference = compare_sigma.copy()
    #     compare_sigma[value_npz['sigma0_min'] < -9000] = -999
    #     std_mask = compare_sigma.copy()
    #     mask_sigma_variation = data_process.make_mask(compare_sigma)
    #     quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
    #                       fig_name='finer_winter_and_melt_std02snr_%s' % year0 + fig_prefix,
    #                       z_value=[-4, 4], points=p_latlon, points_index=p_index)

    # ascat 0.5 (winter_std + summer_std)
    # if isplot[4]:
    #     compare_sigma = 0.5*(value_npz['sigma0_winter_std'] + value_npz['sigma0_summer_std'])
    #     compare_sigma[value_npz['sigma0_min'] < -9000] = -999
    #     mask_sigma_variation = data_process.make_mask(compare_sigma)
    #     quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
    #                       fig_name='finer_winter_and_melt_std03_mean_%s' % year0 + fig_prefix,
    #                       z_value=[0, 1], points=p_latlon, points_index=p_index)

    # ascat difference (melt - winter)
    if isplot[5]:
        compare_sigma = ascat_winter - ascat_mean_melt
        compare_sigma[ascat_min_melt< -9000] = -999
        mask_sigma_variation = data_process.make_mask(compare_sigma)
        quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                          fig_name='finer_winter_and_melt_difference_%s' % year0 + fig_prefix,
                          z_value=[-4, 4], points=p_latlon, points_index=p_index)

    # ascat onset
    # onset with lvl < 1.5 are masked
    # mask_01 = [(lvl_value > -1) |
    #            (ascat_min_melt < -9000) |
    #            (winter_std > 0.75)]  # < 6*winter_std
    # onset[mask_01] = -999
    # onset[compare_sigma > threshold_sigma] = -999
    value_array = data_process.make_mask(onset)
    quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160],
                      fig_name='finer_ascat_%s_%d' % (year0 + fig_prefix, 0),
                      points=p_latlon, p_name=p_name, points_index=p_index)
    onset_convs = value_npz['conv_on_melt_date']
    onset_convs[(onset_secs!=0)&(onset_secs!=-999)] = -999
    value_array2 = data_process.make_mask(onset_convs)
    quick_plot_map_v2(value_array2, resolution=12.5, z_value=[-10, 0],
                      fig_name='finer_ascat_%s_%d_convs' % (year0 + fig_prefix, 0),
                      points=p_latlon, p_name=p_name, points_index=p_index)

    # ascat winter
    if isplot[6]:
        winter_array = data_process.make_mask(ascat_winter)
        quick_plot_map_v2(winter_array, resolution=12.5, fig_name='winter_%s' % year0 + fig_prefix,
                          z_value=[-20, -5], points=p_latlon, points_index=p_index, p_name=p_name)

    # ascat kernels
    if isplot[7]:
        kernel1, kernel2 = value_npz['sigma0_kernels'][0], value_npz['sigma0_kernels'][1]
        kernel1_grid = data_process.make_mask(kernel1)
        kernel2_grid = data_process.make_mask(kernel2)
        quick_plot_map_v2(kernel1_grid, resolution=12.5, fig_name='kernels_1_%s' % year0 + fig_prefix,
                          z_value=[0, 15], points=p_latlon, points_index=p_index, p_name=p_name)
        quick_plot_map_v2(kernel2_grid, resolution=12.5, fig_name='kernels_2_%s' % year0 + fig_prefix,
                          z_value=[0, 15], points=p_latlon, points_index=p_index, p_name=p_name)
    if isplot[8]:
        # correct_secs = value_npz['melt_events_time'][0]
        # correct_conv = value_npz['melt_events_conv'][0]
        correct_secs = value_npz['melt_events_time'].T
        conv_events = value_npz['melt_events_conv'].T
        # corrected onsets, convolution value should less than -1
        secs_valid_events = [correct_secs_0[(conv0 < -1) & (conv0 > -99)]
                             for correct_secs_0, conv0 in zip(correct_secs, conv_events)]
        onsets_valid = np.array([event0[0] if event0.size > 0 else -999 for event0 in secs_valid_events])
        onset_correct_temp = bxy.time_getlocaltime(onsets_valid)[-2]
        onset_correct_temp[onsets_valid < -99] = -999
        onset_correct = onset.copy()
        onset_correct[onset_correct_temp != -999] = onset_correct_temp[onset_correct_temp != -999]
        # convolution value on corrected onsets
        conv_valid_events = [conv0[(conv0 < -1) & (conv0 > -99)] for conv0 in conv_events]
        conv_valid_first = [cv0[0] if cv0.size > 0 else -999 for cv0 in conv_valid_events]
        valid_conv_on_onsets = np.array([event0[0] if event0.size>0 else -999 for event0 in secs_valid_events])
        # signifcant level
        correct_conv0 = value_npz['conv_on_melt_date']
        correct_conv0[onset_correct_temp != -999] = -1  # pixels with the newly corrected onset
        correct_lvl = correct_conv0/-1
        un_valid_id = correct_conv0 < -99  # mask unvalid id
        correct_lvl[un_valid_id] = -999


        # plotting
        value_array = data_process.make_mask(onset_correct)
        value_array2 = data_process.make_mask(valid_conv_on_onsets)
        value_array3 = data_process.make_mask(correct_lvl)
        quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160],
                          fig_name='finer_ascat_correct_%s_%d' % (year0 + fig_prefix, 0),
                          points=p_latlon, p_name=p_name, points_index=p_index)
        quick_plot_map_v2(value_array2, resolution=12.5, z_value=[-10, 0],
                          fig_name='finer_ascat_correct_%s_%d_convs' % (year0 + fig_prefix, 0),
                          points=p_latlon, p_name=p_name, points_index=p_index)
        quick_plot_map_v2(value_array3, resolution=12.5, z_value=[0, 1],
                          fig_name='finer_ascat_correct_%s_%d_new_lvl' % (year0 + fig_prefix, 0),
                          points=p_latlon, p_name=p_name, points_index=p_index)
    if mode == 'quick':
        print 'quick location has been done'
        return 0

    # other masks
    compare_sigma = (value_npz['sigma0_winter'] - value_npz['sigma0_melt_mean']) / \
             np.sqrt(value_npz['sigma0_winter_std']**2 + value_npz['sigma0_melt_std']**2)
    winter_melt_difference = compare_sigma.copy()
    compare_sigma[value_npz['sigma0_min'] < -9000] = -999
    mask_sigma_variation = data_process.make_mask(compare_sigma)
    # fig_name='finer_winter_std_sigma0_%s'
    # 'finer_winter_and_melt_%s'
    quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                      fig_name='finer_winter_and_melt_std03root_%s' % year0 + fig_prefix,
                      z_value=[-4, 4], points=p_latlon, points_index=p_index)

    compare_sigma = value_npz['sigma0_winter'] + value_npz['sigma0_melt_std'] - value_npz['sigma0_melt_mean']
    winter_melt_difference = compare_sigma.copy()
    compare_sigma[value_npz['sigma0_min'] < -9000] = -999
    mask_sigma_variation = data_process.make_mask(compare_sigma)
    # fig_name='finer_winter_std_sigma0_%s'
    # 'finer_winter_and_melt_%s'
    quick_plot_map_v2(mask_sigma_variation, resolution=12.5,
                      fig_name='finer_winter_and_melt_diff_%s' % year0 + fig_prefix,
                      z_value=[-4, 4], points=p_latlon, points_index=p_index)

    # compare winter and summmer sigma0 variation
    summer = value_npz['sigma0_winter']
    # compare_sigma = melt-melt_std - (winter - winter_std)
    # compare_sigma = (melt- melt_std -winter)/winter_std
    compare_sigma = summer-winter
    compare_sigma[value_npz['sigma0_min'] < -9000] = -999
    mask_sigma_variation = data_process.make_mask(compare_sigma)
    quick_plot_map_v2(mask_sigma_variation, resolution=12.5, fig_name='fine_sigma0_difference_%s' % year0 + fig_prefix,
                      z_value=[-4, 4], points=p_latlon, points_index=p_index)
    # smap onset
    h0 = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102')
    lons_grid = h0['cell_lon'].value
    onset_array = np.zeros(lons_grid.shape)
    onset_s = value_npz['smap_onset']
    onset_s_doy = np.zeros(onset_s.shape)
    valid = onset_s > 1e5
    onset_s_doy[valid] = bxy.time_getlocaltime(onset_s[valid])[-2]
    mask = np.load(('./result_05_01/other_product/mask_ease2_360N.npy'))
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]
    land_2d = np.unravel_index(land_id, mask.shape)
    onset_array[land_2d] = onset_s_doy
    quick_plot_map_v2(onset_array, resolution=36, fig_name='smap_%s' % year0 + fig_prefix,
                      z_value=[30, 160], points=p_latlon, p_name=p_name, points_index=p_index)
    p_name=0
    return 0


def map_plot1_check_pixel(npz_name, pixel_ind, key_list, ind_table, year0=2016):
    # read npz file
    print 'read variables form npz file', key_list, 'pixels: ', pixel_ind
    loaded_npz = np.load(npz_name)
    lons_grid, lats_grid, _ = ind2latlon(pixel_ind, resolution=12.5)
    coordinate_heads = ',longitude,latitude'
    coordinates_array = np.array([lons_grid.ravel()[pixel_ind], lats_grid.ravel()[pixel_ind]])
    smap_ind_heads = ',smap_index'
    index_in_array = []
    smap_indices = np.zeros(pixel_ind.size)-1
    i_pixel = -1
    for ind0 in pixel_ind:
        i_pixel += 1
        index_in_array.append(np.where(ind_table[0] == ind0)[0][0])
        smap_indices[i_pixel] = ind_table[1][ind_table[0] == ind0]
    array_check_pixel = [loaded_npz[key0][index_in_array] for key0 in key_list]
    array_check_pixel = np.array(array_check_pixel)
    out_array = np.vstack((pixel_ind, array_check_pixel, coordinates_array, smap_indices))
    head0 = ','.join(key_list)
    np.savetxt('map_plot_check_pixel_%s.txt' % (year0), out_array.T,
               delimiter=',', fmt='%.3f', header='pixel_index,'+head0+coordinate_heads+smap_ind_heads)
    return out_array


def map_plot1_mask_test():
    std_th = 7
    onset[value_npz['ascat_lvl'] < 1.5] = -999  # onset with lvl < 1.5 are masked
    onset[(value_npz['sigma0_winter'] - value_npz['sigma0_min'] < 1) |
          (value_npz['sigma0_min'] < -9000) |
          (winter_std10 > std_th)] = -999
    # onset[compare_sigma > threshold_sigma] = -999
    value_array = data_process.make_mask(onset)
    quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160],
                      fig_name='finer_ascat_%s_%d' % (year0 + fig_prefix, std_th),
                      points=p_latlon, p_name=p_name, points_index=p_index)

    std_th = 6
    onset[value_npz['ascat_lvl'] < 1.5] = -999  # onset with lvl < 1.5 are masked
    onset[(value_npz['sigma0_winter'] - value_npz['sigma0_min'] < 1) |
          (value_npz['sigma0_min'] < -9000) |
          (winter_std10 > std_th)] = -999
    # onset[compare_sigma > threshold_sigma] = -999
    value_array = data_process.make_mask(onset)
    quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160],
                      fig_name='finer_ascat_%s_%d' % (year0 + fig_prefix, std_th),
                      points=p_latlon, p_name=p_name, points_index=p_index)


def ms_station_new(sno_all=False, pixel_type='interest', detectors=False, site_plot='single'):
    """
    Read the saved npz file, plot time seris at the specific station.
    :param sno_all:
    :param pixel_type: outlier or interest (station)
    :param site_plot: Plot the pixel with a station in two types: single: plot one by one with in situ measurements
                     'grid': plot all interested station in to a gridded figure, without in situ measurements
    :return:
        smap pixel:
    """
    # initials
    ind_mat = []
    # structure of npz file:
    # [0 time of measurements, 1 value of measurements, 2 time & value of convolution]
    # ['output_array', 'ascat_pixel', 'smap_pixel']
    # i.e., [onset_array_smap, onset_array_ascat, conv_array, lvl_array], [smap_pixel, sigma0_pixel]
    station_z_2016 = np.load('onset_%s_%d.npz' % (pixel_type, 2016))
    station_z_2017 = np.load('onset_%s_%d.npz' % (pixel_type, 2017))
    station_z_2018 = np.load('onset_%s_%d.npz' % (pixel_type, 2018))
    # check the number of pixels
    station_array = np.loadtxt('npz_indices_interest.txt', delimiter=',')
    station_list = station_array[2]
    pixel_name_int = station_list.astype(int)
    if pixel_type == 'outlier':
        outlier_info = np.loadtxt('npz_indices_outlier.txt', delimiter=',')
        station_array = outlier_info[[1, 0]]
        pixel_name_int = outlier_info[0].astype(int)  # we named those pixels with their smap indices (36N ease grid)
        print 'checking the outliers time series: ', pixel_name_int
    # prepare time series and plot
    if sno_all is not False:
        print 'now check the specified stations'
    elif pixel_type=='interest':
        print 'all interested station are checked'
        sno_all = site_infos.get_id(int)
    elif pixel_type=='outlier':
        sno_all = outlier_info[0].astype(int)

    if station_z_2016['output_array'][0].size == station_list.size:
        print 'all the 1-d index of pixels are saved'
    else:
        pause = 0
        # print 'the number of saved pixels is %d, but the number of saved pixel indices is %d' % \
        #       (station_z_2016['output_array'][0].size, station_list.size)
    site_order = -1
    for sno0 in sno_all:
        if sno0 == 30507:
            pause = 0
        site_order += 1
        print 'station number %d' % (sno0)
        i0 = np.where(pixel_name_int == int(sno0))[0][0]
        smap_id = station_array[0][i0].astype(int)
        ascat_id = station_array[1][i0].astype(int)

        npr_time = ms_ca_station(station_z_2016, station_z_2017, station_z_2018, 'smap_pixel', i0, 0)
        npr_value = ms_ca_station(station_z_2016, station_z_2017, station_z_2018, 'smap_pixel', i0, 1)
        npr_time[npr_time < 0] == np.nan
        npr_plot = np.array([npr_time, npr_value])

        npr_conv_plot = ms_ca_station(station_z_2016, station_z_2017, station_z_2018, 'smap_pixel', i0, 2)
        positive_edge = np.array([station_z_2016['output_array'][0, i0], station_z_2017['output_array'][0, i0],
                                station_z_2018['output_array'][0, i0]])

        sigma_time = ms_ca_station(station_z_2016, station_z_2017, station_z_2018, 'ascat_pixel', i0, 0)
        sigma_value = ms_ca_station(station_z_2016, station_z_2017, station_z_2018, 'ascat_pixel', i0, 1)
        sigma_time[sigma_time<0] == np.nan
        sigma_plot = np.array([sigma_time, sigma_value])

        sigma_conv_plot = ms_ca_station(station_z_2016, station_z_2017, station_z_2018, 'ascat_pixel', i0, 2)
        negative_edge = np.array([station_z_2016['output_array'][1, i0], station_z_2017['output_array'][1, i0],
                                    station_z_2018['output_array'][1, i0]])
        positive_edge2 = np.array([station_z_2016['output_array'][2, i0], station_z_2017['output_array'][2, i0],
                                    station_z_2018['output_array'][2, i0]])


        if detectors is not False:
            ascat_satellite = 4
            # keys: [ini_onset, main_onset, end_onset, sigma0_winter, sigma0_melt], pixels
            print 'do detection in year 2016'
            onset_2016_new, pixel_2016_new = data_process.re_detection(station_z_2016,
                                                                       np.array([[ascat_id], [smap_id]]), i0, 2016,
                                                                       kernels=detectors, sigma0_type=ascat_satellite)
            print 'do detection in year 2017'
            onset_2017_new, pixel_2017_new = data_process.re_detection(station_z_2017,
                                                                       np.array([[ascat_id], [smap_id]]), i0, 2017,
                                                                       kernels=detectors, sigma0_type=ascat_satellite)
            print 'do detection in year 2018'
            onset_2018_new, pixel_2018_new = data_process.re_detection(station_z_2018,
                                                                       np.array([[ascat_id], [smap_id]]), i0, 2018,
                                                                       kernels=detectors, sigma0_type=ascat_satellite)
            # pixel index guide: [ascat/smap][pixel_no][t/value/convolutions]; onset index: [pixel_no][keys]
            npr_plot = np.array([pixel_2016_new[0][0][0], pixel_2016_new[0][0][1],
                                 pixel_2017_new[0][0][0], pixel_2017_new[0][0][1],
                                 pixel_2018_new[0][0][0], pixel_2018_new[0][0][1]])
            npr_conv_plot = [pixel_2016_new[0][0][2],
                             pixel_2017_new[0][0][2],
                             pixel_2018_new[0][0][2]]
            npr_conv_bar = np.vstack((pixel_2016_new[0][0][3],
                                      pixel_2017_new[0][0][3],
                                      pixel_2018_new[0][0][3]))
            t_2016, v_2016 = bxy.remove_unvalid_time(pixel_2016_new[1][0][0], pixel_2016_new[1][0][1])
            t_2017, v_2017 = bxy.remove_unvalid_time(pixel_2017_new[1][0][0], pixel_2017_new[1][0][1])
            t_2018, v_2018 = bxy.remove_unvalid_time(pixel_2018_new[1][0][0], pixel_2018_new[1][0][1])
            sigma_plot = np.array([t_2016, v_2016, t_2017, v_2017, t_2018, v_2018])
            sigma_conv_plot = [pixel_2016_new[1][0][2],
                               pixel_2017_new[1][0][2],
                               pixel_2018_new[1][0][2]]
            simga_conv_plot_more = [pixel_2016_new[1][0][3],
                                   pixel_2017_new[1][0][3],
                                   pixel_2018_new[1][0][3]]
            sigma_conv_bar_max = np.vstack((pixel_2016_new[1][0][4],
                                            pixel_2017_new[1][0][4],
                                            pixel_2018_new[1][0][4]))
            sigma_conv_bar_min = np.vstack((pixel_2016_new[1][0][5],
                                            pixel_2017_new[1][0][5],
                                            pixel_2018_new[1][0][5]))
            sigma_qa_w = [onset_2016_new[3], onset_2017_new[3], onset_2018_new[3]]
            sigma_qa_m = [onset_2016_new[4], onset_2017_new[4], onset_2018_new[4]]
            text_qa_w, text_qa_m = ['%.2f$\pm$\n%.2f' % (list0[0], list0[1]) for list0 in sigma_qa_w], \
                                   ['%.2f$\pm$\n%.2f' % (list0[0], list0[1]) for list0 in sigma_qa_m]
            # calculate data quality indicator: [(mean1-std1)-mean0]/std0
            indicator0 = [[(list_m[0] - list_m[1] - list_w[0])/list_w[1]][0][0]
                          for list_w, list_m in zip(sigma_qa_w,  sigma_qa_m)]
            ind_array = np.zeros(5)
            ind_array[0: 2] = np.array([int(sno0), smap_id])
            ind_array[2:] = indicator0
            ind_mat.append(ind_array)
            # np.savetxt('sigma_variation_indicator.txt', np.array([]))
            # simga_conv_plot.shape = 6, -1
            print onset_2016_new
            positive_edge = np.array([onset_2016_new[0][0], onset_2017_new[0][0], onset_2018_new[0][0]])
            negative_edge = np.array([onset_2016_new[1][0], onset_2017_new[1][0], onset_2018_new[1][0]])
            positive_edge2 = np.array([onset_2016_new[2][0], onset_2017_new[2][0], onset_2018_new[2][0]])
        # plotting
        # value for timing
        doy_p, doy_n, doy_end = bxy.time_getlocaltime(positive_edge)[-2], \
                                bxy.time_getlocaltime(negative_edge, ref_time=[2000, 1, 1, 0])[-2], \
                                bxy.time_getlocaltime(positive_edge2, ref_time=[2000, 1, 1, 0])[-2]
        doy_all = np.concatenate((doy_p, doy_n, doy_end))
        sec_all = np.concatenate((positive_edge, negative_edge, positive_edge2))

        v_line_local = [sec_all, ['k-', 'k-', 'k-', 'r-', 'r-', 'r-', 'b-', 'b-', 'b-'],
                        ['p', 'p', 'p', 'n', 'n', 'n', 'p1', 'p1', 'p1']]
        # read measurement
        if pixel_type == 'interest':
            if site_plot == 'single':
                insitu_plot2 = get_3_year_insitu(int(sno0), m_name='snow')
                insitu_plot = get_3_year_insitu(int(sno0), m_name="air")
                plot_funcs.plot_subplot([npr_plot, sigma_plot, insitu_plot[0:2]],
                                        [[npr_conv_plot[0], npr_conv_plot[1], npr_conv_plot[2]],
                                         [sigma_conv_plot[0], sigma_conv_plot[1], sigma_conv_plot[2]],
                                         insitu_plot2[0:2]],
                                        main_label=['npr', '$\sigma^0$ mid', 'snow'],
                                        figname='ms_pixel_test_%d_%d' % (sno0, smap_id), x_unit='doy', vline=v_line_local,
                                        vline_label=doy_all, h_line=[[-1], [0], [':']], y_lim=[[1], [[-20, -6]]]
                                        )
            # plotting 3*2
            elif site_plot == 'grid':
                subplot_loc = np.unravel_index(site_order, (3, 2))
                ax = plt.subplot2grid((3, 2), subplot_loc)
                ax_2 = ax.twinx()
                ax.plot(npr_plot[0], npr_plot[1]*100, 'k.',  npr_plot[2], npr_plot[3]*100, 'k.',
                        npr_plot[4], npr_plot[5]*100, 'k.', markersize=2)
                ax_2.plot(sigma_plot[0], sigma_plot[1], 'b.', sigma_plot[2], sigma_plot[3], 'b.',
                          sigma_plot[4], sigma_plot[5], 'b.', markersize=2)
                ax.set_ylim([-5, 5])
                ax_2.set_ylim([-15, 10])
                ax.text(0.5, 0.5, str(sno0), transform=ax.transAxes, va='top', ha='left', fontsize=16)
                # ax_2.plot(**sigma_plot)
                if site_order == 5:
                    plt.savefig('gridded_plot')

        elif pixel_type == 'outlier':
            valid0 = npr_plot[1] > 0
            # plot_funcs.plot_subplot([npr_plot[:, valid0], sigma_plot],
            #                         [npr_conv_plot, simga_conv_plot],
            #                         main_label=['npr', '$\sigma^0$ mid'],
            #                         figname='ms_pixel_test_%s_%d' % (sno0, smap_id), x_unit='doy', vline=v_line_local, vline_label=doy_all,
            #                         h_line=[[-1], [0], [':']]
            #                         )
            if detectors:
                # plot_funcs.plot_subplot([npr_plot, sigma_plot, sigma_plot],
                #                         [[npr_conv_plot[0], npr_conv_plot[1], npr_conv_plot[2]],
                #                          [sigma_conv_plot[0], sigma_conv_plot[1], sigma_conv_plot[2]],
                #                          [simga_conv_plot_more[0], simga_conv_plot_more[1], simga_conv_plot_more[2]]],
                #                         main_label=['npr', '$\sigma^0$ mid', '$\sigma^0$'],
                #                         figname='ms_pixel_test_%d_%d' % (sno0, smap_id), x_unit='doy',
                #                         vline=v_line_local, vline_label=doy_all,
                #                         h_line2=[[0, 1, 2], [0.01, -1, 1], [':', ':', ':']],
                #                         annotation_sigma0=[text_qa_w, text_qa_m],
                #                         y_lim=[[0, 1], [[0, 0.1], [-18, -4]]],
                #                         y_lim2=[[0, 1, 2], [[0, 0.05], [-6, 0], [0, 6]]]
                #                         )
                # npr_conv_bar, sigma_conv_bar_max
                plot_funcs.plot_subplot([npr_plot, sigma_plot, sigma_plot],
                                        [[npr_conv_bar[:, 1], npr_conv_bar[:, 2]],
                                         [sigma_conv_bar_max[:, 1], sigma_conv_bar_max[:, 2]],
                                         [sigma_conv_bar_min[:, 1], sigma_conv_bar_min[:, 2]]],
                                        main_label=['npr', '$\sigma^0$ mid', '$\sigma^0$'],
                                        figname='ms_pixel_test_%d_%d' % (sno0, smap_id), x_unit='doy',
                                        vline=v_line_local, vline_label=doy_all,
                                        h_line2=[[0, 1, 2], [0.01, 1, -1], [':', ':', ':']],
                                        annotation_sigma0=[text_qa_w, text_qa_m],
                                        y_lim=[[0, 1, 2], [[0, 0.1], [-18, -4], [-18, -4]]],
                                        y_lim2=[[0, 1, 2], [[0, 0.05], [0, 6], [-6, 0]]],
                                        type_y2='bar'
                                        )
            else:
                valid0 = npr_plot[1] > 0
                plot_funcs.plot_subplot([npr_plot[:, valid0], sigma_plot],
                                        [npr_conv_plot, sigma_conv_plot],
                                        main_label=['npr', '$\sigma^0$ mid'],
                                        figname='ms_pixel_test_%d_%d' % (sno0, smap_id), x_unit='doy', vline=v_line_local, vline_label=doy_all,
                                        h_line=[[-1], [0], [':']]
                                        )

        np.savetxt('sigma_variation_indicator.txt', np.array(ind_mat), delimiter=',', fmt='%.2f')
        # text='lat/lon: %.2f, %.2f' % (p_latlon[i0, 1], p_latlon[i0, 0])


def check_map_pixel_v2(loc0=[-160, -162, 61, 58], time_in_str='2016075', a0=1, resolution=36, npz_name=False):
    if npz_name:
        value_npz = np.load(npz_name)
    else:
        value_npz = np.load('onset_%s.npz' % time_in_str[0: 4])
    secs_lat60 = bxy.get_total_sec(time_in_str, fmt='%Y%j', reftime=[2000, 1, 1, 12])
    if resolution == 36:
        # grid 360 N
        h0 = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102')
        lons_grid = h0['cell_lon'].value
        lats_grid = h0['cell_lat'].value
        onset_s = value_npz['smap_onset']
        mask = np.load(('./result_05_01/other_product/mask_ease2_360N.npy'))
    else:
        # grid 125 N
        lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                                np.load('./result_05_01/other_product/lat_ease_grid.npy')
        onset_s = value_npz['ascat_onset']
        mask = np.load(('./result_05_01/other_product/mask_ease2_125N.npy'))
    onset_array = np.zeros(lons_grid.shape)
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]
    land_2d = np.unravel_index(land_id, mask.shape)
    onset_array[land_2d] = onset_s
    # get target
    loc_bool = (lons_grid < loc0[0]) & (lons_grid > loc0[1]) & (lats_grid < loc0[2]) & (lats_grid > loc0[3])
    v_bool = (onset_array*a0 < secs_lat60*a0) & (onset_array != 0)
    co_2d = np.where(loc_bool & v_bool)
    tar_latlon = np.array([lons_grid[co_2d], lats_grid[co_2d]]).T
    indices = co_2d[0]*lons_grid.shape[0]+co_2d[1]
    # get second indices
    p = 0
   # plot in map to check
    return tar_latlon, indices, onset_array[co_2d]


def check_map_pixel(loc0=[-160, -162, 61, 58], time_in_str='2016075', a0=1, resolution=36):
    value_npz = np.load('onset_%s.npz' % time_in_str[0: 4])
    secs_lat60 = bxy.get_total_sec(time_in_str, fmt='%Y%j', reftime=[2000, 1, 1, 12])
    if resolution == 36:
        # grid 360 N
        h0 = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102')
        lons_grid = h0['cell_lon'].value
        lats_grid = h0['cell_lat'].value
        onset_s = value_npz['smap_onset']
        mask = np.load(('./result_05_01/other_product/mask_ease2_360N.npy'))
    else:
        # grid 125 N
        lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                                np.load('./result_05_01/other_product/lat_ease_grid.npy')
        onset_s = value_npz['ascat_onset']
        mask = np.load(('./result_05_01/other_product/mask_ease2_125N.npy'))
    onset_array = np.zeros(lons_grid.shape)
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]
    land_2d = np.unravel_index(land_id, mask.shape)
    onset_array[land_2d] = onset_s
    # get target
    loc_bool = (lons_grid < loc0[0]) & (lons_grid > loc0[1]) & (lats_grid < loc0[2]) & (lats_grid > loc0[3])
    v_bool = (onset_array*a0 < secs_lat60*a0) & (onset_array != 0)
    co_2d = np.where(loc_bool & v_bool)
    tar_latlon = np.array([lons_grid[co_2d], lats_grid[co_2d]]).T
    co_1d = co_2d[0]*lons_grid.shape[0]+co_2d[1]
   # plot in map to check
    return tar_latlon


def check_outlier_pixel():
    p_name00 = 7000
    loc0 = np.array([-160, -162, 61, 58])
    p_latlon0 = check_map_pixel()

    loc0 = np.array([-156.25, -158, 64, 62.5])
    p_latlon1 = check_map_pixel(loc0, time_in_str='2016110', a0=-1)  # '2211', 63.63900, -158.03010


    loc1 = np.array([-154, -155, 66.8, 66.1])
    p_latlon2 = check_map_pixel(loc1, time_in_str='2016110', a0=-1)


    loc2 = np.array([-153, -155, 66.3, 65.3])
    p_latlon3 = check_map_pixel([-153, -155, 66.3, 65.3], time_in_str='2016075', a0=1)


    loc3 = np.array([-145, -150, 70, 68.6])
    p_latlon4 = check_map_pixel([-145, -150, 70, 68.6], time_in_str='2016108', a0=1)


    p_latlon = np.vstack((p_latlon0, p_latlon1, p_latlon2, p_latlon3, p_latlon4))
    p_num = p_latlon.shape[0]
    p_name_list = np.arange(1, p_num+1) + 7000

    return p_latlon, p_name_list.astype(str)


def check_outlier_pixel_v2(r=12.5):
    p_name00 = 5000
    loc0 = np.array([-160, -162, 61, 58])
    p_latlon0 = check_map_pixel()


def check_pixels_125N():
    # '968', 68.61683, -149.30017, p0 close to this
    # '1175', 67.93333, -162.28333, p1 close to this
    # '1266', 60.98, -153.92, p2 close to this
    # '957', 68.130, -149.478, p3 close to this
    pixels_latlon = np.array([[68.51683, -148.00017], [67.76333, -162.0], [60.98, -153.62],
                       [68.030, -150.278], [60.0, -160.0], [60.2, -160.0]])
    p_latlon = np.zeros(pixels_latlon.shape)
    p_latlon[:, 0] = pixels_latlon[:, 1]
    p_latlon[:, 1] = pixels_latlon[:, 0]
    p_name = np.arange(1, pixels_latlon.shape[0]+1) + 5000
    return p_latlon, p_name.astype(str)


def ms_get_interest_series_v2(pixel_type='interest', year0=2016,
                              input_pixels=[-1, -1, -1, -1], time_window=[0, 210],
                              input_ascat_index=[]):
    '''
    new version, no loops
    :param is_check:
    :param pixel_type: interest, all, outlier
    :param input_pixels: [0 pixel id, 1 [ 10 (lon, lat), 11 ind_smap, 12 ind_ascat]]
           i.e. input_pixels=[p02_ind_ascat.astype(str),
                            [p02_new, p02_ind_smap, p02_ind_ascat]]
    :return:
        ascat_dict_yearly: dictionary, keys include incidence angle (e.g., 'inc_angle_trip_aft'),
                           dimension: (land pixels x time series)
        npr_series:
    '''
    # get time series of specified pixels
    if pixel_type == 'interest':
        points = site_infos.get_id()
        p_latlon = np.zeros([len(points), 2])
        for i0, sno in enumerate(points):
            p0 = site_infos.change_site(sno)
            p_latlon[i0, 0], p_latlon[i0, 1] = p0[2], p0[1]
    elif pixel_type == 'outlier':
        ascat_index = input_ascat_index
        points, p_latlon, smap_index, ascat_index_na = \
            input_pixels[0], input_pixels[1], input_pixels[2], input_pixels[3]
    # smap_yearly = get_yearly_smap(t_window=[0, 300], year0=year0)
    smap_yearly = data_process.get_smap_dict(np.arange(0, 300), y=year0)
    start0 = bxy.get_time_now()
    if pixel_type == 'all':  # all alaska pixels
        # set index
        n125_n360 = np.load('n12_n36_array.npy').astype(int)
        ascat_all_id = np.unravel_index(n125_n360[0], (300, 300))
        smap_all_id = np.unique(n125_n360[1])
        # read data
        ascat_dict_yearly = ms_read_ascat(year0, t_window=time_window, pixel_id=ascat_all_id)
        npr_series, pid_smap = \
            prepare_smap_series(smap_all_id, input_ascat=ascat_dict_yearly, input_smap=smap_yearly)
    else:
        # set index
        # N125_2d, ascat_125n_1d = bxy.latlon2index(p_latlon)
        # smap_360n_index = index_ascat2smap(ascat_125n_1d)
        N125_2d = np.unravel_index(ascat_index, (300, 300))
        # n125_n360 = np.array([ascat_index, smap_index])
        # np.savetxt('npz_indices_%s.txt' % pixel_type,
        #            np.array([ascat_index, smap_index, np.array(points).astype(int)]), delimiter=',', fmt='%d')
        # read data based on index
        ascat_dict_yearly = ms_read_ascat(year0, t_window=time_window, pixel_id=N125_2d)
        # save the interested pixel measurments
        np.savez('npy_series_file/ascat_%s_series_%d.npz' % (pixel_type, year0), **ascat_dict_yearly)
        # ascat_dict_yearly = {'sigma0_trip_mid': np.array([0, 0]), 'inc_angle_trip_mid': np.array([0, 0]),
        #                      'utc_line_nodes': np.array([0, 0])}
        npr_series, pid_smap = prepare_smap_series(smap_index,
                                                   input_ascat=ascat_dict_yearly, input_smap=smap_yearly,
                                                   ascat_index=ascat_index)
        start1 = bxy.get_time_now()
        print("----read ascat part in %d: %s seconds ---" % (year0, start1-start0))
        # onsets, pixels = combine_detect_v2(n125_n360, year0, ascat_dict_yearly, npr_series, save_sp=True,
                                        # npz_name=pixel_type, pid_smap=pid_smap, gk=gk, npz_doc='npz_folder_085')
    return ascat_dict_yearly, npr_series, pid_smap

def ms_get_interest_series(pixel_type='interest', is_check=False,
                           ascat_index=np.array([]), smap_index=np.array([]), pixel_name=np.array([]),
                           time_window=[0, 210], gk=[7, 9, 12]):
    '''
    :param is_check:
    :param pixel_type: interest, all, outlier; gk: Gaussian kernels
    :param input_pixels: [0 pixel id, 1 ind_smap, 2 ind_ascat]]
           i.e. input_pixels=[p02_ind_ascat.astype(str),
                            [p02_new, p02_ind_smap, p02_ind_ascat]]
    :return:
    '''
    # get time series of specified pixels
    if pixel_type == 'interest':
        pixel_name = site_infos.get_id(int)
        p_latlon = np.zeros([len(pixel_name), 2])
        for i0, sno in enumerate(pixel_name):
            p0 = site_infos.change_site(sno)
            p_latlon[i0, 0], p_latlon[i0, 1] = p0[2], p0[1]
        save_index = pixel_name
        np.savetxt('ascat_index_%s_pixels.txt' % (pixel_type), save_index, fmt='%d', dilimiter=',')
    elif pixel_type == 'outlier':
        if smap_index.size == 0:
            # search the smap index from the indices table corresponding to the given ascat index
            index_table = np.load('n12_n36_array.npy')
            indices_1 = [np.where(index_table[0] == id0)[0][0] for id0 in ascat_index]
            smap_index = index_table[1][indices_1].astype(int)
        if pixel_name.size == 0:
            pixel_name = ascat_index
        save_index = ascat_index
        np.savetxt('ascat_index_%s_pixels.txt' % (pixel_type), save_index, fmt='%d', delimiter=',')
    # plot the interested pixel on map
    if is_check:
        a1 = -1
        print 'dont check in ms_get_interest_series'
        return a1
    for year0 in [2016, 2017, 2018]:
        smap_yearly = get_yearly_smap(t_window=[0, 210], year0=year0)
        start0 = bxy.get_time_now()
        if pixel_type == 'all':  # all alaska pixels
            # set index
            n125_n360 = np.load('n12_n36_array.npy').astype(int)
            ascat_all_id = np.unravel_index(n125_n360[0], (300, 300))
            smap_all_id = np.unique(n125_n360[1])
            # read data
            ascat_dict_yearly = ms_read_ascat(year0, t_window=time_window, pixel_id=ascat_all_id)
            npr_series, pid_smap = \
                prepare_smap_series(smap_all_id, input_ascat=ascat_dict_yearly, input_smap=smap_yearly)
        else:
            # set index
            # N125_2d, ascat_125n_1d = bxy.latlon2index(p_latlon)
            # smap_360n_index = index_ascat2smap(ascat_125n_1d)
            N125_2d = np.unravel_index(ascat_index, (300, 300))
            n125_n360 = np.array([ascat_index, smap_index])
            np.savetxt('npz_indices_%s.txt' % pixel_type,
                       np.array([ascat_index, smap_index, np.array(pixel_name).astype(int)]), delimiter=',', fmt='%d')
            # read data based on index
            ascat_dict_yearly = ms_read_ascat(year0, t_window=time_window, pixel_id=N125_2d)
            # ascat_dict_yearly = {'sigma0_trip_mid': np.array([0, 0]), 'inc_angle_trip_mid': np.array([0, 0]),
            #                      'utc_line_nodes': np.array([0, 0])}
            npr_series, pid_smap = prepare_smap_series(smap_index,
                                                       input_ascat=ascat_dict_yearly, input_smap=smap_yearly,
                                                       ascat_index=ascat_index)

        start1 = bxy.get_time_now()

        print("----read ascat part in %d: %s seconds ---" % (year0, start1-start0))
        onsets, pixels = combine_detect_v2(n125_n360, year0, ascat_dict_yearly, npr_series, all_region=True,
                                        npz_name=pixel_type, pid_smap=pid_smap, gk=gk, npz_doc='npz_folder_085_new')
        ascat_dict_yearly, npr_series = 0, 0


def index_ascat2smap(ascat_index):
    land_id = 0
    n125_n360_all = np.load('n12_n36_array.npy').astype(int)
    # if ascat_index.size > 1e3:
    #     print 'too much ascat pixels'
    # else:
    if ascat_index.size < 1e3:
        n125_n360_id = np.array([np.where(n125_n360_all[0] == ascat_index0)[0][0] for ascat_index0 in ascat_index])
        n125_n360 = n125_n360_all[:, n125_n360_id]
        land_id = n125_n360[1]
    # return specified smap id 1d, all smap id 1d, all ascat id 2d
    return land_id


def find_outlier():
    # 957 [68.130, -149.478] north east to [69.01, -145]
    # 1183 [69.42, -148.7] southeastern wards to [68.050, -145.0]
    p_latlon = np.zeros([12, 2])
    p_indices =p_latlon.copy()
    p_latlon[0:6, 1] = np.linspace(68.130, 69.01, num=6)
    p_latlon[6:, 1] = np.linspace(69.42, 68.05, num=6)
    p_latlon[0:6, 0] = np.linspace(-149.478, -145.01, num=6)
    p_latlon[6:, 0] = np.linspace(-148.7, -145.01, num=6)
    p_name = 5000 + np.arange(1, p_latlon.shape[0]+1)  # name of pixel
    # p_name.shape = 6, -1
    # index in 1d, rows, and cols
    N125_2d, ascat_125n_1d = bxy.latlon2index(p_latlon)
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    n125_land_index = n125_n360[0]
    target_index = [np.where(n125_land_index == ind0)[0][0] for ind0 in ascat_125n_1d]
    smap_360n_id = index_ascat2smap(ascat_125n_1d)
    p_indices[:, 0] = target_index
    p_indices[:, 1] = smap_360n_id
    p_onsets = np.zeros([p_latlon.shape[0], 3])
    i0 = 0
    for year0 in [2016, 2017, 2018]:
        value_npz = np.load('onset_%d.npz' % year0)
        onset = value_npz['ascat_onset']  # arr_1
        # ascat_winter = value_npz['sigma0_winter']
        # lvl = value_npz['ascat_lvl']
        onset_target = onset[target_index]
        p_onsets[:, i0] = onset_target
        i0 += 1
    save_array = np.zeros([p_latlon.shape[0], 1+ p_latlon.shape[1] + p_indices.shape[1] + p_onsets.shape[1]])
    save_array[:, 0], save_array[:, 1:3], save_array[:, 3:5], save_array[:, 5:] = p_name, p_latlon, p_indices, p_onsets
    np.savetxt('outlier_info.txt', save_array, delimiter=',', fmt='%.2f')
    return p_name, p_latlon, p_onsets, p_indices


def find_outlier2():
    loc0 = np.array([-145.0, -150.0, 69, 68.13])
    p_latlon1, indice_ascat, ons1 = check_map_pixel_v2(loc0, time_in_str='2016145', a0=-1, resolution=12)  # '2211', 63.63900, -158.03010

    loc0 = np.array([-146.5, -149, 69.3, 68.8])
    p_latlon2, indice_ascat2, ons2 = check_map_pixel_v2(loc0, time_in_str='2016144', a0=-1, resolution=12)

    loc0 = np.array([-144.0, -145.0, 70, 69.0])
    p_latlon3, indice_ascat3, ons3 = check_map_pixel_v2(loc0, time_in_str='2016135', a0=-1, resolution=12)

    loc0 = np.array([-152.50, -154.0, 62.0, 61.0])
    p_latlon4, indice_ascat4,  ons4 = check_map_pixel_v2(loc0, time_in_str='2016130', a0=-1, resolution=12)

    loc0 = np.array([-152.50, -153.2, 63.0, 62.0])  # -154 -- -153.2
    p_latlon5, indice_ascat5, ons5 = check_map_pixel_v2(loc0, time_in_str='2016105', a0=-1, resolution=12)

    p_latlon1 = np.vstack((p_latlon1, p_latlon2, p_latlon3, p_latlon4[[0, 5]], p_latlon5[[0, 5]]))
    indice_ascat = np.concatenate((indice_ascat, indice_ascat2, indice_ascat3,
                                   indice_ascat4[[0, 5]], indice_ascat5[[0, 5]]))
    ons = np.concatenate((ons1, ons2, ons3, ons4[[0, 5]], ons5[[0,5]]))
    # save_array = np.zeros([p_latlon.shape[0], 1+ p_latlon.shape[1] + 2])
    #
    # save_array[:, 0], save_array[:, 1:3], save_array[:, 3:5], save_array[:, 5:] = p_name, p_latlon, p_indices, p_onsets
    # np.savetxt('outlier_info.txt', save_array, delimiter=',', fmt='%.2f')
    return p_latlon1, indice_ascat, ons


def find_outlier3():
    loc0 = np.array([-150.0, -155.0, 68.13, 66.0])
    p_latlon1, indice_ascat, ons1 = check_map_pixel_v2(loc0, npz_name='onset_all_2018_18220.npz',
                                                       time_in_str='2018150', a0=-1, resolution=12)  # '2211', 63.63900, -158.03010

    loc0 = np.array([-144.5, -145.5, 68.0, 66.13])
    p_latlon2, indice_ascat2, ons2 = check_map_pixel_v2(loc0, npz_name='onset_all_2016_18229.npz',
                                                        time_in_str='2016122', a0=-1, resolution=12)

    loc0 = np.array([-150.0, -153.0, 60.13, 66.0])
    p_latlon3, indice_ascat3, ons3 = check_map_pixel_v2(loc0, npz_name='onset_all_2018_18220.npz',
                                                        time_in_str='2018145', a0=-1, resolution=12)
    #
    # loc0 = np.array([-152.50, -154.0, 62.0, 61.0])
    # p_latlon4, indice_ascat4,  ons4 = check_map_pixel_v2(loc0, time_in_str='2016130', a0=-1, resolution=12)
    #
    # loc0 = np.array([-152.50, -153.2, 63.0, 62.0])  # -154 -- -153.2
    # p_latlon5, indice_ascat5, ons5 = check_map_pixel_v2(loc0, time_in_str='2016105', a0=-1, resolution=12)

    p_latlon1 = np.vstack((p_latlon1, p_latlon2, p_latlon3))
    indice_ascat = np.concatenate((indice_ascat, indice_ascat2, indice_ascat3))
    ons = np.concatenate((ons1, ons2, ons3))
    # save_array = np.zeros([p_latlon.shape[0], 1+ p_latlon.shape[1] + 2])
    #
    # save_array[:, 0], save_array[:, 1:3], save_array[:, 3:5], save_array[:, 5:] = p_name, p_latlon, p_indices, p_onsets
    # np.savetxt('outlier_info.txt', save_array, delimiter=',', fmt='%.2f')
    return p_latlon1, indice_ascat, ons


def ms_ca_station(z0, z1, z2, key_name, id0, id1):
    return np.hstack((z0[key_name][id0][id1], z1[key_name][id0][id1], z2[key_name][id0][id1]))


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


def save_new_125_360_array():
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    out_array = np.zeros([4, n125_n360.shape[1]])
    lons_2d, lats_2d, _ = ind2latlon([0, np.array([29908])])
    out_array[0], out_array[1] = n125_n360[0], n125_n360[1]
    out_array[2], out_array[3] = lons_2d.ravel()[n125_n360[0]], lats_2d.ravel()[n125_n360[0]]
    np.save('n12_n36_array_4cols.npy', out_array)


def hystory(x=2):
    if x == 0:
        a = 0
        data_process.re_detection_plot()
    elif x < 0:
        print 'test function, no processing'
    elif x==2:
        n125_n360 = np.load('n12_n36_array_4cols.npy')
        type = 'all'
        for year0 in [2016, 2018]:
            start0 = bxy.get_time_now()
            ascat_dict_yearly, npr_series, pid_smap = ms_get_interest_series_v2(pixel_type=type, year0=year0,
                                                                                time_window=[0, 210])
            # save the required series
            np.save('npy_series_file/npr_series_%d.npy', npr_series)
            print 'prepare two series data for year %d takes %s seconds' % (year0, bxy.get_time_now()-start0)
            start0 = bxy.get_time_now()
            onsets, pixels = combine_detect_v2(n125_n360, year0, ascat_dict_yearly, npr_series, all_region=True,
                                               npz_name=type, pid_smap=pid_smap, gk=[5, 7, 7],
                                               npz_doc='npz_folder_final_hopefully')
            ascat_npy2npz(year0=year0)
            print 'combine detect v2 for year %d takes %s seconds' % (year0, bxy.get_time_now()-start0)
    elif x==1:  # save data of interested
        ascat_outlier_index = np.loadtxt('map_plot_check_pixel_2018.txt', delimiter=',')
        ascat_1d_id = ascat_outlier_index[0:2, 0].astype(int)
        smap_1d_id = ascat_outlier_index[0:2, -1].astype(int)
        for year0 in [2016, 2017, 2018]:
            ## save smap do not delete
            # smap_yearly = data_process.get_smap_dict(np.arange(0, 300), y=year0)
            # np.savez('smap_all_series_D_%d.npz' % (year0), **smap_yearly[1])
            # save ascat
            ascat_row_col = np.unravel_index(ascat_1d_id, (300, 300))
            ascat_dict_yearly = ms_read_ascat(year0, t_window=[0, 300], pixel_id=ascat_row_col)  # reading
            ascat_dict_yearly['pixel_id'] = np.array([ascat_1d_id, smap_1d_id])
            np.savez('prepare_files/npz/ascat/ascat_interest_pixel_series_%d.npz' % year0, **ascat_dict_yearly)
    elif x==3:

        site_array = np.loadtxt('npz_indices_interest.txt', delimiter=',').astype(int)  # 0: a, 1: s, 2: site no
        ascat_outlier_index = np.loadtxt('map_plot_check_pixel_2018.txt', delimiter=',')
        ascat_1d_id = site_array[0]
        smap_1d_id = site_array[1]
        # list_str = site_array[2].astype(str)
        # lat_list = [site_infos.change_site(list0)[1] for list0 in list_str]
        # lon_list = [site_infos.change_site(list0)[2] for list0 in list_str]
        # save_all_station = np.zeros([5, ascat_1d_id.size])
        # head = 'ascat_1d_id, smap_1d_id, site_number, site_lat, site_lon'
        # save_all_station[0:3] = site_array
        # save_all_station[3] = lat_list
        # save_all_station[4] = lon_list
        # np.savetxt('indice_interest_all_stations.txt', save_all_station.T, delimiter=',', fmt='%.3f', header=head)
        for year0 in [2016, 2017, 2018]:
            ## save smap do not delete
            # smap_yearly = data_process.get_smap_dict(np.arange(0, 300), y=year0)
            # np.savez('smap_all_series_D_%d.npz' % (year0), **smap_yearly[1])
            # save ascat
            ascat_row_col = np.unravel_index(ascat_1d_id, (300, 300))
            ascat_dict_yearly = ms_read_ascat(year0, t_window=[0, 300], pixel_id=ascat_row_col)  # reading
            ascat_dict_yearly['pixel_id'] = np.array([ascat_1d_id, smap_1d_id])
            np.savez('prepare_files/npz/ascat/ascat_interest_pixel_series_%d.npz' % year0, **ascat_dict_yearly)

    quit0()


def hystory_plot_map(year0=2016, p_type='interest', site_no=[]):
    p_array=np.loadtxt('npz_indices_interest.txt', delimiter=',').astype(int)
    if len(site_no) > 0:
        ind_site_in_p_array = [np.where(p_array[2] == int(s0))[0][0] for s0 in site_no]
        p_array = p_array[:, ind_site_in_p_array]
    files_year0 = glob.glob('onset_all_%d_22*.npz' % year0)
    a0 = [map_plot1(year0, onset_name=f0, p_name=p_array[2], p_index=[p_array[1], p_array[0]])
              for f0 in files_year0]
    return a0


def check_no_lvl_region():
    ind_2d, ind_1d = bxy.latlon2index(np.array([[-157.3, 62.5], [-158.3, 63.5], [-160.1, 60.1], [-160.0, 61.3],
                                                [-161.1, 60.3]]))
    print '1D INDEX OF TARGET', ind_1d
    # p_temp = np.array([46721, 47320, 47019, 49120, 47619, 48518, 49716, 50912, 50012])
    p02_ind_ascat = ind_1d
    lons, lats, _ = ind2latlon([0, p02_ind_ascat])
    p02_new = np.array([lons.ravel()[p02_ind_ascat], lats.ravel()[p02_ind_ascat]]).T
    p_table = np.load('n12_n36_array.npy')
    p02_ind_smap = np.array([p_table[1][p_table[0] == p02_0] for p02_0 in p02_ind_ascat]).ravel().astype(int)
    ms_get_interest_series(False, pixel_type='outlier', input_pixels=[p02_ind_ascat.astype(str),
                                                                     [p02_new, p02_ind_smap, p02_ind_ascat]])
    ms_station_new(p02_ind_ascat.astype(str), 'outlier', detectors=True)
    quit0()


def check_final_outlier():
    p02, p02_ind_ascat, ons_ascat = find_outlier3()
    n2d, n1d = bxy.latlon2index(np.array([[-150.0, 70]]))
    p_temp = np.array([50916, 30507, 50918, 45527, 45526, 44931, 51516, 20911, 20909, 32568])
    p02_ind_ascat = np.concatenate((p02_ind_ascat, p_temp, n1d))
    lons, lats, _ = ind2latlon([0, p02_ind_ascat])
    p02_new = np.array([lons.ravel()[p02_ind_ascat], lats.ravel()[p02_ind_ascat]]).T
    p_table = np.load('n12_n36_array.npy')
    p02_ind_smap = np.array([p_table[1][p_table[0] == p02_0][0] for p02_0 in p02_ind_ascat]).astype(int)

    # a0 = map_plot1(2016, onset_name='onset_all_2016_18229.npz', p_name=p02_ind_ascat,
    #                p_index=[p02_ind_smap, p02_ind_ascat])
    # a1 = map_plot1(2018, onset_name='onset_all_2018_18220.npz', p_name=p02_ind_ascat,
    #                p_index=[p02_ind_smap, p02_ind_ascat])
    ms_get_interest_series(False, pixel_type='outlier', input_pixels=[p02_ind_ascat.astype(str),
                                                                     [p02_new, p02_ind_smap, p02_ind_ascat]])


def quick_location(loc=np.array([[-150.0, 70], [-156.0, 58.0]]), map_name='onset_all_2018_18220.npz',
                   index_1d_input=np.array([0]), pixel_name=np.array([0]), plot=True, year0=2016):
    '''
    enter the lon/lat or 1d index in grid 300 x 300, plot the map together with interested pixels
    :param loc: 2d array, pixels X lon/lat
    :return:
    '''
    index_2d_125, index_1d_125 = bxy.latlon2index(loc)
    # lons, lats, _ = ind2latlon([0, n1d])
    # p02_new = np.array([lons.ravel()[n1d], lats.ravel()[n1d]]).T  # how to precisely get the lat/lon from 1d ind
    p_table = np.load('n12_n36_array.npy')
    if index_1d_input[0] > 0:
        pixel_name = np.concatenate((index_1d_125, pixel_name))
        index_1d_125 = np.concatenate((index_1d_125, index_1d_input))
    index_1d_360 = np.array([p_table[1][p_table[0] == p02_0] for p02_0 in index_1d_125])
    # check if any station pixels are included
    index_table_station = np.loadtxt('npz_indices_interest.txt', delimiter=',').astype(int)
    if plot:
        year = year0
        a1 = map_plot1(year, onset_name=map_name, p_name=pixel_name,
                       p_index=[index_1d_360, index_1d_125], mode='quick')
    return index_1d_360, index_1d_125


def script_20190627_outliers(npz_name='onset_all_2016_302222.npz', pixel_type='interest',
                             pixel_index=np.array([])):
    p_array = np.loadtxt('npz_indices_%s.txt' % pixel_type, delimiter=',').astype(int)
    if pixel_index.size > 0:
        i_col = bxy.index_match(p_array[0], pixel_index)
        p_array = p_array[:, i_col]
        p_array.shape = p_array.shape[0], p_array.shape[1]
    lon_lat = np.array([[-162.0, 68.5], [-160.1, 67.2], [-157.5, 67.5], [-160, 70], [-160, 69.2],
                        [-153.1, 67.4], [-150.6, 65.8], [-144.8, 66], [-143, 68.2], [-162.7, 65]])
    interested_pixel = np.array([27504, 28102, 29908, 30507, 49716, 50912, 50612, 50012, 49718, 47320]).astype(int)
    ind_smap, ind_ascat = quick_location(lon_lat,
                                         index_1d_input=p_array[0], pixel_name=p_array[2],
                                         map_name=npz_name, plot=True)
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    map_plot1_check_pixel(npz_name, p_array[0],
                      ['sigma0_winter', 'sigma0_winter_std',
                       'sigma0_melt_mean', 'sigma0_melt_std',
                       'sigma0_summer',
                       'sigma0_min', 'ascat_lvl'],
                      n125_n360[0])
    # quit0()
    # lons, lats, _ = ind2latlon(ind_ascat)
    # latlon_ascat = np.array([lons.ravel()[ind_ascat], lats.ravel()[ind_ascat]])
    # quit0()
    # ms_get_interest_series(False, pixel_type='outlier',
    #                        input_pixels=[ind_smap.ravel().astype(int).astype(str),
    #                                     [latlon_ascat, ind_smap.ravel().astype(int), ind_ascat]],
    #                        gk=[7, 15, 10])
    # ms_station_new(detectors=[7, 15, 10])
    # ms_station_new(sno_all=['27504', '30507', '35596'], pixel_type='outlier', detectors=[7, 15, 10])
    # print ind_smap, ind_ascat


def script_outlier_20190705(m_name='onset_all_2016_1229.npz', year0=2016, interested_pixel=np.array([])):
    if interested_pixel.size < 1:
        pixel_array = np.loadtxt('npz_indices_outlier.txt', delimiter=',')
        interested_pixel = pixel_array[2].astype(int)
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    # array_out = map_plot1_check_pixel(m_name, interested_pixel,
    #                   ['sigma0_winter', 'sigma0_winter_std',
    #                    'sigma0_melt_mean', 'sigma0_melt_std',
    #                    'sigma0_summer',
    #                    'sigma0_min', 'ascat_lvl'],
    #                   n125_n360, year0=year0)
    ind_smap, ind_ascat = quick_location(np.array([[-160, 60]]),
                                         index_1d_input=interested_pixel, pixel_name=interested_pixel,
                                         map_name=m_name, plot=True, year0=year0)


def outlier_maps(test_no=3):
    # onset_all_2016_132323.npz, set 09.
    if test_no == 0:
        script_outlier_20190705('npz_folder_085/onset_all_2016_15938.npz', year0=2016)
        # npz_folder_085/onset_all_2017_15958.npz, npz_folder_085/onset_all_2018_151014.npz
        script_outlier_20190705('npz_folder_085/onset_all_2017_15958.npz', year0=2017)
        script_outlier_20190705('npz_folder_085/onset_all_2018_151014.npz', year0=2018)
    elif test_no == 1:
        script_outlier_20190705('npz_folder_085_new2/onset_all_2016_191739.npz', year0=2016)
        # npz_folder_085/onset_all_2017_15958.npz, npz_folder_085/onset_all_2018_151014.npz
        script_outlier_20190705('npz_folder_085_new2/onset_all_2017_191758.npz', year0=2017)
        script_outlier_20190705('npz_folder_085_new2/onset_all_2018_191813.npz', year0=2018)
    elif test_no == 2:
        # onset_all_2016_192051.npz  onset_all_2017_192110.npz  onset_all_2018_192127.npz
        script_outlier_20190705('npz_folder_085_new2/onset_all_2016_192051.npz', year0=2016)
        script_outlier_20190705('npz_folder_085_new2/onset_all_2017_192110.npz', year0=2017)
        script_outlier_20190705('npz_folder_085_new2/onset_all_2018_192127.npz', year0=2018)
    elif test_no == 3:
        # onset_all_2016_192051.npz  onset_all_2017_192110.npz  onset_all_2018_192127.npz
        # script_outlier_20190705('npz_folder_085_new/onset_all_2016_211431.npz', year0=2016)
        # script_outlier_20190705('npz_folder_085_new/onset_all_2017_211450.npz', year0=2017)
        script_outlier_20190705('npz_folder_085_new/onset_all_2018_21156.npz', year0=2018)
    print 'outlier maps are plotted'
    quit0()


def alaska_map1():
    # npz_folder_085_new/onset_all_2016_21212.npz
    # npz_folder_085_new/onset_all_2017_212120.npz
    # npz_folder_085_new/onset_all_2018_212136.npz
    script_outlier_20190705('npz_folder_085_new//onset_all_2016_21212.npz', year0=2016)
    script_outlier_20190705('npz_folder_085_new/onset_all_2017_212120.npz', year0=2017)
    script_outlier_20190705('npz_folder_085_new/onset_all_2018_21156.npz', year0=2018)


def alaska_map2():
    # onset_all_2016_221142.npz  onset_all_2017_22121.npz  onset_all_2018_221217.npz
    script_outlier_20190705('npz_folder_085_new/onset_all_2016_221142.npz', year0=2016)
    script_outlier_20190705('npz_folder_085_new/onset_all_2017_22121.npz', year0=2017)
    script_outlier_20190705('npz_folder_085_new/onset_all_2018_221217.npz', year0=2018)


def alaska_map3():
    # 07 24
    # npz_folder_085/onset_all_2016_25015.npz  npz_folder_085/onset_all_2018_25052.npz
    # npz_folder_085/onset_all_2017_25035.npz
    # 07 25
    # onset_all_2016_251131.npz  onset_all_2017_251151.npz  onset_all_2018_25127.npz
    # script_outlier_20190705('npz_folder_725/onset_all_2016_251131.npz', year0=2016)
    # script_outlier_20190705('npz_folder_725/onset_all_2017_251151.npz', year0=2017)
    # script_outlier_20190705('npz_folder_725/onset_all_2018_25127.npz', year0=2018)
    # # 08 23
    script_outlier_20190705('npz_folder_0813/ascat_onset_all_2017_08221916.npz', year0=2017,
                            interested_pixel=np.array([44897]))


def switch_plot_alaska(num):
    options = {0: alaska_map1,
               1: alaska_map1,
               2: alaska_map2,
               3: alaska_map3
    }
    options[num]()
    quit0()


def ascat_npy2npz(npz_doc='npz_folder_0813', npz_type='all', year0=2018):
    array0 = np.load('npy_file/file2316.npy')
    f_list = glob.glob('npy_file/file*.npy')
    ascat_array_unsorted = np.zeros([len(f_list), array0.size])
    for i0, f0 in enumerate(f_list):
        # print f0
        ascat_array_unsorted[i0] = np.load(f0)
    # sort the ascat_array
    ascat_array = ascat_array_unsorted[ascat_array_unsorted[:, 0].argsort()].T
    time_prefix = bxy.get_time_now()
    time_array = np.array([time_prefix.month, time_prefix.day, time_prefix.hour, time_prefix.minute])
    time_str_array = [time_array[l0].astype(str) for l0 in range(time_array.size)]
    time_str_array_formated = ['0'+item if len(item) < 2 else item for item in time_str_array]
    npz_name = '%s/ascat_onset_%s_%d_%s%s%s%s.npz' % (npz_doc, npz_type, year0, time_str_array_formated[0],
                                                time_str_array_formated[1], time_str_array_formated[2],
                                                time_str_array_formated[3])
    np.savez(npz_name,
         **{
            'ascat_onset': ascat_array[11],
            'ascat_end': ascat_array[14], 'conv_on_melt_date': ascat_array[12], 'lvl_on_melt_date': ascat_array[13],
            'sigma0_mean_winter': ascat_array[5], 'sigma0_mean_summer': ascat_array[6],
            'sigma0_mean_melt_zone': ascat_array[7],
            'sigma0_std_winter': ascat_array[8], 'sigma_std_summer': ascat_array[9],
            'simga0_std_melt_zone': ascat_array[10],
            'winter_edge': ascat_array[20], 'time_zero_conv': ascat_array[15], 'sigma0_on_melt_date': ascat_array[16],
            'sigma0_min_melt_zone': ascat_array[17],
            'coef_a': ascat_array[1], 'coef_b': ascat_array[2],
            'winter_conv_mean': ascat_array[20], 'winter_conv_min': ascat_array[21], 'winter_conv_std': ascat_array[22],
            'sigma0_5dmean_after_onset': ascat_array[18],
            'sigma0_kernels': ascat_array[3: 5, :], 'melt_events_time': ascat_array[25: 33, :],
            'melt_events_conv': ascat_array[33: 41, :]
            })
    print 'saved as %s' % npz_name
    return 0


def file_name_formated(npz_doc='npz_folder_0813', npz_type='all', year0=2018):
    time_prefix = bxy.get_time_now()
    time_array = np.array([time_prefix.month, time_prefix.day, time_prefix.hour, time_prefix.minute])
    time_str_array = [time_array[l0].astype(str) for l0 in range(time_array.size)]
    time_str_array_formated = ['0'+item if len(item) < 2 else item for item in time_str_array]
    return '%s/smap_onset_%s_%d_%s%s%s%s.npz' % (npz_doc, npz_type, year0, time_str_array_formated[0],
                                                time_str_array_formated[1], time_str_array_formated[2],
                                                time_str_array_formated[3])


def kernel_series_edges():
    # data input
    std_0 = 7
    dict_year = {}
    for year0 in [2018]:  # add 2016, 2018
    #  N pixles, the corresponded ascat_id
        sigma0_on_station = np.load('prepare_files/npz/ascat/ascat_interest_pixel_series_%d.npz' % (year0))
        id_array = sigma0_on_station['pixel_id']
        smap_all_pixels = np.load('prepare_files/npz/smap/smap_all_series_A_%d.npz' % (year0))
        # smap measurements of the interested pixels id = id_array[1]
        tbv_on_station, tbh_on_station, secs_smap_all_station = smap_all_pixels['cell_tb_v_aft'][id_array[1]], \
                                                          smap_all_pixels['cell_tb_h_aft'][id_array[1]], \
                                                          smap_all_pixels['cell_tb_time_seconds_aft.npy'][id_array[1]]
        series_npr_all_staions = (tbv_on_station-tbh_on_station)/(tbv_on_station+tbh_on_station)
        series_npr_all_staions[tbv_on_station < 100] = -999
        dict_year[str(year0)] = [sigma0_on_station, series_npr_all_staions, secs_smap_all_station]
    # loop by pixel
    for i_pixel, smap_id0 in enumerate(id_array[1]):
        if i_pixel > 1:
            continue
        ascat_note = id_array[0][i_pixel]
        plot_dict = dict()  # a dictionary save the
        plot_dict['npr'], plot_dict['npr_conv'], plot_dict['ascat'] = [], [], []
        plot_dict['v_line'] = []
        # plot_dict['ascat_conv_melt'], plot_dict['ascat_conv_thaw'] = [], []
        plot_dict['npr_conv_bar'], plot_dict['ascat_conv_thaw_bar'], plot_dict['ascat_conv_melt_bar'] = [], [], []
        # dictionary save the onset
        plot_dict['npr_onset'], plot_dict['ascat_onset'] = [], []
        plot_dict['onset0'], plot_dict['onset1'], plot_dict['onset2'] = [], [], []
        for year0 in [2018]:  # 2016, 2017, 2018
            # for each pixel, estimate the onset year by year
            sigma0_one_year = dict_year[str(year0)][0]
            npr_one_year = dict_year[str(year0)][1]
            npr_secs_one_year = dict_year[str(year0)][2]
            # saving npr and back scatter
            npr_array = np.array([npr_secs_one_year[i_pixel], npr_one_year[i_pixel]])  # save npr
            valid_index = bxy.get_valid_index(npr_array, key_id=[0, 1], invalid=[-999])
            plot_dict['npr'].append(npr_array[:, valid_index])
            ascat_array = np.array([sigma0_one_year['utc_line_nodes'][i_pixel],  # save ascat back scatter
                                                sigma0_one_year['sigma0_trip_aft'][i_pixel]])
            valid_index = bxy.get_valid_index(ascat_array, key_id=[0, 1], invalid=[0])
            plot_dict['ascat'].append(ascat_array[:, valid_index])
            # detection
            m_zone, su_zone, th_zone, win_zone = data_process.zone_intiation(year0)
            # npr
            conv_npr_pixel, thaw_secs_npr, all_local_max, all_local_min, _\
                = data_process.smap_melt_initiation(npr_one_year[i_pixel], npr_secs_one_year[i_pixel],
                                                           win_zone, su_zone, year0, gk=std_0, one_pixel_return=True)
            plot_dict['npr_onset'].append(thaw_secs_npr)
            plot_dict['npr_conv_bar'].append(all_local_max[:, [1, 2]].T)
            convolution_series, convolution_event = data_process.two_series_sigma_process\
                (0, sigma0_one_year['sigma0_trip_aft'][i_pixel], sigma0_one_year['inc_angle_trip_aft'][i_pixel],
                 sigma0_one_year['utc_line_nodes'][i_pixel],
                 thaw_secs_npr, m_zone, th_zone, win_zone, su_zone,
                 7, [7, 7, 7], False,  save_path='prepare_files/npy_ascat_one_station', is_return=True)
            plot_dict['ascat_conv_thaw_bar'].append(convolution_event[0][:, [1, 2]].T)
            plot_dict['ascat_conv_melt_bar'].append(convolution_event[1][:, [1, 2]].T)
            combine_result = np.load('prepare_files/npy_ascat_one_station/file0.npy')
            plot_dict['onset0'].append(thaw_secs_npr), plot_dict['onset1'].append(combine_result[11])
            plot_dict['onset2'].append(combine_result[14])
        # npr: npr series, npr kernel, npr edges (convolution bars)
            k_npr = bxy.make_kernal(std_0)
            np.savez('npr_KSE.npz', series=plot_dict['npr'], kernel=k_npr, kernel2=0, edge=plot_dict['npr_conv_bar'])
            # k_ascat = bxy.make_kernal(combine_result[3], combine_result[4])
            k_ascat = bxy.make_kernal(7, 3)
            np.savez('ascat_KSE.npz', series=plot_dict['ascat'], kernel=k_ascat, edge=plot_dict['ascat_conv_melt_bar'])


def plot_KSE(npz_name, fig_name, mode='npr'):
    npr_kse = np.load(npz_name)
    plot_npr = npr_kse['series']
    if mode == 'npr':
        series = np.array([bxy.time_getlocaltime(plot_npr[0][0])[3], plot_npr[0][1]*100])
        kernel = np.array([npr_kse['kernel'][0]+40+10, npr_kse['kernel'][1]*10])
        edge_npr = npr_kse['edge']
        edges = np.array([bxy.time_getlocaltime(edge_npr[0][0])[3], edge_npr[0][1]*100])
    elif mode == 'sigma0':
        series = np.array([bxy.time_getlocaltime(plot_npr[0][0])[3], plot_npr[0][1]])
        kernel = np.array([npr_kse['kernel'][0]+40+10, npr_kse['kernel'][1]*10])
        edge_npr = npr_kse['edge']
        edges = np.array([bxy.time_getlocaltime(edge_npr[0][0])[3], edge_npr[0][1]])
        edges = edges[:, edges[1]<0]
    series50 = series[:, (series[0] > 30) & (series[0] < 180)]
    plot_funcs.plot_convolution(series50, kernel, edges, figname=fig_name, mode=mode)


def plot_KSE_melt():
    return 0


def quick_process(station=False, single_check=False):
    '''

    :param station:
    :param single_check: quickly run this routine to plot for 1 single pixel
    :return:
    '''
    # hystory(x=2)  # get the npz file of regional onset
    # hystory(x=1)  # save npz file of interested pixel
    #  prepare data 20190819
    # data_prepare_ascat(p0=['20170214', '20170215'], un_grid=False)
    # id_array = np.loadtxt('npz_indices_interest.txt', delimiter=',')
    n125_n360 = np.load('n12_n36_array_4cols.npy').astype(int)
    dict_year = dict()
    year0 = 2017
    id_pixel = 0
    for year0 in [2016, 2017, 2018]:  # add 2016, 2018
    #  N pixles, the corresponded ascat_id
        sigma0_on_station = np.load('prepare_files/npz/ascat/ascat_interest_pixel_series_%d.npz' % (year0))
        id_array = sigma0_on_station['pixel_id']
        smap_all_pixels = np.load('prepare_files/npz/smap/smap_all_series_A_%d.npz' % (year0))
        # smap measurements of the interested pixels id = id_array[1]
        tbv_on_station, tbh_on_station, secs_smap_all_station = smap_all_pixels['cell_tb_v_aft'][id_array[1]], \
                                                          smap_all_pixels['cell_tb_h_aft'][id_array[1]], \
                                                          smap_all_pixels['cell_tb_time_seconds_aft.npy'][id_array[1]]
        series_npr_all_staions = (tbv_on_station-tbh_on_station)/(tbv_on_station+tbh_on_station)
        series_npr_all_staions[tbv_on_station < 100] = -999
        dict_year[str(year0)] = [sigma0_on_station, series_npr_all_staions, secs_smap_all_station]

    for i_pixel, smap_id0 in enumerate(id_array[1]):
        if single_check:
            if i_pixel > 1:
                print 'check the pixel smap id %d' % smap_id0
                return 0
        ascat_id_1d = id_array[0][i_pixel]
        plot_dict = dict()  # a dictionary save the
        plot_dict['npr'], plot_dict['npr_conv'], plot_dict['ascat'] = [], [], []
        plot_dict['v_line'] = []
        # plot_dict['ascat_conv_melt'], plot_dict['ascat_conv_thaw'] = [], []
        plot_dict['npr_conv_bar'], plot_dict['ascat_conv_thaw_bar'], plot_dict['ascat_conv_melt_bar'] = [], [], []
        # dictionary save the onset
        plot_dict['npr_onset'], plot_dict['ascat_onset'] = [], []
        plot_dict['onset0'], plot_dict['onset1'], plot_dict['onset2'] = [], [], []
        for year0 in [2016, 2017, 2018]:  # 2016, 2017, 2018
            # for each pixel, estimate the onset year by year
            sigma0_one_year = dict_year[str(year0)][0]
            npr_one_year = dict_year[str(year0)][1]
            npr_secs_one_year = dict_year[str(year0)][2]
            # saving npr and back scatter
            npr_array = np.array([npr_secs_one_year[i_pixel], npr_one_year[i_pixel]])  # npr t series
            valid_index = bxy.get_valid_index(npr_array, key_id=[0, 1], invalid=[-999])
            plot_dict['npr'].append(npr_array[:, valid_index])
            ascat_array = np.array([sigma0_one_year['utc_line_nodes'][i_pixel],  # ascat back scatter t seies
                                    sigma0_one_year['sigma0_trip_aft'][i_pixel],
                                    sigma0_one_year['inc_angle_trip_aft'][i_pixel]])

            sigma0_45 = data_process.angular_correct(ascat_array[1], ascat_array[2], ascat_array[0])
            ascat_array[1] = sigma0_45
            # plot check
            index0 = ascat_array[0]>0
            plot_funcs.quick_plot(ascat_array[0][index0], sigma0_45[index0])
            valid_index = bxy.get_valid_index(ascat_array, key_id=[0, 1], invalid=[0])
            plot_dict['ascat'].append(ascat_array[:, valid_index])
            # detection
            m_zone, su_zone, th_zone, win_zone = data_process.zone_intiation(year0)
            conv_npr_pixel, thaw_secs_npr, all_local_max, all_local_min,\
            _ = data_process.smap_melt_initiation(npr_one_year[i_pixel], npr_secs_one_year[i_pixel],
                                                           win_zone, su_zone, year0, gk=7, one_pixel_return=True)
            plot_dict['npr_onset'].append(thaw_secs_npr)
            plot_dict['npr_conv_bar'].append(all_local_max[:, [1, 2]].T)
            convolution_series, convolution_event = data_process.two_series_sigma_process\
                (0, sigma0_one_year['sigma0_trip_aft'][i_pixel], sigma0_one_year['inc_angle_trip_aft'][i_pixel],
                 sigma0_one_year['utc_line_nodes'][i_pixel],
                 thaw_secs_npr, m_zone, th_zone, win_zone, su_zone,
                 7, [7, 7, 7], False,  save_path='prepare_files/npy_ascat_one_station', is_return=True)
            plot_dict['ascat_conv_thaw_bar'].append(convolution_event[0][:, [1, 2]].T)
            plot_dict['ascat_conv_melt_bar'].append(convolution_event[1][:, [1, 2]].T)
            combine_result = np.load('prepare_files/npy_ascat_one_station/file0.npy')
            plot_dict['onset0'].append(thaw_secs_npr), plot_dict['onset1'].append(combine_result[11])
            plot_dict['onset2'].append(combine_result[14])
        # plotting
        # prepare the main axis and second axis
        second_axis = [
                        np.hstack((plot_dict['npr_conv_bar'][0],
                                   plot_dict['npr_conv_bar'][1],
                                   plot_dict['npr_conv_bar'][2])),
                        np.hstack((plot_dict['ascat_conv_thaw_bar'][0],
                                   plot_dict['ascat_conv_thaw_bar'][1],
                                   plot_dict['ascat_conv_melt_bar'][2])),
                        np.hstack((plot_dict['ascat_conv_melt_bar'][0],
                                   plot_dict['ascat_conv_melt_bar'][1],
                                   plot_dict['ascat_conv_melt_bar'][2])),
                      ]
        npr_plot = np.hstack((plot_dict['npr'][0], plot_dict['npr'][1], plot_dict['npr'][2]))
        ascat_plot = np.hstack((plot_dict['ascat'][0], plot_dict['ascat'][1], plot_dict['ascat'][2]))
        vline_secs = plot_dict['onset0']+plot_dict['onset1']+plot_dict['onset2']
        site_id_array = np.loadtxt('indice_interest_all_stations.txt', delimiter=',')
        site_id_ascat = site_id_array[:, 0].astype(int)
        site_id_no = site_id_array[:, 2].astype(int)
        if ascat_id_1d in site_id_ascat:
            print "the site is", site_id_no[site_id_ascat==ascat_id_1d]
            figname = 'ms_pixel_test_%d_%d' % (id_array[0][i_pixel], site_id_no[site_id_ascat==ascat_id_1d][0])
        else:
            figname='ms_pixel_test_%d_%d' % (id_array[0][i_pixel], smap_id0)
        # # 3 subs
        # plot_funcs.plot_subplot([npr_plot, ascat_plot, ascat_plot],
        #                         second_axis,
        #                         main_label=['npr', '$\sigma^0$ mid', '$\sigma^0$'],
        #                         figname=figname, x_unit='doy',
        #                         vline=[vline_secs,
        #                                ['k-', 'k-', 'k-', 'r-', 'r-', 'r-', 'b-', 'b-', 'b-'],
        #                                ['p', 'p', 'p', 'n', 'n', 'n', 'p1', 'p1', 'p1']],
        #                         vline_label=bxy.time_getlocaltime(vline_secs, ref_time=[2000, 1, 1, 0])[-2],
        #                         h_line2=[[0, 1, 2], [0.01, 1, -1], [':', ':', ':']],
        #                         # annotation_sigma0=[text_qa_w, text_qa_m],
        #                         # x_lim=[bxy.get_total_sec('20160101'), bxy.get_total_sec('20181230')],
        #                         y_lim=[[0, 1, 2], [[0, 0.1], [-18, -4], [-18, -4]]],
        #                         y_lim2=[[0, 1, 2], [[0, 0.05], [0, 6], [-6, 0]]],
        #                         type_y2='bar'
        #                         )
        # 2 subs
        ascat_bars = np.hstack((plot_dict['ascat_conv_thaw_bar'][0],
                                plot_dict['ascat_conv_thaw_bar'][1],
                                plot_dict['ascat_conv_thaw_bar'][2],
                                plot_dict['ascat_conv_melt_bar'][0],
                                plot_dict['ascat_conv_melt_bar'][1],
                                plot_dict['ascat_conv_melt_bar'][2]))

        second_axis = [
                        np.hstack((plot_dict['npr_conv_bar'][0],
                                   plot_dict['npr_conv_bar'][1],
                                   plot_dict['npr_conv_bar'][2])),
                        ascat_bars,
                      ]
        plot_funcs.plot_subplot([npr_plot, ascat_plot],
                                second_axis,
                                main_label=['npr', '$\sigma^0$ mid', '$\sigma^0$'],
                                figname=figname, x_unit='doy',
                                vline=[vline_secs,
                                       ['k-', 'k-', 'k-', 'r-', 'r-', 'r-', 'b-', 'b-', 'b-'],
                                       ['p', 'p', 'p', 'n', 'n', 'n', 'p1', 'p1', 'p1']],
                                vline_label=bxy.time_getlocaltime(vline_secs, ref_time=[2000, 1, 1, 0])[-2],
                                h_line2=[[0, 1, 1, 1], [0.01, 1, -1, 0], [':', ':', ':', '--']],
                                # annotation_sigma0=[text_qa_w, text_qa_m],
                                # x_lim=[bxy.get_total_sec('20160101'), bxy.get_total_sec('20181230')],
                                y_lim=[[0, 1], [[0, 0.1], [-18, -4]]],
                                y_lim2=[[0, 1], [[0, 0.05], [-4, 15]]],
                                type_y2='bar'
                                )
        if station:
            site_id = 947
            site_measure0 = in_situ_series(site_id, air_measure='air')
    # plotting pixel by pixel
    #     npr_plot = np.array([pixel_2016_new[0][0][0], pixel_2016_new[0][0][1],
    #                          pixel_2017_new[0][0][0], pixel_2017_new[0][0][1],
    #                          pixel_2018_new[0][0][0], pixel_2018_new[0][0][1]])
    return 0


if __name__ == "__main__":
    quick_process(single_check=True)
    hystory(x=-1)
    # hystory(x=1)
    # quick_process()
    # kernel_series_edges()
    plot_KSE('npr_KSE.npz', 'npr_method.png')
    plot_KSE('ascat_KSE.npz', 'ascat_method.png', mode='sigma0')
    quit0()

    # switch_plot_alaska(3)
    # quit0()
    # data_prepare_ascat(p0=['20170102', '20171230'], un_grid=False)
    # hystory(x=1)
    # ms_station_new(sno_all=np.array([52712]), pixel_type='outlier', detectors=[5,7,7])
    # quit0()
    # hystory(x=2)  # get the npz file of regional onset
    # ascat_npy2npz(year0=2016)
    # a
    # quit0()
    switch_plot_alaska(3)
    quit0()
    ms_station_new(sno_all=np.array([52712]), pixel_type='outlier', detectors=[5,7,7])
    quit0()
    # switch_plot_alaska(3)
    # (npz_name, pixel_ind, key_list, ind_table, year0=2016)
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    map_plot1_check_pixel('npz_folder_085_new/onset_all_2018_221217.npz',
                          np.array([44629, 44623, 23054, 23349, 47021, 47320]),
                          ['sigma0_winter', 'sigma0_winter_std', 'sigma0_summer', 'sigma0_summer_std'],
                          n125_n360, year0=2018)
    quit0()
    p_test = np.array([44629, 44623, 23054, 23349])
    quick_location(map_name='npz_folder_085_new/onset_all_2018_221217.npz',  # loc=np.array([[-165.0, 61.5]]),
                   index_1d_input=p_test, pixel_name=p_test, year0=2018)
    quit0()
    array0 = np.loadtxt('npz_indices_outlier.txt', delimiter=',')
    p_test = np.concatenate((array0[0], p_test)).astype(int)
    # np.array([[-144.2, 62.1]])
    ms_get_interest_series(pixel_type='outlier', ascat_index=np.unique(p_test), gk=[7, 7, 7])

    quit0()
    ms_station_new(sno_all=np.array([36470, 50622, 49752, 45229, 52712, 33567]),
                   pixel_type='outlier', detectors=[7, 7, 7])
    quit0()
    pixel_list_new = np.array([48254, 50031])
    array0 = np.loadtxt('map_plot_check_pixel_2016.txt', delimiter=',')
    input_ascat_index = np.unique(np.concatenate((array0[:, 0], pixel_list_new))).astype(int)
    ms_get_interest_series(pixel_type='outlier', ascat_index=input_ascat_index,
                           pixel_name=input_ascat_index, gk=[7, 7, 7])
    quit0()
    pixel_list_new = np.array([48254, 50031])
    array0 = np.loadtxt('map_plot_check_pixel_2016.txt', delimiter=',')
    input_ascat_index = np.unique(np.concatenate((array0[:, 0], pixel_list_new)))
    n125_n360 = np.load('n12_n36_array.npy')
    array_out = map_plot1_check_pixel('npz_folder_085_new2/onset_all_2016_192051.npz', input_ascat_index.astype(int),
                      ['sigma0_winter', 'sigma0_winter_std',
                       'sigma0_melt_mean', 'sigma0_melt_std',
                       'sigma0_summer',
                       'sigma0_min', 'ascat_lvl'],
                        n125_n360, year0=2016)
    quit0()
    # outlier_maps()
    # hystory(x=0)
    # ms_station_new(sno_all=np.array([48241]), pixel_type='outlier', detectors=[7, 7, 7])
    pixel_lon_lat = np.array([[-150.0, 70.0], [-157.5, 70.8]])
    pixel_list_new = np.array([48254, 50031])
    array0 = np.loadtxt('map_plot_check_pixel_2016.txt', delimiter=',')
    input_ascat_index = np.unique(np.concatenate((array0[:, 0], pixel_list_new))).astype(int)
    ms_get_interest_series(pixel_type='outlier', ascat_index=input_ascat_index,
                           pixel_name=input_ascat_index, gk=[7, 7, 7])
    # quick_location(pixel_lon_lat, index_1d_input=interested_pixel, pixel_name=interested_pixel,
    #                map_name='npz_folder_085_new2/onset_all_2018_191813.npz', plot=True, year0=2018)
    # ms_station_new(sno_all=np.array([44056, 47021]), pixel_type='outlier', detectors=[7, 7, 7])
    quit0()
    array0 = np.loadtxt('map_plot_check_pixel_2016.txt', delimiter=',')
    pixel_list_new = np.array([])
    input_ascat_index = np.unique(np.concatenate((array0[:, 0], pixel_list_new)))
    n125_n360 = np.load('n12_n36_array.npy')
    array_out = map_plot1_check_pixel('npz_folder_085/onset_all_2016_15938.npz', input_ascat_index.astype(int),
                      ['sigma0_winter', 'sigma0_winter_std',
                       'sigma0_melt_mean', 'sigma0_melt_std',
                       'sigma0_summer',
                       'sigma0_min', 'ascat_lvl'],
                        n125_n360, year0=2016)
    # ms_get_interest_series(pixel_type='outlier', gk=[7, 12, 10],
    #                        ascat_index=input_ascat_index.astype(int), pixel_name=input_ascat_index.astype(int))
    quit0()
    outlier_maps()
    quit0()
    ms_station_new(pixel_type='outlier', detectors=[7, 7, 7])
    quit0()
    outlier_maps()
    # script_outlier_20190705('npz_folder_085/onset_all_2016_15938.npz', year0=2016)
    # hystory(x=0)
    # n125_n360 = np.load('n12_n36_array.npy').astype(int)
    # array0 = np.loadtxt('map_plot_check_pixel_2016.txt', delimiter=',')
    # array_out = map_plot1_check_pixel('onset_all_2016_91353.npz', array0[:, 0].astype(int),
    #                   ['sigma0_winter', 'sigma0_winter_std',
    #                    'sigma0_melt_mean', 'sigma0_melt_std',
    #                    'sigma0_summer',
    #                    'sigma0_min', 'ascat_lvl'],
    #                   n125_n360)   # onset_all_2018_91425.npz
    ms_station_new(sno_all=['50012'], pixel_type='outlier', detectors=[7, 13, 10])
    quit0()
    # script_outlier_20190705('onset_all_2016_13115.npz')
    # # script_outlier_20190705('onset_all_2018_122010.npz')
    # # ms_station_new(sno_apixel_type='outlier', detectors=[7, 12, 10])
    # quit0()
    array0 = np.loadtxt('map_plot_check_pixel_2016.txt', delimiter=',')
    ms_get_interest_series(False, pixel_type='outlier', gk=[7, 12, 10],
                           input_pixels=[array0[:, 0].astype(int), array0[:, [8, 9]],
                                         array0[:, 10].astype(int), array0[:, 0].astype(int)])
    # quit0()
    # interested_pixel = np.array([30476, 30482, 30476, 30512, 30793])
    # ind_smap, ind_ascat = quick_location(np.array([[-160, 60]]),
    #                                  index_1d_input=interested_pixel, pixel_name=interested_pixel,
    #                                  map_name='onset_all_2016_1229.npz', plot=True)
    # hystory(x=0)
    # quit0()
    # script_outlier_20190705('onset_all_2016_91353.npz')
    a1 = map_plot1(2016, onset_name='onset_all_2016_91353.npz', p_name=np.array([47320]),
               p_index=np.array([[5355], [47320]]), mode='quick')
    ms_station_new(sno_all=['47320'], pixel_type='outlier', detectors=[7, 12, 10])
    quit0()
    lon_lat = np.array([[-160.1, 67.2], [-160, 70], [-160, 69.2], [-162.793, 69.201], [-157.104, 65.828],
                        [-153.1, 67.4], [-150.6, 65.8], [-144.8, 66], [-162.7, 65]])
    interested_pixel_list = np.loadtxt('map_plot_check_pixel_2016.txt', delimiter=',')
    interested_pixel = interested_pixel_list[:, 0].astype(int)
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    array_out = map_plot1_check_pixel('onset_all_2016_302222.npz', interested_pixel,
                      ['sigma0_winter', 'sigma0_winter_std',
                       'sigma0_melt_mean', 'sigma0_melt_std',
                       'sigma0_summer',
                       'sigma0_min', 'ascat_lvl'],
                      n125_n360)
    ms_get_interest_series(False, pixel_type='outlier',
                           input_pixels=[array_out[:, 0].astype(int).astype(str),
                                         [array_out[:, 8:10],
                                          array_out[:, 10].astype(int),
                                          array_out[:, 0].astype(int)]],
                           gk=[7, 15, 10])
    ind_smap, ind_ascat = quick_location(np.array([[-160, 60]]),
                                         index_1d_input=interested_pixel, pixel_name=interested_pixel,
                                         map_name='onset_all_2016_302222.npz', plot=True)

    quit0()
    # hystory(0)
    # ms_station_new(['1091', '1090'], detectors=[7, 15, 10])
    interested_pixel = np.array([27504, 28102, 29908, 30507, 49716, 50912, 50612, 50012, 49718, 47320]).astype(int)
    script_20190627_outliers(npz_name='onset_all_2016_302222.npz', pixel_type='outlier')
    quit0()
    script_20190627_outliers(npz_name='onset_all_2016_1229.npz', pixel_type='outlier',
                             pixel_index=interested_pixel)
    # quit0()
    script_20190627_outliers(npz_name='onset_all_2017_12226.npz', pixel_type='outlier',
                             pixel_index=interested_pixel)
    script_20190627_outliers(npz_name='onset_all_2018_12241.npz', pixel_type='outlier',
                             pixel_index=interested_pixel)

    ms_station_new(np.array([49718, 47320]).astype(str), pixel_type='outlier', detectors=[7, 15, 10])
        # check some pixels



    quit0()
    # ms_station_new(pixel_type='outlier', detectors=[7, 15, 10])
    # hystory(x=1)
    # ms_station_new(['947', '962', '1090', '1177', '2210'], detectors=[7, 15, 10])
    # quit0()
    map_name = 'onset_all_2016_281316.npz'
    index_table_station = np.loadtxt('npz_indices_interest.txt', delimiter=',').astype(int)
    a1 = map_plot1(2016, onset_name=map_name, p_name=index_table_station[2],
                   p_index=[index_table_station[1], index_table_station[0]], mode='quick')
    quit0()
    script_20190627_outliers()
    # ind_smap, ind_ascat = quick_location(np.array([[-160.0, 60.0],
    #                                            [-165.4, 65.6]
    #                                            ]),
    #                                      index_1d_input=np.array([35596, 23323, 33525, 24248, 30507, 27504]),
    #                                      map_name='onset_all_2016_271737.npz')
    quit0()
    # ms_station_new(['948', '949', '950', '952', '960', '1090'], detect=[7, 15, 10])
    ms_station_new(['1094'], detectors=[7, 15, 10])
    quit0()
    ms_station_new(['1089', '1091', '1092', '1093', '1094', '1096'], detectors=[7, 15, 10], site_plot='grid')
    ms_station_new(['1089', '1091', '1092', '1093', '1094', '1096'], detectors=[7, 15, 10])
    hystory_plot_map(site_no=['1089', '1091', '1092', '1093', '1094', '1096'])
    quit0()
    # temp0()
    # ms_station_new(pixel_type='outlier', detect=[7, 15, 10])
    # quit0()
    hystory(x=0)
    quit0()
    # check outlier
    p02, p02_ind_ascat, ons_ascat = find_outlier3()
    n2d, n1d = bxy.latlon2index(np.array([[-150.0, 70], [-156.0, 58.0]]))
    p_temp = np.array([50916])
    p02_ind_ascat = np.concatenate((p02_ind_ascat, p_temp, n1d))
    lons, lats, _ = ind2latlon([0, p02_ind_ascat])
    p02_lonlat = np.array([lons.ravel()[p02_ind_ascat], lats.ravel()[p02_ind_ascat]]).T
    p_table = np.load('n12_n36_array.npy')
    p02_ind_smap = np.array([p_table[1][p_table[0] == p02_0] for p02_0 in p02_ind_ascat])
    ms_get_interest_series(True, pixel_type='outlier', input_pixels=[p02_ind_ascat.astype(str),
                                                                     [p02_lonlat, p02_ind_smap, p02_ind_ascat]])
    ms_station()
    quit0()

    # ms_get_interest_series(False, pixel_type='interest', twin0=[0, 210])
    # quit0()
    # p_latlon0, p_list0 = check_outlier_pixel()
    # p_latlon2 = p_latlon0 + 0.5
    # p_latlon = np.vstack((p_latlon0, p_latlon2))
    # p_list2 = ['%s_nn' % (name0) for name0 in p_list0]
    # p_list = np.concatenate((p_list0, p_list2))
    # a1 = [map_plot1(year0, p_latlon=p_latlon, p_name=p_list) for year0 in [2016, 2017, 2018]]
    # quit0()
    # station series new 201906
    # ms_station_new()
    # print locations
    # p_latlon, p_list = check_outlier_pixel()
    # p_latlon, p_list = check_pixels_125N()

    list_str = ['954', '957', '1003', '1037', '1055', '1062', '1089',
                    '1092', '1267', '1268', '1094']
    lat_list = [site_infos.change_site(list0)[1] for list0 in list_str]
    lon_list = [site_infos.change_site(list0)[2] for list0 in list_str]

    p_lonlat = np.array([lon_list, lat_list]).T
    ms_get_interest_series(True, pixel_type='outlier', input_pixels=[list_str, p_lonlat], time_window=[0, 210])
    ms_station_new(list_str, 'interest', detectors=True)  # list0.astype(str)
    # ms_get_interest_series(True, pixel_type='outlier', input_pixels=[outlier_info[:, 0], outlier_info[:, [3, 4]]])
    quit0()
    p_name, p_latlon, p_onsets, p_ind = find_outlier()
    ms_get_interest_series(False, pixel_type='outlier', input_pixels=[p_name, p_latlon], time_window=[0, 210])
    quit0()

    # ms_get_interest_series(False, pixel_type='interest')
    # quit0()
    # get p_list
    # p_list = site_infos.get_id()
    # ms_station_new(['957', '1183'], 'interest', detect=True)
    outlier_info = np.loadtxt('outlier_info.txt', delimiter=',')
    list0 = outlier_info[:, 0].astype(int)
    ms_station_new(list0.astype(str), 'outlier', detectors=True)

    quit0()

    # target = '7002'
    # p0 = p_latlon[p_list == target]
    # print 'pixel %s, lat/lon: %.2f, %.2f' % (target, p0[0,0], p0[0, 1])
    # quit0()
    points = site_infos.get_id()
    # ms_get_interest_series(False, pixel_type='onset_interest')
    # ms_get_interest_series(False, pixel_type='onset')
    ms_get_interest_series(False, pixel_type='all')
    quit0()
    # ms station also works on specific pixel
    # p_latlon, p_list = check_outlier_pixel()
    # p_list2 =['8%s' % (name0[1:4]) for name0 in p_list]
    # p_list_all = np.concatenate((p_list, p_list2))
    # ms_station_new(p_list_all, 'outlier')
    # quit0()

    # find the 1d, 2d, lat/lon of the errorneous pixel, spec ial check
    p_latlon0, p_list0 = check_outlier_pixel()
    p_latlon2 = p_latlon0 + 0.5
    p_latlon = np.vstack((p_latlon0, p_latlon2))
    p_list2 = ['%s_nn' % (name0) for name0 in p_list0]
    p_list = np.concatenate((p_list0, p_list2))
    a1 = [map_plot1(year0, p_latlon=p_latlon, p_name=p_list) for year0 in [2016, 2017, 2018]]
    quit0()
    # get time series of specified pixels
    points = site_infos.get_id()  # use all stations
    p_latlon = np.zeros([len(points), 2])
    for i0, sno in enumerate(points):
        p0 = site_infos.change_site(sno)
        p_latlon[i0, 0], p_latlon[i0, 1] = p0[2], p0[1]
    twin0 = [0, 210]
    for year0 in [2016, 2017, 2018]:
        xlim0 = [bxy.get_total_sec('%d0101' % year0), bxy.get_total_sec('%d0801' % year0)]
        N125_2d, N125_1d = bxy.latlon2index(p_latlon)
        N360_2d, N360_1d = bxy.latlon2index(p_latlon, resolution=36)
        smap_yearly = get_yearly_smap(t_window=[0, 210], year0=year0)
        start0 = bxy.get_time_now()
        # ascat_dict_yearly = ms_read_ascat(year0, t_window=twin0)
        ascat_dict_yearly = ms_read_ascat(year0, t_window=twin0, pixel_id=N125_2d)
        start1 = bxy.get_time_now()
        print("----read ascat part in %d: %s seconds ---" % (year0, start1-start0))
        n125_n360, ascat_series, npr_series, smp_id = prepare_smap_series(input_ascat=ascat_dict_yearly, input_smap=smap_yearly,
                                                                 ascat_index=N125_1d)
        # np.savetxt('result_08_01/pixel_loc.txt', np.array([n125_n360[0], n125_n360[1], np.array(points).astype(int)]), fmt='%d', delimiter=',')
        onsets, pixels = combine_detect(n125_n360, year0, ascat_series, npr_series, save_sp=True, id_36N=smp_id,
                                        npz_name='onset_outlier')  # use smp_id as input
    quit0()
    # # new station result, 0: interpolated sigma0, 1: not interpolated, now not interpolate
    # y0 = 2016
    # ms_station_new(['1183', '968'])
    # quit0()
    points = site_infos.get_id()
    points.append('1183'), points.append('961'), points.append('952'), points.append('948')
    sp_points = np.zeros([len(points), 2])
    for i0, sno in enumerate(points):
        p0 = site_infos.change_site(sno)
        sp_points[i0, 0], sp_points[i0, 1] = p0[2], p0[1]
    # a0 = [map_plot1(year0, p_latlon=sp_points, p_name=points) for year0 in [2016, 2017, 2018]]
    # p0 = site_infos.change_site('947')
    # p1 = site_infos.change_site('1090')
    # p2 = site_infos.change_site('960')
    # quit0()
    twin0 = [0, 210]
    for year0 in [2016, 2017, 2018]:
        xlim0 = [bxy.get_total_sec('%d0101' % year0), bxy.get_total_sec('%d0801' % year0)]
        p_latlon = sp_points
        N125_2d, N125_1d = bxy.latlon2index(p_latlon)
        N360_2d, N360_1d = bxy.latlon2index(p_latlon, resolution=36)
        smap_yearly = get_yearly_smap(t_window=[0, 210], year0=year0)
        start0 = bxy.get_time_now()
        # ascat_dict_yearly = ms_read_ascat(year0, t_window=twin0)
        ascat_dict_yearly = ms_read_ascat(year0, t_window=twin0, pixel_id=N125_2d)
        start1 = bxy.get_time_now()
        print("----read ascat part in %d: %s seconds ---" % (year0, start1-start0))
        n125_n360, ascat_series, npr_series, smp_id = prepare_smap_series(input_ascat=ascat_dict_yearly, input_smap=smap_yearly,
                                                                 ascat_index=N125_1d)
        # np.savetxt('result_08_01/pixel_loc.txt', np.array([n125_n360[0], n125_n360[1], np.array(points).astype(int)]), fmt='%d', delimiter=',')
        onsets, pixels = combine_detect(n125_n360, year0, ascat_series, npr_series, save_sp=True, id_36N=smp_id)  # use smp_id as input
        # save the results to npz, all pixels

        # plot them in the curve
        # for i0 in np.arange(0, p_latlon.shape[0]):
        #     t_npr, npr, conv_npr = pixels[0][i0][0], pixels[0][i0][1], pixels[0][i0][2]
        #     t_sigma, sigma, conv_sigma = pixels[1][i0][0], pixels[1][i0][1], pixels[1][i0][2]
        #     np.savez('temp_save_pixel_%d_%d' % (i0, year0), *[[t_npr, npr], [t_sigma, sigma], conv_npr, conv_sigma])
        #     plot_funcs.plot_subplot([[t_npr, npr], [t_sigma, sigma]],
        #                             [conv_npr, conv_sigma], main_label=['npr', '$\sigma^0$ mid'],
        #                             figname='ms_pixel_test_%d_%d' % (i0, year0), x_unit='doy', x_lim=xlim0,
        #                             text='lat/lon: %.2f, %.2f' % (p_latlon[i0, 1], p_latlon[i0, 0]))
        # map_plot1(year0, p_latlon=np.array())
    quit0()
    ms_station(insitu="Soil Moisture Percent -2in (pct)")
    quit0()
    points = ['947', '960', '1090']
    sp_points = np.zeros([len(points), 2])
    for i0, sno in enumerate(points):
        p0 = site_infos.change_site(sno)
        sp_points[i0, 0], sp_points[i0, 1] = p0[2], p0[1]
    p0 = site_infos.change_site('947')
    p1 = site_infos.change_site('1090')
    p2 = site_infos.change_site('960')
    a0 = [map_plot1(year0, p_latlon=sp_points) for year0 in [2016, 2017, 2018]]
    quit0()
    for year0 in [2016, 2017, 2018]:
        xlim0 = [bxy.get_total_sec('%d0101' % year0), bxy.get_total_sec('%d0801' % year0)]
        p_latlon = np.array([[-155.5, 66.1], [-156., 65.6]])
        N125_2d, N125_1d = bxy.latlon2index(p_latlon)
        start0 = bxy.get_time_now()
        ascat_dict_yearly = ms_read_ascat(year0, pixel_id=N125_2d)
        start1 = bxy.get_time_now()
        print("----read ascat part in %d: %s seconds ---" % (year0, start1-start0))
        smap_yearly = get_yearly_smap(t_window=[0, 366], year0=year0)
        n125_n360, ascat_series, npr_series, smp_id = prepare_smap_series(input_ascat=ascat_dict_yearly, input_smap=smap_yearly,
                                                                 ascat_index=N125_1d)
        onsets, pixels = combine_detect(n125_n360, year0, ascat_series, npr_series, save_sp=True)
        # plot them in the curve
        for i0 in np.arange(0, p_latlon.shape[0]):
            t_npr, npr, conv_npr = pixels[0][i0][0], pixels[0][i0][1], pixels[0][i0][2]
            t_sigma, sigma, conv_sigma = pixels[1][i0][0], pixels[1][i0][1], pixels[1][i0][2]
            np.savez('temp_save_pixel_%d_%d' % (i0, year0), *[[t_npr, npr], [t_sigma, sigma], conv_npr, conv_sigma])
            plot_funcs.plot_subplot([[t_npr, npr], [t_sigma, sigma]],
                                    [conv_npr, conv_sigma], main_label=['npr', '$\sigma^0$ mid'],
                                    figname='ms_pixel_test_%d_%d' % (i0, year0), x_unit='doy', x_lim=xlim0,
                                    text='lat/lon: %.2f, %.2f' % (p_latlon[i0, 1], p_latlon[i0, 0]))
        map_plot1(year0, p_latlon=np.array())
    # map_plot1(2016)
    a0 = [map_plot1(year0) for year0 in [2016, 2017, 2018]]
    quit0()
    # station with soil moisture
    ms_station(insitu="Soil Moisture Percent -2in (pct)")
    quit0()
    # map_plot1(2018)
    # quit0()
    # ms_regional_onset(2017)
    # quick_alaska_onset(2016)
    # quit0()
    years = [2016, 2017, 2018]
    # map_plot1(years[0])
    for year0 in years:
        quick_alaska_onset(year0)
    quit0()
    for year0 in years:
        xlim0 = [bxy.get_total_sec('%d0101' % year0), bxy.get_total_sec('%d0901' % year0)]
        p_latlon = np.array([[-155.5, 66.1], [-156., 65.6]])
        N125_2d, N125_1d = bxy.latlon2index(p_latlon)
        start0 = bxy.get_time_now()
        ascat_dict_yearly = ms_read_ascat(year0, pixel_id=N125_2d)
        start1 = bxy.get_time_now()
        print("----read ascat part in %d: %s seconds ---" % (year0, start1-start0))
        smap_yearly = get_yearly_smap(t_window=[0, 366], year0=year0)
        n125_n360, ascat_series, npr_series, smp_id = prepare_smap_series(input_ascat=ascat_dict_yearly, input_smap=smap_yearly,
                                                                 ascat_index=N125_1d)
        onsets, pixels = combine_detect(n125_n360, year0, ascat_series, npr_series, save_sp=True)
        # plot them in the curve
        for i0 in np.arange(0, p_latlon.shape[0]):
            t_npr, npr, conv_npr = pixels[0][i0][0], pixels[0][i0][1], pixels[0][i0][2]
            t_sigma, sigma, conv_sigma = pixels[1][i0][0], pixels[1][i0][1], pixels[1][i0][2]
            np.savez('temp_save_pixel_%d_%d' % (i0, year0), *[[t_npr, npr], [t_sigma, sigma], conv_npr, conv_sigma])
            plot_funcs.plot_subplot([[t_npr, npr], [t_sigma, sigma]],
                                    [conv_npr, conv_sigma], main_label=['npr', '$\sigma^0$ mid'],
                                    figname='ms_pixel_test_%d_%d' % (i0, year0), x_unit='doy', x_lim=xlim0,
                                    text='lat/lon: %.2f, %.2f' % (p_latlon[i0, 1], p_latlon[i0, 0]))
    # plot them in the map
    value_npz = np.load('onset_%d.npz' % year0)
    onset = value_npz['ascat_onset']  # arr_1
    v_valid = (onset!=0)&(onset!=-999)
    onset[v_valid] = bxy.time_getlocaltime(onset[v_valid], ref_time=[2000, 1, 1, 0])[-2]
    value_array = data_process.make_mask(onset)
    quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160],
                      fig_name='finer_ascat_%d' % year0, points=p_latlon)
    value_array2 = data_process.make_mask(value_npz['ascat_lvl'])  # arr_3
    quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_lvl_%d' % year0, z_value=[-2, 10], points=p_latlon)
    quit0()
    n125_n360, ascat_series, npr_series, smp_id = prepare_smap_series(input_smap=smap_yearly, year0=year0,
                                           savename='p_coord', ascat_index=N125_1d)
    combine_detect(n125_n360, year0, ascat_series, npr_series, save_sp=False)

    quit0()
    value_npz = np.load('onset_%d.npz' % year0)
    onset = value_npz['arr_1']
    v_valid = (onset!=0)&(onset!=-999)
    onset[v_valid] = bxy.time_getlocaltime(onset[v_valid], ref_time=[2000, 1, 1, 0])[-2]
    value_array = data_process.make_mask(onset)
    quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160], points=p_latlon)
    value_array2 = data_process.make_mask(value_npz['arr_3'])
    quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_lvl', z_value=[-2, 10], points=p_latlon)
    quit0()
    year0=2018
    ms_read_ascat(year0)
    ms_map(year0)
    quit0()
    # map
    year0=2017
    # ms_regional_onset(year0)
    ms_map(year0)

    year0=2018
    ms_read_ascat(year0)
    ms_map(year0)
    quit0()
    # station
    ms_station()
    quit0()
    ms_station_prepare()
    quit0()

    # 051
    value_npz = np.load('onset_2016.npz')
    onset = value_npz['arr_1']
    v_valid = (onset!=0)&(onset!=-999)
    onset[v_valid] = bxy.time_getlocaltime(onset[v_valid], ref_time=[2000, 1, 1, 0])[-2]
    value_array = data_process.make_mask(onset)
    quick_plot_map_v2(value_array, resolution=12.5, z_value=[30, 160])
    value_array2 = data_process.make_mask(value_npz['arr_3'])
    quick_plot_map_v2(value_array2, resolution=12.5, fig_name='finer_lvl', z_value=[-2, 10])
    # mask='/home/xiyu/PycharmProjects/R3/result_05_01/other_product/mask_ease2_125N.npy',
    #                   mask_value = [-999, 0]
    quit0()
    # data process
    year0 = 2016
    path_files = get_yearly_files(t_window=[0, 210], year0=year0)
    mask0 = np.load('/home/xiyu/PycharmProjects/R3/result_05_01/other_product/mask_ease2_125N.npy')  # 0: ocean, 1: land
    ascat_att0=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore',
                                  'inc_angle_trip_fore', 'sigma0_trip_mid', 'inc_angle_trip_mid']
    dict_2016 = data_process.ascat_alaska_grid_v3(ascat_att0, path_files, pid=np.where(mask0>0))
    get_finer_onset(input_ascat=dict_2016)
    quit0()

    # before
    get_regional_ascat(year0=year0)
    quit0()
    get_alaska_onset(year0=2016)
    quit0()
    quick_plot_map([-1], s_info=['947', -65.12422, -146.73390])
    quit0()
    year0 = 2016
    get_regional_ascat(year0=year0)
    get_alaska_onset(year0=year0)
    quick_plot_map([-1], s_info=['947', -65.12422, -146.73390], year0=year0)
    quit0()
    site_nos = site_infos.get_id(mode='int')
    # [get_3_year_insitu(sno0, m_name='snow') for sno0 in site_nos]
    a00 = [ascat_all_mode_series(sno0) for sno0 in site_nos]
    quit0()
    start0 = bxy.get_time_now()
    get_regional_ascat()
    start1 = bxy.get_time_now()
    print("----angular correct part: %s seconds ---" % (start1-start0))
    quit0()
    site_nos = site_infos.get_id(mode='int')
    tri_years_plotting(win=[0, 210], id0=site_nos)
    quit0()
    a00 = [ascat_all_mode_series(sno0) for sno0 in np.array([960, 2065, 968, 1090, 947, 2211, 2213, 1175]).astype(int)]
    quit0()
    tri_years_plotting(win=[0, 210])
    data_prepare_ascat(p0=['20151001', '20151231'], year_no=2015, win=[0, 210])
    quit0()
    year0 = 2016
    name_secs_txt, name_doy_txt = 'result_agu/result_2019/onset_secs_%d.txt' % year0, \
                                    'result_agu/result_2019/onset_doy_%d.txt' % year0
    onset_in_secs = np.loadtxt(name_secs_txt, delimiter=',')
    sno_in_table = onset_in_secs[:, 0]
    snos = site_infos.get_id(mode='int')
    air0_array = np.zeros([snos.size, 12])
    i0 = 0
    for sno0 in snos:
        is_site = sno_in_table==sno0
        if sum(is_site) > 0:
            n_date = onset_in_secs[is_site, 1]
            a_date = onset_in_secs[is_site, 2]
        else:
            continue
        t_air_date, snd0, snd1, air0, air1 = in_situ_tair_snd(sno0, npr_date=n_date, ascat_date=a_date)
        air_static0, air_static1 = bxy.get_statics(air0[1]), bxy.get_statics(air1[1])
        air_gt0_days = np.array([sum(air0[1] > 0), sum(air1[1] > 0)])
        air0_array[i0, 0:4] = np.array([sno0, t_air_date, n_date, a_date])
        air0_array[i0, 4:7], air0_array[i0, 7:10], air0_array[i0, 10: 12] = air_static0, air_static1, air_gt0_days
        i0 += 1
    np.savetxt('./result_agu/result_2019/air_variation.txt', air0_array, delimiter=',', fmt='%.1f')
    quit0()
    # get in situ measurements during melting period
    name_secs_txt, name_doy_txt = 'result_agu/result_2019/onset_secs_%d.txt' % year0, \
                                    'result_agu/result_2019/onset_doy_%d.txt' % year0
    onset_results = np.loadtxt(name_secs_txt, delimiter=',')
    sm = in_situ_during_onsets("Soil Moisture Percent -2in (pct)", fromfile=onset_results)
    tsoil, tair, snow \
        = in_situ_during_onsets("Soil Temperature Observed -2in (degC)", fromfile=onset_results), \
          in_situ_during_onsets("Air Temperature Observed (degC)", fromfile=onset_results), \
          in_situ_during_onsets("snow", fromfile=onset_results)
    np.save('period_insitu_%d' % year0, np.array([sm, tsoil, tair, snow]))
    # ploting period measurement
    period_name = ['t1', 't2', 't1t2_npr', 't1_t2_ascat']
    period_insitu = np.load('period_insitu_%d.npy' % year0)
    period_insitu_plot(period_insitu, year0, 2)
    period_insitu_plot(period_insitu, year0, 3)
    quit0()

    # save the result (secs, day of year), onset, time series, and in situ measurements
    onset_results = compare_stational(y=year0, all_site=True, id_check=[968])
    name_secs_txt, name_doy_txt = 'result_agu/result_2019/onset_secs_%d.txt' % year0, \
                                    'result_agu/result_2019/onset_doy_%d.txt' % year0
    onset_doy = onset_results.copy()
    for i0, onset0 in enumerate(onset_results):
        doy0 = bxy.time_getlocaltime(onset0[1: -1])[-2]  # check the reftime
        onset_doy[i0, 1: -1] = doy0
    np.savetxt(name_secs_txt, onset_results, delimiter=',', fmt='%d')
    np.savetxt(name_doy_txt, onset_doy, delimiter=',', fmt='%d')
    # in_situ_during_onsets("Soil Moisture Percent -2in (pct)", fromfile=onset_results)
    quit0()
    # combinng and detection
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175',
                '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'] #'1089'
    # site_nos = ['1177']
    site_nos_int = [int(str0) for str0 in site_nos]
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    points_info = []
    points_index = []
    for sno in site_nos:
        s_info = site_infos.change_site(sno)
        points_info.append(s_info)
        dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
        p_index = np.argmin(dis_1d)  # nearest
        # temp check the distance of neighbor pixels
        nn_index = np.argsort(dis_1d)[0:9]
        if sno == '947':
            p_index = 4547
        points_index.append(p_index)
    # odd_plot_in_map([-1], s_info=['947', -65.12422, -146.73390])  # plot the map
    land_id = points_index
    land_ids=np.array(points_index)
    np.savetxt('result_agu/result_2019/points_num_tp.txt', np.array([land_ids, site_nos_int]), delimiter=',', fmt='%d')
    combining2(np.arange(1, 200), y=year0, pixel_plot=True, onset_save=False, land_id=land_ids, id_name=np.array(site_nos_int),
               ascat_atts=['sigma0_trip_mid', 'inc_angle_trip_mid', 'utc_line_nodes'])
    quit0()

    # save the result (secs, day of year), onset, time series, and in situ measurements
    onset_results = compare_stational(y=year0, all_site=True, id_check=[968])
    name_secs_txt, name_doy_txt = 'result_agu/result_2019/onset_secs_%d.txt' % year0, \
                                    'result_agu/result_2019/onset_doy_%d.txt' % year0
    onset_doy = onset_results.copy()
    for i0, onset0 in enumerate(onset_results):
        doy0 = bxy.time_getlocaltime(onset0[1: -1])[-2]  # check the reftime
        onset_doy[i0, 1: -1] = doy0
    np.savetxt(name_secs_txt, onset_results, delimiter=',', fmt='%d')
    np.savetxt(name_doy_txt, onset_doy, delimiter=',', fmt='%d')
    # in_situ_during_onsets("Soil Moisture Percent -2in (pct)", fromfile=onset_results)
    quit0()
    # combinng and detection
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175',
                '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'] #'1089'
    # site_nos = ['1177']
    site_nos_int = [int(str0) for str0 in site_nos]
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    points_info = []
    points_index = []
    for sno in site_nos:
        s_info = site_infos.change_site(sno)
        points_info.append(s_info)
        dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
        p_index = np.argmin(dis_1d)  # nearest
        # temp check the distance of neighbor pixels
        nn_index = np.argsort(dis_1d)[0:9]
        if sno == '947':
            p_index = 4547
        points_index.append(p_index)
    # odd_plot_in_map([-1], s_info=['947', -65.12422, -146.73390])  # plot the map
    land_id = points_index
    land_ids=np.array(points_index)
    np.savetxt('result_agu/result_2019/points_num_tp.txt', np.array([land_ids, site_nos_int]), delimiter=',', fmt='%d')
    combining2(np.arange(1, 200), y=2018, pixel_plot=True, onset_save=False, land_id=land_ids, id_name=np.array(site_nos_int),
               ascat_atts=['sigma0_trip_mid', 'inc_angle_trip_mid', 'utc_line_nodes'])
    quit0()

    for doy0 in np.arange(366+365+160, 366+365+259):
        t_str=bxy.doy2date(doy0, fmt='%Y%m%d')
        spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap_area_result', orbit='A')
        spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap_area_result', orbit='D')
    quit0()
    # data preparing smap
    t_window = ['2018.06.09', '2018.09.06']
    doy_array, year_no = bxy.get_doy_array(t_window[0], t_window[1], fmt='%Y.%m.%d')
    Read_radar.radar_read_alaska('_D_', ['alaska'], t_window, 'vv', year=year_no)
    Read_radar.radar_read_alaska('_A_', ['alaska'], t_window, 'vv', year=year_no)
    for doy0 in doy_array:
        t_str=bxy.doy2date(doy0, fmt='%Y%m%d', year0=year_no)
        spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap_area_result', orbit='A')
        spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap_area_result', orbit='D')
    quit0()
    # # data preparing ascat
    lat_gd, lon_gd = spt_quick.get_grid()
    for doy0 in np.arange(366+365+1, 366+365+159):
        status = Read_radar.read_ascat_alaska(doy0, year0=2016)
        if status == -1:  # not nc data for this specific date
            continue
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d")
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
    quit0()
    # # data preparing ascat new, updated 2019 03 05
    lat_gd, lon_gd = spt_quick.get_grid()
    # metopB 2017
    year_no = 2017
    doy_0501 = bxy.get_doy(['20170501'], year0=year_no)
    doy_0731 = bxy.get_doy(['20170731'], year0=year_no)
    for doy0 in np.arange(1, doy_0501):
        status = Read_radar.read_ascat_alaska(doy0, year0=2017, sate='B')
        if status == -1:  # not nc data for this specific date
            continue
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d", year0=year_no)
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
    # metopA 2017
    for doy0 in np.arange(1, doy_0731):
        status = Read_radar.read_ascat_alaska(doy0, year0=year_no, sate='A')
        if status == -1:  # not nc data for this specific date
            continue
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d", year0=year_no)
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, sate='A', format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
        # spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
    quit0()
    doy = np.arange(0, 2)
    lat_gd, lon_gd = spt_quick.get_grid()
    for doy0 in doy:
        Read_radar.read_ascat_alaska(doy0, year0=2016)

    # 201902
    onset_results = compare_stational()
    compare_onset_insitu("Soil Moisture Percent -2in (pct)", fromfile=onset_results)
    compare_onset_insitu("Soil Temperature Observed -2in (degC)", fromfile=onset_results)
    compare_onset_insitu("Air Temperature Observed (degC)", fromfile=onset_results)
    quit0()
    sm_array = compare_onset_insitu_plot("Soil Moisture Percent -2in (pct)", mode='site')
    quit0()
    land_ids = np.loadtxt('result_agu/result_2019/points_num.txt', delimiter=',')
    # land_ids = land_ids[:, land_ids[1]>2211]
    drawing_parameter_v2(land_ids, isplot=True, orbit='A', ks=[9, 9])
    quit0()
    # "Air Temperature Observed (degC)", "Soil Temperature Observed -2in (degC)", "Soil Moisture Percent -2in (pct)"
    # insitu_vlaue_a = compare_onset_insitu("Soil Moisture Percent -2in (pct)")
    compare_onset_insitu("Soil Moisture Percent -2in (pct)")
    quit0()
    land_ids = np.loadtxt('result_agu/result_2019/points_num.txt', delimiter=',')
    drawing_parameter_v2(land_ids)
    drawing_parameter_v2(land_ids, orbit='D')
    quit0()
    site_nos = ['947', '949', '950', '960', '962', '967', '968','1090','1175',
                '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'] #'1089'
    # site_nos = ['947']
    site_nos_int = [int(str0) for str0 in site_nos]
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    points_info = []
    points_index = []
    for sno in site_nos:
        s_info = site_infos.change_site(sno)
        points_info.append(s_info)
        dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
        p_index = np.argmin(dis_1d)  # nearest
        # temp check the distance of neighbor pixels
        nn_index = np.argsort(dis_1d)[0:9]
        if sno == '947':
            p_index = 4547
        points_index.append(p_index)
    # odd_plot_in_map([-1], s_info=['947', -65.12422, -146.73390])  # plot the map
    land_id = points_index
    land_ids=np.array(points_index)
    np.savetxt('result_agu/result_2019/points_num.txt', np.array([land_ids, site_nos_int]), delimiter=',', fmt='%d')
    combining2(np.arange(0, 360), pixel_plot=True, onset_save=False, land_id=land_ids, id_name=np.array(site_nos_int),
               ascat_atts=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes'])

    quit0()
    # comparison between amsr2 results and other results such as smap t1, t2, and t3
    # amsr2_l3_comparison()
    # quit0()
    # land_ids = np.loadtxt('result_agu/result_2019/points_num.txt', delimiter=',')
    # drawing_parameter(land_ids)

    snowmelt_amsr2 = []
    # site_nos = ['947', '949', '950', '960', '962', '967', '968','1090', '1175',
    #         '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    site_nos = ['947', '968', '2213']
    s_info, s_measurements, s_secs = Read_radar.read_amsr2_l3(site_nos, prj='EQMD')
    # dimensions: ('date', 'atts', 'location', 'variables')
    s_info_d, s_measurements_d, s_secs_d = Read_radar.read_amsr2_l3(site_nos, prj='EQMD')
    # test the pass hour
    d_pass = np.abs(s_measurements_d[:, 1, 0, 0])/60
    a_pass = np.abs(s_measurements[:, 1, 0, 0])/60
    s_measurements[s_measurements < -30000] = np.nan
    for i_no, site_no in enumerate(site_nos):
        fig_name = 'result_08_01/%s_snow_amsr2.png' % site_no
        p_doy = bxy.time_getlocaltime(s_secs, ref_time=[2000, 1, 1, 0], t_out='utc')[3]
        air_measure, air_sec = read_site.get_secs_values(site_no, "Air Temperature Observed (degC)",
                                                           p_doy, nan_value=-0.5, pass_hr=13)
        if site_no in ['2065', '2081']:
            air_measure, air_sec = read_site.get_secs_values(site_no, "Air Temperature Average (degC)",
                                                           p_doy, nan_value=-0.5)
        snow_measure, snow_sec = read_site.get_secs_values(site_no, 'snow', p_doy)
        snd_50 = data_process.zero_find([s_secs, s_measurements[:, 0, i_no, 0].ravel()], th=5)
        pass_hr = np.abs(s_measurements[:, 1, i_no, 0]/60)
        # test time zone
        sec00 = bxy.get_total_sec('20160101')
        tuple00 = bxy.time_getlocaltime([sec00], ref_time=[2000, 1, 1, 0], t_out='utc')
        # plotting
        if site_no in ['947', '949', '950', '967', '1089']:
            snow_label = 'SWE (mm)'
        else:
            snow_label = 'SND (mm)'
            snow_measure *= 10
        # value: s_measurements[:, 0, i_no, 0], time: 1
        # mainaxes0 : [x0, y0, x1, y1 ...]
        plot_funcs.plot_subplot([[s_secs, s_measurements[:, 0, i_no, 0].ravel(), snow_sec, snow_measure],
                             [s_secs, s_measurements[:, 0, i_no, 1].ravel(), s_secs, pass_hr]],
                            [[air_sec, air_measure], [air_sec, air_measure]],
                            main_label=['SND (mm) vs %s' % snow_label, 'SWE (mm) vs %s' % snow_label],
                            x_unit='sec', vline=[[snd_50-4*3600*24], ['r-'], ['snow free']],
                            figname='result_08_01/2019_snow_%s_.png' % (site_no),
                            main_syb = ['k-', 'r-', 'b-'])
        snowmelt_amsr2.append(snd_50-4*3600*24)
    melt_date_amsr2 = bxy.time_getlocaltime(snowmelt_amsr2, ref_time=[2000, 1, 1, 0])[3]
    # with open('result_agu/result_2019/result_amsr2.txt', 'w') as f0:
    #     f0.writelines(site_nos)
    #     f0.write('\n')
    #     f0.writelines(melt_date_amsr2)

    np.savetxt('result_agu/result_2019/result_amsr2.txt', np.array([site_nos, melt_date_amsr2]).astype(int).T,
               delimiter=',', fmt='%d', header='id,melt_amsr2')

    quit0()
    # site_nos = ['947', '949', '950', '960', '962', '967', '968','1090','1175',
    #         '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    #
    # for sno in site_nos:
    #     s_info = site_infos.change_site(sno)
    #     with open('station_location.txt', 'a') as f0:
    #         f0.write('%s,%4.f,%4.f' % (s_info[0], s_info[1], s_info[2]))
    # quit0()
    # plot results
    # odd_plot_in_map([-1], s_info=['947', -65.12422, -146.73390])  # plot the map
    # quit0()
    # all_result = np.load('20181104_result.npz')
    # for fname in all_result.files:
    #     np.save('result_agu/%s_onset.npy' % fname, all_result[fname])
    # quit0()
    # tiff plot 20181204
    # v_z = np.load('result_agu/smap_thaw_obdh_onset.npy')[0]   # shape 2, 9000
    # v_z_doy = bxy.time_getlocaltime(v_z)[-2]
    # v_z2 = np.zeros(v_z.size) + 100
    # h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151102.h5'
    # h0 = h5py.File(h5_name)
    # lons_grid = h0['cell_lon'].value.ravel()
    # lats_grid = h0['cell_lat'].value.ravel()
    # index = (lons_grid>-170) & (lons_grid<-140) & (lats_grid>50) & (lats_grid<72)
    # array_new = np.array([lons_grid[index], lats_grid[index], v_z_doy[index]]).T
    # array_new = np.array([lons_grid, lats_grid, v_z_doy]).T
    # array_sort = array_new[array_new[:, 0].argsort()]
    # np.savetxt('source_obdh.csv', array_sort, '%.3f', delimiter=',', header='Lon,Lat,Var', comments='')
    # build vrt
    # data_process.vrt_write(array_sort[:, 0].size, array_sort[:, 1].size, 'source_obdh.txt')
    # data_process.csv2tif_projected('source_obdh.csv', 'source_obdh.tif', 3408)

    # land_id = np.array([4648, 3770, 5356])
    # combining2(np.arange(0, 360), pixel_plot=True, onset_save=False, land_id=land_id,
    #            ascat_atts=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes'])
    # quit0()

    s_infos = [['947', 65.12422, -146.73390], ['949', 65.07833, -145.87067],
             ['950', 64.85033, -146.20945], ['1090', 65.36710, -146.59200], ['960', 65.48, -145.42],
             ['962', 66.74500, -150.66750], ['1233', 59.82001, -156.99064], ['2213', 65.40300, -164.71100],
             ['2081', 64.68582, -148.91130], ['2210', 65.19800, -156.63500], ['2211', 63.63900, -158.03010],
             ['2212', 66.17700, -151.74200], ['1175', 67.93333, -162.28333],
             ['2065', 61.58337, -159.57708], ['967', 62.13333, -150.04167], ['968', 68.61683, -149.30017],
             ['1177', 70.26666, -148.56666]]
    s_infos = [[2563, 61.22422, -162.63390], [1, 60.90, -162.01]]  # [0, 66.22422, -146.73390]
    # s_info=[4265, -64.3, -155.80]
    ## 20181104, the level, conv, two time series in regions where no melt event was detected
    site_nos = ['947', '949', '950', '960', '962', '967', '968','1090','1175',
                '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213'] #'1089'
    # site_nos = ['947']
    site_nos_int = [int(str0) for str0 in site_nos]
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    points_info = []
    points_index = []
    for sno in site_nos:
        s_info = site_infos.change_site(sno)
        points_info.append(s_info)
        dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
        p_index = np.argmin(dis_1d)  # nearest
        # temp check the distance of neighbor pixels
        nn_index = np.argsort(dis_1d)[0:9]
        # if sno == '947':
        #     p_index = 4547
        points_index.append(p_index)
    # odd_plot_in_map([-1], s_info=['947', -65.12422, -146.73390])  # plot the map
    land_id = points_index
    land_ids=np.array(points_index)
    np.savetxt('result_agu/result_2019/points_num.txt', np.array([land_ids, site_nos_int]), delimiter=',', fmt='%d')
    combining2(np.arange(0, 360), pixel_plot=False, onset_save=True, land_id=land_ids, id_name=np.array(site_nos_int),
               ascat_atts=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes'])

    quit0()

    ## 20181104, the level, conv, two time series in regions where no melt event was detected
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    points_info = []
    points_index = []
    for s_info in s_infos:
        points_info.append(s_info)
        dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
        p_index = np.argmin(dis_1d)  # nearest
        points_index.append(p_index)
    # land_id = np.array([4546, 4847, 4754, 3647, 5356, 4647, 4569, 5858, 2453, 2859, 4349, 3959, 3459, 4555, 3770, 5263,
    #       3867, 3542, 5267])
    # land_id = np.array([3770, 4546])

    land_id = np.array(points_index)
    combining2(np.arange(0, 360), pixel_plot=False, onset_save=True, land_id=land_id,
               ascat_atts=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes'])
    print land_id
    quit0()
    # 2018.11.20
    # combine_detection_ad([5356], [5356], ascat_detect=True)
    # quit0()
    # -57, 366+55
    combining2(np.arange(0, 360), pixel_plot=False, onset_save=True,
               ascat_atts=['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes'])
    #
    # combining2(np.arange(-57, 366+55), ascat_atts=['sigma',
    #                                                'incidence', 'pass_utc'])
    quit0()
    temp_test.plot_ascat_ad_difference(5356)
    temp_test.plot_ascat_ad_difference(3770)
    # quit0()
    # 2018.11.20 obtain regions
    lat_gd, lon_gd = spt_quick.get_grid()
    spt_quick.ascat_area_plot2('20151108', lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
    start0 = bxy.get_time_now()
    for doy0 in np.arange(-57, 366+55):
        # Read_radar.read_ascat_alaska(doy0, year0=2016, sate='B')
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d")
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5', attributs=[2, 11, 8, -1, 5])
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5', attributs=[2, 11, 8, -1, 5])
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, sate='A', format_ascat='h5', attributs=[2, 11, 8, -1, 5])
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5', attributs=[2, 11, 8, -1, 5])
    start1 = bxy.get_time_now()
    print("----save h5 files: %s seconds ---" % (start1-start0))
    quit0()
    # applying a sub_region, pixels out side this region are masked
    lat_range = [68, 71]
    lon_range = [-161, -145]
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    pixel_index = np.where((lons_1d > lon_range[0]) & (lons_1d < lon_range[1]) & \
               (lat_range > lat_range[0]) & (lat_range < lat_range[1]))
    # 2018.11.10
    combine_detection_ad([5356], [5356], ascat_detect=True)
    quit0()
    # att_list_smap = ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_v_fore', 'cell_tb_h_fore', 'cell_tb_time_seconds_fore',
    #                  'cell_tb_time_seconds_aft']
    # att_list_smap2 = ['cell_tb_v_aft']
    # # ak_series(np.arange(-60, 365), smap=True, orbit='A', smap_format='h5', att_list=att_list_smap)
    # ak_series(np.arange(-60, 365), smap=True, orbit='D', smap_format='h5', att_list=att_list_smap, add_att=True)
    # # ak_series(np.arange(-60, 365), smap=True, orbit='D', smap_format='h5', att_list=att_list_smap)
    # # ak_series(np.arange(-60, 365), smap=True, orbit='A', smap_format='h5', att_list=att_list_smap)
    # quit0()
    Read_radar.radar_read_alaska('_D_', ['alaska'], ['2015.12.01', '2016.12.31'], 'vv')

    # Read_radar.radar_read_alaska('_A_', ['alaska'], ['2015.12.01', '2016.12.31'], 'vv')
    for doy0 in range(-60, 365):
        t_str=bxy.doy2date(doy0, fmt='%Y%m%d')
        spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap_area_result', orbit='D')
    #     # spt_quick.smap_area_plot(t_str, save_dir='./result_08_01/area/smap_area_result', orbit='A')
    # quit0()
    # # 2018.11.09
    # # doy = np.arange(1, 250)
    # # # for doy0 in doy:
    # # #     Read_radar.read_ascat_alaska(doy0, year0=2018)
    # # for doy0 in doy:
    # #     t_str = bxy.doy2date(doy0, fmt="%Y%m%d", year0=2018)  # 1 is 20160101
    # #     spt_quick.ascat_area_plot2(t_str, 0, 0)
    # # # ak_series(np.arange(-60, 366+55))
    # # quit0()
    # # # generate 2000.01.01
    # # doys = np.arange(0, 250)
    # # for doy0 in doys:
    # #     t0 = bxy.doy2date(doy0, year0=2018, fmt='%Y.%m.%d')
    # #     with open('smap_2018_foldername.txt', 'a') as f0:
    # #         f0.write('%s\n' %t0)
    # # quit0()

    ######################################################## above are newly updated############################
    ascat_sigma = 3
    level00 = 4
    # combine_detection(np.array([30, 180]), np.array([250, 340]), ascat_detect=True, onset_save=True, sigma_ascat=3,
    #                   melt_zone=20)
    # combine_detection(np.array([30, 180]), np.array([250, 340]), ascat_detect=True, onset_save=True, sigma_ascat=ascat_sigma,
    #                   melt_zone=20)

    # cal the difference
    # d0 = np.load('result_08_01/test_difference_melt_7.npy')
    # d1 = np.abs(d0 - np.nanmean(d0))
    # mdev = np.nanmedian(d1)
    # s = d1/mdev if mdev else 0.
    # d0[s>3] = np.nan
    os1, os2 = np.load('result_08_01/test_onset0_7.npy'), np.load('result_08_01/melt_onset_ascat_7.npy')
    level1 = np.load('result_08_01/melt_level_new_7.npy')
    os1[os1==-999] = np.nan
    os2[os2==-999] = np.nan
    os_new = os1 - os2
    # os_new[os_new<-60]=np.nan
    os_new[(level1<level00)|(level1>900)] = 0  # mask the difference in less significant region
    np.save('result_08_01/test_difference_melt_7_new.npy', os_new)  # npr - ascat, snowmelt
    os2[(level1<level00)|(level1>900)] = 0
    np.save('result_08_01/melt_onset_ascat_38_7.npy', os2)
    os1[(os1<30)|(os1>150)] = 0

    # combine_detection(np.array([30, 180]), np.array([250, 340]), ascat_detect=True, onset_save=True, sigma_ascat=3)
    # quit0()

    ################ the station detections## ############################
    # # 20181031, test melt detection at station

    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    # site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    # site_nos = ['947', '968']
    # # get 1d index
    # points_index = []
    # points_info = []
    # points_dict = {}
    # for sno in site_nos:
    #     s_info = site_infos.change_site(sno)
    #     points_info.append(s_info)
    #     dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
    #     p_index = np.argmin(dis_1d)  # nearest
    #     points_index.append(p_index)
    #     points_dict[sno] = p_index
    # # 2_1 the specific results based on NPR and ASCAT
    # # list0 = site_infos.change_site('968')
    # melt_out = melt_map(points_info, pixel_index=points_index, pixel_id=site_nos)


    ## 20181104, the level, conv, two time series in regions where no melt event was detected
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    points_info = []
    points_index = []
    # for s_info in [[0, 66.2, -145.0], [1, 66.7, -145.0], [2, 64.7, -145.0], [5356, 68.61683, -149.30017],
    #                [2751, 60.3, -155.00017], [22, 60.5, -159.6], [4648, 65.12422, -146.73390],
    #                [4646, 65.07833, -145.87067], [4569, 67.93333, -162.28333]]:
    # for s_info in [[4669, 68.23333, -161.90333], [4648, 65.12422, -146.73390], [0, 66.22422, -152.83390],
    #                [4870, 68.83333, -162.30333], [5159, 68.63333, -152.80333],1
    #                [4547, 64.82422, -146.73390], [2360, 60.3, -161.30333], [2756, 60.8, -158.0],
    #                [4054, 64.42422, -152.83390], [4849, 65.82422, -146.73390], [5554, 68.82422, -146.73390],
    #                [5254, 68.02422, -148.13390], [5151, 67.22422, -146.73390], [3964, 65.6, -160.0]]:

    for s_info in [[3770,  65.40300, -164.71100], [3770,  65.40300, -164.71100], [0, 66.6, -159.80]]:
        points_info.append(s_info)
        dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
        p_index = np.argmin(dis_1d)  # nearest
        points_index.append(p_index)

    # 2_1 the specific results based on NPR and ASCAT
    list0 = site_infos.change_site('968')
    melt_out = melt_map(points_info, pixel_index=points_index, pixel_id=points_index, ascat_sigma=ascat_sigma)
    s_info = site_infos.change_site('1090')
    # 70.26666, -148.56666
    #
    s_info = [0, s_info[2], s_info[1]]
    s_info = [0,  -151.5, 62.1]
    # s_info = [0, 1, 1]
    odd_latlon = [s_info[2], s_info[1]]
    thaw_win = np.array([30, 180])
    fr_win = np.array([250, 340])
    odd_points_rc, odd_points_1d = data_process.latlon2rc(odd_latlon)
    for custom3 in [
                    # 'result_08_01/melt_level_7.npy',
                    # 'melt_conv_ascat_7.npy',
                    # 'result_08_01/ascat_winter_std_7.npy',
                    # 'result_08_01/ascat_winter_mean_7.npy',
                    'result_08_01/test_difference_melt_7_new.npy'
                    ]:
        # no odd pixel are plotted
        points_index = [-1]
        data_process.ascat_onset_map('A', odd_point=np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]]),
                                 points_index=np.array(points_index),
                                 product='customize', mask=False, mode=['_norm_'], version='old', std=7,
                                 f_win=fr_win, t_win=thaw_win,
                                 custom=['result_08_01/test_onset0_7.npy',
                                         'result_08_01/melt_onset_ascat_38_7.npy',
                                         # 'result_08_01/melt_onset_ascat_7.npy',
                                         custom3])
    quit0()
    ## 20181101, test tibet 1, remote sensing data read
    # Read_radar.radar_read_main('_A_', ['20000', '20001', '20002', '20003'], ['2015.11.01', '2016.08.01'], 'vv',
    #                            pre_path='result_08_01/20181101/smap_tibet')
    # Read_radar.radar_read_main('_D_', ['20000', '20001', '20002', '20003'], ['2015.11.01', '2016.08.01'], 'vv',
    #                            pre_path='result_08_01/20181101/smap_tibet')
    # doy0 = range(-60, 210)
    # for doy in doy0:
    #     Read_radar.getascat(['20000', '20001', '20002', '20003'], doy)
    # spt_quick.ascat_point_plot(site_nos=['20000', '20001', '20002', '20003'], site_loc='tibet')
    for sno in ['20000', '20001', '20002', '20003']:
    # for sno in ['20001']:
        exp_tibet(sno)
    quit0()
    spt_quick.ascat_point_plot(site_nos=['20000', '20001', '20002', '20003'], site_loc='tibet')
    quit0()
    site_nos_new = ['20000', '20001', '20002', '20003']
    for site_no in site_nos_new:
        for orb in ['_A_', '_D_']:
            full_path = './result_08_01/20181101/smap_series/tb_'+site_no+orb+'tibet'
            Read_radar.read_tb2txt(site_no, orb, fname=full_path, attribute_name='smap_ta_lonlat_colrow',
                                   year_type='tibet', is_inter=True, ipt_path='_08_01', site_loc='tibet')
            print '%s-%s has been all extracted' % (site_no, orb)
    quit0()

    exp_soil_tb_sigma('ali')
    # test
    # 20181101,
    combine_detection(np.array([30, 180]), np.array([250, 340]), ascat_detect=True, onset_save=True)
    # 20181031, test melt detection at station
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_1d = h0['cell_lon'].value.ravel()
    lats_1d = h0['cell_lat'].value.ravel()
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    # get 1d index
    points_index = []
    points_info = []
    points_dict = {}
    for sno in site_nos:
        s_info = site_infos.change_site(sno)
        points_info.append(s_info)
        dis_1d = bxy.cal_dis(s_info[2], s_info[1], lons_1d, lats_1d)
        p_index = np.argmin(dis_1d)  # nearest
        points_index.append(p_index)
        points_dict[sno] = p_index
    # 2_1 the specific results based on NPR and ASCAT
    # list0 = site_infos.change_site('968')
    melt_out = melt_map(points_info, pixel_index=points_index, pixel_id=site_nos)
    quit0()
    # 20181018
    # 2_0 add satellite type

    # 2_4 find snow masked index: 2 pixels
    path = './result_05_01/other_product/snow_mask_360_2.npy'
    mask_snow = np.load(path)
    mask_snow_1d = mask_snow.ravel()
    snow_pixel_index = np.where(mask_snow_1d != 0)[0]

    # 2_3 amsr2, npr and ascat at stations
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    points_site = []
    for sno in site_nos:
        list0 = site_infos.change_site(sno)
        points_site.append(list0)
    # 2_1 the specific results based on NPR and ASCAT
    pi = [4648, 4646, 4546, 4847, 4754, 3647, 5356, 4647, 4569, 5858, 2453, 2859, 4349, 3959, 3459, 4555, 3770, 5263,
          3867, 3542, 5267]
    pi = [3772, 3768, 3773, 3868, 3865, 3866]
    pi = [5356]
    melt_out = melt_map(points_site, pixel_index=pi)
    # melt_out_structure = np.dtype({'names': ['index', ]})
    for out0 in melt_out:
        savename = 'result_08_01/odd_pixel_npr_ascat_%d' % out0[0]
        np.save(savename, np.array([out0[1], out0[2]]))
    np.save('melt_out_pixel_index', np.array(pi))

    # 2_2 amsr2
    op_list = pi
    amsr2_out = spt_quick.amsr2_detection(orbit='_A_', att=['Brightness Temperature (res23,18.7GHz,H)',
                                                 'Brightness Temperature (res23,18.7GHz,V)',
                                   'Brightness Temperature (res23,36.5GHz,H)',
                                   'Brightness Temperature (res23,36.5GHz,V)'],
                              odd_plot=op_list, extract_point=op_list,
                              is_plot=False)
    for out0 in amsr2_out:
        savename_time = 'result_08_01/odd_pixel_amsr2_time_%d' % out0[0]
        savename_att = 'result_08_01/odd_pixel_amsr2_attribute_%d' % out0[0]  # check the order of attribute
        np.save(savename_time, np.array(out0[1]))
        np.save(savename_att, np.array(out0[2]))
        # print savename_time

    # 3_1 plot pixel station, region, and snow masked
    for p0 in pi:
        amsr2_time = np.load('./result_08_01/odd_pixel_amsr2_time_%d.npy' % (p0))
        amsr2_value = np.load('./result_08_01/odd_pixel_amsr2_attribute_%d.npy' % (p0))
        npr_ascat = np.load('./result_08_01/odd_pixel_npr_ascat_%d.npy' % (p0))
        # npr_ascat, 0: npr_time, ascat_time, conv_time, 1: values
        # plot
        fname = 'result_08_01/amsr2_npr_sigma_%d.png' % p0
        # set unvalid value of amsr2
        amsr2_value[:, 0][amsr2_value[:, 0]<150] = np.nan
        amsr2_value[:, 2][amsr2_value[:, 2]<150] = np.nan
        plot_funcs.plot_subplot([[amsr2_time, amsr2_value[:, 0]],
                                 [npr_ascat[0][0], npr_ascat[1][0]],
                                 [npr_ascat[0][1], npr_ascat[1][1]]],
                                [[amsr2_time, amsr2_value[:, 2]]],
                                main_label=['19 and 35 (green)', 'npr', 'sigma'], figname=fname, red_dots=False,
                                x_unit='doy', symbol2='g.', x_lim=[502243200.0, 521078400.0])
    quit0()
    # find index of odd point based on latitude and longitude

    #1 ascat, metop A and B, both orbits
    compare_metop('947')
    quit0()

    quit0()
    # build_subgrid()
    # quit0()
    # read amsr2 data of this region
    Read_radar.read_amsr2(['alaska'], ['2015.12.01', '2016.07.01'], orb='A', th=[8.58, 17.54])
    Read_radar.read_amsr2(['alaska'], ['2015.12.01', '2016.07.01'], orb='D', th=[8.58, 17.54])
    spt_quick.amsr2_area_resample(['Brightness Temperature (res23,18.7GHz,H)',
                                   'Brightness Temperature (res23,18.7GHz,V)',
                                   'Brightness Temperature (res23,36.5GHz,H)',
                                   'Brightness Temperature (res23,36.5GHz,V)'], 'result_08_01/area/amsr2_resample',
                                  raw_path='result_08_01/area/amsr2', grid_name='alaska')
    # Read_radar.read_amsr2(['north'], ['2015.11.01', '2016.07.31'], orb='D')
    quit0()
    north_region()
    custom=['thaw_onset_ascat_7.npy', 'melt_onset_ascat_7.npy', 'melt_conv_ascat_7.npy']
    # ak_series(np.arange(-60, 366+55), ascat_format='h5')
    quit0()
    # spt_quick.ascat_point_plot(sate='A')
    # spt_quick.ascat_point_plot(sate='B')
    # doy = np.arange(-60, 366+60)
    # # 0815/2018
    # melt_map([[1, 1, 1]])
    # melt_map([[0,  -162.7, 69.1], [0,  -155.2, 69.5], [0, -153.5, 68.8], [0, -147.5, 68.2], [0, -153.5, 67.8],
    #                [0, -159.1, 60.5], [0, -159.0, 62.2], [0, -163.0, 61.2], [0, -150.3, 64.7], [0, -147.3, 64.4], [0, -150.0, 62.0], [0, -162.5, 65.5]
    #                ,[0, -162.5, 63.0],  [0, -150.3, 66.7], [0, -147.3, 66.2]])
    # # melt_map([[0,  0, 0]])
    # # quit0()
    # # add all odd pixels and labeled in the melt map
    # if os.path.exists('pixel_index.txt'):
    #     odd_pixel_index = np.loadtxt('pixel_index.txt')
    #     data_process.ascat_onset_map('A', product='customize', mask=False, mode=['_norm_'],
    #                                 version='old', std=7, f_win=np.array([250, 340]), t_win=np.array([30, 180]),
    #                                  custom=['test_onset0_7.npy', 'melt_onset_ascat_7.npy', 'melt_conv_ascat_7.npy'])
    # quit0()

    lat_gd, lon_gd = spt_quick.get_grid()
    for doy0 in np.arange(-60, 366+55):
        # Read_radar.read_ascat_alaska(doy0, year0=2016, sate='A')
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d")
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, sate='A', format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
    quit0()
    for kernel0 in [7]:
        s_info = site_infos.change_site('1090')
        # 70.26666, -148.56666
        #
        s_info = [0, s_info[2], s_info[1]]
        s_info = [0,  -151.5, 62.1]
        # s_info = [0, 1, 1]
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
                print p_index
                # p_index = [4859, 4850]
            combine_detection(thaw_win, fr_win, sigma_npr=kernel0, ascat_detect=True, odd_plot=p_index, sigma_ascat=7)
        else:
            combine_detection(thaw_win, fr_win, sigma_npr=kernel0, ascat_detect=True)
    data_process.ascat_onset_map('A', odd_point=np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]]),
                                 product='customize', mask=False, mode=['_norm_'], version='old', std=kernel0,
                                 f_win=fr_win, t_win=thaw_win,
                                 custom=['thaw_onset_ascat_7.npy', 'melt_onset_ascat_7.npy', 'melt_conv_ascat_7.npy'])
    # print 'check location: ', np.array([odd_points_rc[0], odd_points_rc[1], s_info[2], s_info[1]])

    quit0()
    ak_series(np.arange(-60, 366+55), ascat_format='h5')
    quit0()
    lat_gd, lon_gd = spt_quick.get_grid()
    for doy0 in np.arange(-60, 366+55):
        # Read_radar.read_ascat_alaska(doy0, year0=2016, sate='A')
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d")
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, sate='A', format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, sate='A', format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0, format_ascat='h5')
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1, format_ascat='h5')
    quit0()
    doy = np.arange(0, 2)
    lat_gd, lon_gd = spt_quick.get_grid()
    for doy0 in doy:
        Read_radar.read_ascat_alaska(doy0, year0=2016)
    for doy0 in doy:
        t_str = bxy.doy2date(doy0, fmt="%Y%m%d")  # 1 is 20160101
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=0)
        spt_quick.ascat_area_plot2(t_str, lat_gd, lon_gd, orbit_no=1)
    quit0()
    ak_series(np.arange(-60, 366+55))
    quit0()
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
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    # site_nos = ['1233']
    # site_nos = ['947', '962', '967', '968', '1177', '2210', '2211', '2213']
    for sno in site_nos:
        compare_metop(sno)
    quit0()
    # 07/09, 2018
    doy = np.arange(-60, 366+60)
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    for doy0 in doy:
        Read_radar.getascat(site_nos, doy0, sate='A')
        # Read_radar.getascat(site_nos, doy0, sate='B')
    spt_quick.ascat_point_plot()
    quit0()
    melt_map([[1, 1, 1]])
    # melt_map([[0,  -162.7, 69.1], [0,  -155.2, 69.5], [0, -153.5, 68.8], [0, -147.5, 68.2], [0, -153.5, 67.8],
    #                [0, -159.1, 60.5], [0, -159.0, 62.2], [0, -163.0, 61.2], [0, -150.3, 64.7], [0, -147.3, 64.4], [0, -150.0, 62.0], [0, -162.5, 65.5]
    #                ,[0, -162.5, 63.0],  [0, -150.3, 66.7], [0, -147.3, 66.2]])
    # melt_map([[0,  0, 0]])
    # quit0()
    # add all odd pixels and labeled in the melt map
    if os.path.exists('pixel_index.txt'):
        odd_pixel_index = np.loadtxt('pixel_index.txt')
        data_process.ascat_onset_map('A', product='customize', mask=False, mode=['_norm_'],
                                    version='old', std=7, f_win=np.array([250, 340]), t_win=np.array([30, 180]),
                                     custom=['test_onset0_7.npy', 'melt_onset_ascat_7.npy', 'melt_conv_ascat_7.npy'])
        # points_index=odd_pixel_index.astype(int)
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