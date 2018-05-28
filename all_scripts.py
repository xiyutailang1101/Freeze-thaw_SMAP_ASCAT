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
                        i_0 = (arr1[:, 0] > 89)  # wetland
                        p_6 = np.sum(arr1[:, 1][i_0])
                        pp = np.array([p_1, p_4, p_0, p_6, p_3, p_5, p_2])
                        i_0 = np.where(pp>5)
                        label0s = {'11': 'water', '12': 'perennial snow', '31': 'Barren Land', '41': 'Forest', '51': 'Shrub', '71': 'Grass', '90': 'wetland'}
                        label1s = ['Water', 'Perennial Snow', 'Barren Land', 'Forest', 'Shrub',  'Grass', 'wetland']
                        label_color = {'Water': 'aqua', 'Perennial Snow': 'snow', 'Barren Land': 'brown', 'Forest': 'forestgreen',
                                       'Shrub': 'olive',  'Grass': 'palegreen', 'wetland': 'steelblue'}
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


def test_method(ft, txt=False):
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
    plt.savefig(fig_fname)


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


def disscus_sm_variation(sno):
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
            vlines = [[0, 0], ons0, ons1, [0, ons_new[1]] ]
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
            # label location
            # if plt_col == 0:
            #     a=0
            #     for axi in [ax2_2, ax3_2, ax4_2]:
            #         axi.get_yaxis().set_visible(False)
            #         for yticks in axi.yaxis.get_major_ticks():
            #             yticks.label2.set_visible(False)
            # else:
            #     for axi in [ax2, ax3, ax4]:
            #         axi.get_yaxis().set_visible(False)
            #         for yticks in axi.yaxis.get_major_ticks():
            #             yticks.label1.set_visible(False)
        plt.tight_layout()
        plt.savefig('test00')
    return 0


def discuss_combining():
    # site_nos = ['2213']
    site_nos = ['2211', '2213']
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
        sigconv, sigseries, ons_new, gg, sig_pass, peakdate_sig = \
                data_process.ascat_plot_series(site_no, orb_no=0, inc_plot=False, sigma_g=k_width) # 0 for ascending

        tbv1, tbh1, npr1, gau1, ons1, sitetime, peakdate1 = test_def.main(site_no, [], sm_wind=7, mode='annual', tbob='_A_', sig0=k_width)  # result npr
        ax0 = plt.subplot2grid((4, 1), (i+1, 0))
        axs.append(ax0)
        npr = [[npr1[0], npr1[1]*100], [gau1[0], gau1[2]*100]]
        _, ax0_2, l2 = pltyy(npr[0][0], npr[0][1], 'test_comp2', 'NPR ($10^{-2}$)',
                     t2=sigseries[0], s2=sigseries[1], label_y2='$\sigma_{45}^0$ (dB)',
                     symbol=l_symbol, handle=[0, ax0], nbins2=6, label_x='')
        # plot snow
        swe_value, swe_date = read_site.read_measurements(site_no, 'snow', np.arange(366, 366+365), hr=0)
        swe_date-=365
        ax_sn = ax0.twinx()
        ax_sn.spines["right"].set_position(("axes", 1.3))
        plot_funcs.make_patch_spines_invisible(ax_sn)
        ax_sn.spines["right"].set_visible(True)
        ax_sn.plot(swe_date, swe_value, 'k:', label="snow depth")
        ax_sn.fill_between(swe_date, 0, swe_value, facecolor='grey')
        ax_sn.set_ylim([0, 150])

        ax_sn.yaxis.set_major_locator(MaxNLocator(4))
        ax0.set_xlim([50, 150])
        ax0.set_ylim([-9, 8])
        ax0.yaxis.set_major_locator(MaxNLocator(4))
        ax0_2.set_ylim([-22, -4])
        yticks = ax0.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        yticks2 = ax0_2.yaxis.get_major_ticks()
        yticks2[-1].label2.set_visible(False)
        ax0.text(0.92, 0.95, text4[i], transform=ax0.transAxes, va='top')

        if i < 1:
            xticks = ax0.axes.get_xticklabels()
            for xt in xticks:
                xt.set_visible(False)
            ax_sn.set_ylabel("Snow depth (cm)")
            # ax_sn.get_yaxis().set_label_coords(0, 0)
        if i<1:
            grey_patch = test_def.make_patch("grey")
            leg0 = ax0.legend([l2[0], l2[1], grey_patch], ['NPR', '$\sigma_{45}^0$', 'Snow depth cm'], bbox_to_anchor=(0., 1.02, 1., 1.02), ncol=3, loc=3, prop={'size': 10}, frameon=False)
            # leg0.get_frame().set_linewidth(0.0)


    axs[-1].set_xlabel('Day of year 2016')
    plt.tight_layout()
    fig0.subplots_adjust(hspace=0.2)
    plt.savefig('test03.png')


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
    hr0 = site_infos.ascat_heads(0)
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


def gdal_clips(sno, ipt='snowf'):
    # gdalwarp -cutline ascat_1126200.shp -crop_to_cutline ims2016127_1km_GIS_v1.3.tif ims126_3.ti
    shp_path = './result_08_01/point/ascat/shp/ascat_shp/ascat_%s*.shp' % sno
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
        value_name0 = '%s/ascat_*_%s_value.npy' % (value_path, sno)
        value_name = glob.glob(value_name0)[0]
        value0 = np.load(value_name)
        value1 = value0[0, :, :]
        value2 = value1[value1[:, -1] > -999]  # the ASCAT measure
        pixel_ids = value2[:, -1]
        snow_lc_name0 = 'result_08_01/point/ascat/pixels/npys/ascat_*_%s_snowlc.txt' % (sno)
        snow_lc_name = glob.glob(snow_lc_name0)[0]
        snow_lc = np.loadtxt(snow_lc_name, delimiter=',')  # the ASCAT ancillary
        mask0 = np.in1d(value2[:, -1], snow_lc[:, 0])
        i0 = 0
        measure_anc_array = np.zeros([180, 60]) - 999
        for id0 in snow_lc[:, 0]:
            pixel_value = value2[pixel_ids==id0]  # find the same pixel
            if pixel_value.shape[0]>1:
                pixel_value = pixel_value[0]
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
                  ['9001', '9002', '9003', '9004', '9005', '9006', '9007']]  # region 1 to 5
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


def regions_plotting(region_id=1, site_no=0, att_xyz=['tair', 'mid', '# date'], exception=False, xlim0=[-30, 30]):
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
    # value_xyz, att_xyz = trip_read(region_id=1, site_no=0, att_xyz=['tair', 'mid', '# date'], exception=False)
    # tair, sigma_mid, z_value = value_xyz[0], value_xyz[1], value_xyz[2]
    # plotting
    fig = plt.figure(figsize=[6, 4.5])
    ax = fig.add_subplot(1, 1, 1)
    m1 = ax.scatter(x_value, sigma_mid, c=z_value, cmap=plt.get_cmap('rainbow'))
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


if __name__ == "__main__":
    # 0508/2018, some area work
    site_nos_new = ['957', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'ns6', 'ns7']
    site_nos_new = ['957', '9001', '9002', '9003', '9004', '9005', '9006', '9007']  # '957',

    site_nos_new = ['957', '9001', '9002', '9003', '9004', '9005', '9006', '9007',
                    '948', '958', '963', '2080', '947', '949', '950',
                    '960', '962', '967', '968', '1090', '1175', '1177', '1233',
                    '2065', '2081', '2210', '2211', '2212', '2213'
                    ]
    for sno in site_nos_new:
        # s_info = site_infos.change_site(sno)
        # s_info_t = np.array([s_info[2], s_info[1], int(s_info[0])]).reshape(3, -1)
        # print sno
        # spt_quick.ascat_sub_pixel(s_info_t, dis0=4, site_nos=[sno])
        # after shape file
        gdal_clips(sno, ipt='lc')
        gdal_clips(sno, ipt='snowf')
        ascat_snow_lc(sno)
        print sno
        check_lc_snow_ascat(sno)
    quit0()
    ascat_snowlc_npy(site_nos_new)
    rid = 5
    sno_sp = regions_extract(rid)
    beams = 'fore'
    land_cover(rid, sno_sp, ['ever', 'decid', 'shrub'], False)  # 2nd var is a list of sites
    check_pass_time(rid)
    regions_plotting(region_id=rid, att_xyz=['# date', beams, 'snowf'], xlim0=[1, 366])
    regions_plotting(region_id=rid, att_xyz=['# date', beams, 'ID'], xlim0=[1, 366])
    # regions_plotting(region_id=rid, att_xyz=['tair', beams, 'snowf'])
    # regions_plotting(region_id=rid, att_xyz=['tair', beams, 'swe'])
    # regions_plotting(region_id=rid, att_xyz=['tair', beams, 'ID'])
    # regions_plotting(region_id=rid, att_xyz=['tair', beams, 'dswe'])
    # regions_plotting(region_id=rid, att_xyz=['dswe', beams, 'swe'])
    for sno in sno_sp:
        #regions_plotting(sno)
        # regions_plotting(region_id=rid, site_no=sno, att_xyz=['tair', beams, '# date'])  # xlim0=[1, 366]
        regions_plotting(region_id=rid, site_no=sno, att_xyz=['# date', beams, 'snowf'], xlim0=[1, 366])
        # regions_plotting(region_id=rid, site_no=sno, att_xyz=['dswe', beams, 'swe'])
    # bxy.test_read_txt()
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
    #discuss_combining()
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