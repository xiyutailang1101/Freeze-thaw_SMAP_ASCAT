__author__ = 'xiyu'
import numpy as np
import basic_xiyu as bxy
import matplotlib.pyplot as plt
import plot_funcs
import glob
import re
import h5py
import spt_quick

def check_daily_ascat(fname):
    value = np.load(fname)
    value_asc = value[value[:, -1]==0]
    np.savetxt('test0412_asc.txt', value_asc, delimiter=',', fmt='%.5f',
               header='latitude, longitude, sigma0_trip0, sigma0_trip1, sigma0_trip2, f_usable0, f_usable1, ' \
               'f_usable2, inc_angle_trip0, inc_angle_trip1, inc_angle_trip2, f_land0, f_land1, f_land2, ' \
               'utc_line_nodes, abs_line_number, sat_track_azi, swath_indicator, kp0, kp1, kp2, ' \
               'azi_angle_trip0, azi_angle_trip1, azi_angle_trip2,  num_val_trip0, num_val_trip1, num_val_trip2, ' \
               'f_f0, f_f1, f_f2, f_v0, f_v1, f_v2, f_oa0, f_oa1, f_oa2, f_sa0, f_sa1, f_sa2, f_tel0, f_tel1, f_tel2,' \
               'f_ref0, f_ref1, f_ref2, as_des')
    value_special = value_asc[np.abs(value_asc[:, 0] - 65.0928878357) < 1e-4]
    np.savetxt('test0412_1103131.txt', value_special, delimiter=',', fmt='%.5f',
               header='latitude, longitude, sigma0_trip0, sigma0_trip1, sigma0_trip2, f_usable0, f_usable1, ' \
               'f_usable2, inc_angle_trip0, inc_angle_trip1, inc_angle_trip2, f_land0, f_land1, f_land2, ' \
               'utc_line_nodes, abs_line_number, sat_track_azi, swath_indicator, kp0, kp1, kp2, ' \
               'azi_angle_trip0, azi_angle_trip1, azi_angle_trip2,  num_val_trip0, num_val_trip1, num_val_trip2, ' \
               'f_f0, f_f1, f_f2, f_v0, f_v1, f_v2, f_oa0, f_oa1, f_oa2, f_sa0, f_sa1, f_sa2, f_tel0, f_tel1, f_tel2,' \
               'f_ref0, f_ref1, f_ref2, as_des')


def check_subpixel():
    site_nos_new = ['948', '957', '958', '963', '2080', '947', '949', '950',
            '960', '962', '967', '968', '1090', '1175', '1177', '1233',
            '2065', '2081', '2210', '2211', '2212', '2213']
    path0 = 'result_08_01/point/ascat/time_series'
    for sno in site_nos_new:
     fname0 = '%s/*%s_corner*' % (path0, sno)
     fname_list = glob.glob(fname0)
     if len(fname_list) > 1 or len(fname_list)<1:
         print 'repeated corner coordinates files'
         return -1
    test00 = np.load(fname_list[0])[0, :, :]
    test00_1 = test00[test00[:, -1] > -999]
    if test00_1.size < 9:
        print 'can not determine the corners at data %s' % (fname_list[0])
        return 1
    return 2


def check_9001():
    directory0 = 'result_08_01/point/ascat/time_series/'
    fname = 'ascat_20160110_0528_957_value.npy'
    fnamec = 'ascat_20160110_0528_957_corner.npy'
    test0 = np.load(directory0+fname)
    test1 = test0[0, :, :]
    test_xy = test1[test1[:, -1] == 1009200]
    print test0.shape
    print test_xy
    testc = np.load(directory0+fnamec)
    testc1 = testc[0, :, :]
    testc_xy = testc1[testc1[:, -1] == 1009200]
    print testc_xy


def check_corner_value():
    value0 = np.load('test_value.npy')
    corner0 = np.load('test_corner.npy')
    sec0, id0 = value0[0, :, 14], corner0[0, :, -1]
    idx = sec0 > -900
    idx_01 = id0 > -900
    sec0, id0 = sec0[idx], id0[idx]
    corner_id = corner0[0, :, -1]
    corner_id = corner_id[idx]
    date_list = bxy.time_getlocaltime(sec0, ref_time=[2000, 1, 1, 0])
    id_daily = 1e6+date_list[-2, :]*1e3+date_list[-1, :]*10
    test0 = np.abs(id0-id_daily)  # compare the sec0 and pixle id
    test1 = np.abs(corner_id-id0)
    print 'the size of id0 %d, the unique size %d' % (id0.size, np.unique(id0).size)
    print 'passing second and the pixel id:', np.where(test0>2)
    print 'the passing id from corner: ', id0[test0>2]
    print 'the passing id from passing secs: ', id_daily[test0>2]
    print 'passing second and the corner id:', np.where(test1>2)


def check_ease_grid2():
    ease_lat_un = np.fromfile('/home/xiyu/Data/easegrid2/gridloc.EASE2_N36km/EASE2_N36km.lats.500x500x1.double', dtype=float).reshape(500, 500)
    ease_lon_un = np.fromfile('/home/xiyu/Data/easegrid2/gridloc.EASE2_N36km/EASE2_N36km.lons.500x500x1.double', dtype=float).reshape(500, 500)
    lat_1d, lon_1d = ease_lat_un.ravel(), ease_lon_un.ravel()
    lat_range = np.array([54, 72])
    lon_range = np.array([-170, -130])
    bbox = [[lon0, lat0] for lon0 in lon_range for lat0 in lat_range]
    bbox_r_c = []
    for corner0 in bbox:
        dis0 = bxy.cal_dis(corner0[1], corner0[0], lat_1d, lon_1d)
        nn_index = np.argmin(dis0)
        row_num0, col_num0 = nn_index/500, nn_index-nn_index/500*500
        bbox_r_c.append([row_num0, col_num0])
    aoi_lat, aoi_lon = ease_lat_un[141: 215, 165: 241], ease_lon_un[141: 215, 165: 241]

    return 0


def check_text_read():
    fname = './result_07_01/txtfiles/site_tb/tb_968_A_2016.txt'
    with open(fname, 'rb') as as0:
        for row in as0:
            print row
            row0 = re.split(', |,|\n', row[2:])
            print row0
            break


def check_meta():
    meta_file = 'meta0_ascat_ak.txt'
    meta_list = []
    with open(meta_file) as meta0:
        content = meta0.readlines()
        metas = [x.strip() for x in content]
        # atts = row0.split(',')
    return 0


def tp_connect():
    ascat_grid_lat, ascat_grid_lon = np.load('lat_ease_grid.npy'), np.load('lon_ease_grid.npy')
    smap_h5 = 'SMAP_alaska_A_GRID_20160105.h5'
    h0 = h5py.File(smap_h5)
    smap_grid_lat, smap_grid_lon = h0[u'cell_lat'].value, h0[u'cell_lon'].value
    ascat_lat, ascat_lon, smap_lat, smap_lon = \
        ascat_grid_lat.ravel(), ascat_grid_lon.ravel(), smap_grid_lat.ravel(), smap_grid_lon.ravel()
    ascat_table_row, ascat_table_col = np.zeros([smap_lat.size, 9]) - 99, np.zeros([smap_lat.size, 9]) - 99
    for id0 in range(0, smap_lat.size):
        # print id0
        smap_ascat_table0 = np.zeros([2, 9]) - 99
        smap0 = [smap_lat[id0], smap_lon[id0]]
        dis = bxy.cal_dis(smap_lat[id0], smap_lon[id0], ascat_lat, ascat_lon)
        sub9 = np.argsort(dis)[0: 9]
        for i0, subi in enumerate(sub9):
            tp_rc = bxy.trans_in2d(subi, [300, 300])
            smap_ascat_table0[0, i0] = tp_rc[0]
            smap_ascat_table0[1, i0] = tp_rc[1]
        ascat_table_row[id0], ascat_table_col[id0] = smap_ascat_table0[0], smap_ascat_table0[1]
    np.savetxt('ascat_row_table.txt', ascat_table_row, fmt='%d', delimiter=',')
    np.savetxt('ascat_col_table.txt', ascat_table_col, fmt='%d', delimiter=',')


def check_smap_grid(smap_tbv):
    # check locations:
    h00 = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20160103.h5')
    ease_lat, ease_lon = h00[u'cell_lat'].value, h00[u'cell_lon'].value
    rc = bxy.geo_2_row([ease_lon, ease_lat], [-146.73390, 65.12422])
    check_series = smap_tbv[rc[0], rc[1], :]
    h5_list = sorted(glob.glob('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_*.h5'))
    check_series2 = np.zeros(len(h5_list)) - 88
    for i0, h5_name0 in enumerate(h5_list):
        daily_h0 = h5py.File(h5_name0, 'r')
        daily_tbv = daily_h0[u'cell_tb_v_aft'].value
        check_series2[i0] = daily_tbv[45, 48]
        daily_h0.close()


def check_daily_data():
    h0 = h5py.File('SMAP_alaska_A_20160103.h5', 'r')
    h00 = h0['North_Polar_Projection']
    col = h00['cell_column'].value
    row = h00['cell_row'].value
    tbv = h00[u'cell_tb_v_aft'].value
    np.savetxt('test_daily_data.txt', np.array([row, col, tbv]).T, fmt='%d, %d, %.2f')
    return 0


def colored():
    x = np.linspace(0,2,100)
    y = np.linspace(0,10,100)

    z = [[np.sinc(i) for i in x] for j in y]

    CS = plt.contourf(x, y, z, 20, # \[-1, -0.1, 0, 0.1\],
                            cmap=plt.cm.rainbow)
    plt.colorbar(CS)
    plt.plot(x,2+plt.sin(y), "--k")

def find_bias(std):
    th_name = 'test_onset0_%s.npy' % std
    fr_name = 'test_onset1_%s.npy' % std
    onset0 = np.load(th_name)
    onset1 = np.load(fr_name)  # test_onset1.npy
    onset0_14 = np.load('test_onset0_14.npy')
    onset1_14 = np.load('test_onset1_14.npy')
    onset0_bias = onset0_14-onset0
    gt18 = np.where(onset0_bias.ravel()<-18)[0]
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%d.h5' % 20151102
    h0 = h5py.File(h5_name)
    lons_grid = h0['cell_lon'].value
    lats_grid = h0['cell_lat'].value
    taget_lons, taget_lats = lons_grid.ravel()[gt18], lats_grid.ravel()[gt18]
    idx00 = np.where((taget_lons>-144)&(taget_lons<-142)&(taget_lats>61)&(taget_lats<63))
    rc_2d = bxy.trans_in2d(gt18[idx00], [90, 100])
    print 's equals 14, thaw onset is ', onset0_14[rc_2d[0, 0], rc_2d[1, 0]]
    print 's equals 7, thaw onset is ', onset0[rc_2d[0, 0], rc_2d[1, 0]]
    return [taget_lons[idx00], taget_lats[idx00], gt18[idx00], rc_2d]


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
        if site_no == '2213':
            soil_t, soil_t_date = read_site.read_measurements(site_no, "Soil Temperature Observed -2in (degC)",
                                                              np.arange(366, 366+365), hr=18)
            soil_sm, soil_sm_date = read_site.read_measurements(site_no, "Soil Moisture Percent -2in (pct)",
                                                                np.arange(366, 366+365), hr=18)
            soil_t_date-=365
            soil_sm_date-=365
            ax_3rd = plt.subplot2grid((4, 1), (i+1, 0))
            axs.append(ax_3rd)
            _, ax_3rd2, l2 = pltyy(soil_sm_date, soil_sm, 'test_comp2', 'VWC (%)',
                                 t2=soil_t_date, s2=soil_t, label_y2='T$_{soil}$ ($^\circ$C)',
                                 symbol=['k-', 'b-'], handle=[0, ax_3rd], nbins2=6)
            ax_3rd.set_xlim(xlimit)
            ax_3rd.set_ylim([0, 60])
            ax_3rd2.set_ylim([-30, 10])
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
        ax0.text(0.92, 0.2, text4[i], transform=ax0.transAxes, va='top')
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

    cax = fig0.add_axes([0.12, 0.1, 0.6, 0.05])
    cb2 = colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, ticks=[-10, -5, 0, 5, 10, 15], orientation='horizontal',
                                label='Air temperature ($^\circ$C)')
    axs[-1].set_xlabel('Day of year 2016')
    plt.tight_layout()
    fig0.subplots_adjust(hspace=0.2)
    plt.savefig('test03.png', dpi=300)


def plot_compare_smap_interpolation(txt0, txt1):
    series0, series1 = np.loadtxt(txt0), np.loadtxt(txt1)
    flags = series1[:, [0, 4]]
    print series0[:, [0, 4]][100]
    plot_funcs.plot_interp_time_series([series0[:, [0, 3]], series1[:, [0, 3]]], ['tbv_v17', 'tbv_v18'])
    return 0


def check_ascat_timeseries():
    row_nums = 10
    ascat_series = np.load('ascat_s1090_2016.npy')
    # print 'the first %d rows: ' % row_nums, ascat_series[0: row_nums]
    heads = 'the first %d rows: ' % row_nums
    np.savetxt('test_ascat_timeseries.txt', ascat_series[0: row_nums], fmt='%.4f', delimiter=',', header=heads)
    return 0


if __name__ == "__main__":
    check_ascat_timeseries()
    doy = bxy.get_doy(['20160101'])
    print doy
    # fname = 'tb_968_A_2016.txt'
    # plot_compare_smap_interpolation('tp/temp_timeseries_0730/'+fname, 'result_07_01/txtfiles/site_tb/'+fname)


    # ipt0 = {'thaw_npr':  [-2, -4, 2], 'thaw_tb': [-2, 5, 2], 'thaw_ascat': [-2, 3, 2],
    #         'freeze_npr': [-1, 8, 2], 'freeze_tb': [-1, 6, 2], 'freeze_ascat': [-1, 4, 2]}
    # for key0 in ipt0.keys():
    #     plot_funcs.plot_comparison('result_08_01/point/onset_result/onset_result.csv', ipt0[key0], key0)
    # # before May
    # N = 5
    # menMeans = (20, 35, 30, 35, 27)
    # womenMeans = (25, 32, 34, 20, 25)
    # menStd = (2, 3, 4, 1, 2)
    # womenStd = (3, 5, 2, 3, 3)
    # ind = np.arange(N)    # the x locations for the groups
    # width = 0.35       # the width of the bars: can also be len(x) sequence
    #
    # p1 = plt.bar(ind, menMeans, width, yerr=menStd)
    # p2 = plt.bar(ind, womenMeans, width,
    #              bottom=menMeans, yerr=womenStd)
    #
    # plt.ylabel('Scores')
    # plt.title('Scores by group and gender')
    # plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    # plt.yticks(np.arange(0, 81, 10))
    # plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    # plt.savefig('test0425')