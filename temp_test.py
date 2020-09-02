__author__ = 'xiyu'
import numpy as np
import basic_xiyu as bxy
import matplotlib.pyplot as plt
import plot_funcs
import glob
import re
import h5py
import spt_quick
import data_process
from scipy import signal
import site_infos
import peakdetect
from matplotlib.legend_handler import HandlerLine2D
import re
import read_site
from multiprocessing import Pool


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

def check_ascat_h5():
    path = 'result_08_01/ascat_resample_all/ascat_20160224_14_A.h5'
    h0 = h5py.File(path, 'r')
    print 'all keys', h0.keys()
    print 'the attribute ', h0.keys()[0]
    print h0[h0.keys()[0]].shape
    print 'the latitude shape is', h0['latitude'].shape
    value0 = h0['sigma'].value
    value_ma = np.ma.masked_array(value0, mask=[value0==0])
    data_process.pass_zone_plot(h0['longitude'].value, h0['latitude'].value, value_ma, './result_08_01/', fname='test_ascat_h5',
                                z_max=-5, z_min=-30, prj='aea',title_str='test01', odd_points=np.array([-170, 60]))


def check_amsr2_h5():
    h0 = h5py.File('result_08_01/area/amsr2/AMSR2_l2r_20160601_north_A.h5')


def check_time_using():
    start0 = bxy.get_time_now()
    # print("----fist part: %s seconds ---" % (dtime2.now()-start0))
    ascat_h0 = h5py.File('result_08_01/area/combine_result/ascat_2016_3d_all.h5')
    ascat_sigma = ascat_h0['sigma'].value.copy()
    ascat_incidence = ascat_h0['incidence'].value.copy()
    ascat_pass_utc = ascat_h0['pass_utc'].value.copy()
    ascat_lat = ascat_h0['latitude'].value.copy()
    ascat_lon = ascat_h0['longitude'].value.copy()
    ascat_h0.close()
    start1 = bxy.get_time_now()
    print("----fist part: %s seconds ---" % (start1-start0))

def check_north_amsr2(gridname='north'):
    if gridname == 'north':
        grid_lon = np.load('/home/xiyu/Data/easegrid2/ease_alaska_north_lon.npy')
        grid_lat = np.load('/home/xiyu/Data/easegrid2/ease_alaska_north_lat.npy')
    else:
        h_t = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151121.h5')
        grid_lon = h_t['cell_lon'].value
        grid_lat = h_t['cell_lat'].value
    date_str=['0701', '0702', '0703', '0704', '0705', '0706']
    orbit = ['A', 'D']
    for d0 in date_str:
        h5_name_a = 'result_08_01/area/amsr2_resample/AMSR2_l2r_2016%s_%s_A_%s.h5' % (d0, gridname, gridname)
        h5_name_d = 'result_08_01/area/amsr2_resample/AMSR2_l2r_2016%s_%s_D_%s.h5' % (d0, gridname, gridname)
        for h5_name, ob_name in zip([h5_name_a, h5_name_d], orbit):
            h0 = h5py.File(h5_name)
            # print gridname
            value0 = h0['Brightness Temperature (res23,18.7GHz,H)'].value*0.01
            # value0 *= 0.01
            print 'the value have size: ', value0.shape
            value_ma = np.ma.masked_array(value0, mask=[value0==0])
            fname0 = 'test_amsr2_h5_resample_%s_%s_%s' % (d0, gridname, ob_name)
            title0 = 'amsr2_%s_%s_%s' % (d0, gridname, ob_name)
            data_process.pass_zone_plot(grid_lon, grid_lat, value_ma, './result_08_01/', fname=fname0,
                                z_max=270, z_min=300, prj='aea',title_str=title0, odd_points=np.array([-170, 60]))
    # with open('Amsr2_key.txt', 'a') as f0:
    #     for key0 in h0.keys():
    #         print key0
    #         f0.write('%s, %d\n' % (key0, h0[key0].value.shape[0]))
    h0.close()


def check_h5_info():
    h0 = h5py.File('amsr2_test_h5.h5')
    pause = 0


def append_2_list(test_append=[]):
    print 'test_append is: ', test_append
    test_append.append(-1), test_append.append(-2)


def check_smap_overpass(check_a_pass, check_d_pass):
    f0 = plt.figure()
    ax0 = f0.add_subplot(111)
    x0 = np.arange(0, check_a_pass.size)
    ax0.plot(x0, check_a_pass, 'ks')
    ax0.plot(x0, check_d_pass, 'ko')
    ax0.plot(x0, check_a_pass-check_d_pass, 'rs')
    plt.savefig('result_08_01/smap_overpass_check.png')

def check_2018_11_24():
    return 0
    # for pass_time0 in t_ascat_9.T:
    #             check0 = np.isnan(pass_time0)
    #             check0 = (~check0) & (pass_time0>0)
    #             if sum(check0) > 1:
    #                 tup_9 = bxy.time_getlocaltime(pass_time0[check0], ref_time=[2000, 1, 1, 0])
    #                 print tup_9
    #                 check_9 = 1
def check_ascat_grid():
    p0 = 'result_08_01/ascat_resample_all/ascat_metopB_20160101_13_A.h5'
    h0 = h5py.File(p0)
    for key0 in h0.keys():
        print key0, h0[key0].value.shape
    x=0

def check_ak_ascat():
    fname = 'ascat_20160423_metopB_alaska.npy'
    p0 = 'result_08_01/area/ascat/'
    check0 = np.load(p0+fname)
    print fname
    print 'the shape is ', check0.shape
    print 'the 1st row is ', check0[0]


def check_ad_ascat():
    check_all = np.load('test_pixel_mean.npy')
    inc0, sigma0 = np.load('check_inc0.npy'), np.load('check_sigma0.npy')
    a_doy = np.load('test_asc.npy')
    d_doy = np.load('test_des.npy')
    same_doy, ia, id = np.intersect1d(a_doy[0], d_doy[0], return_indices=True)
    same_doy2, ia2, id2 = np.intersect1d(np.flip(a_doy[0]), np.flip(d_doy[0]), return_indices=True)
    ia_last, id_last = -ia2 + np.flip(a_doy[0]).size, -id2 + np.flip(d_doy[0]).size
    i_check = 0
    for i0, same_doy0 in enumerate(same_doy):
        indice0_org = np.where(a_doy[0] == same_doy0)[0]  # serach by value in original array1
        indice0_new = np.arange(ia[i0], ia_last[i0])
        print 'the index of same doy are:', indice0_org, indice0_new
        if all(indice0_org == indice0_new):
            if i0>0 & i0<same_doy.size-1:
                print 'the same doys are: ', a_doy[0][indice0_new]
                print 'the previous one:', a_doy[0][indice0_new[0]-1]
                print 'the next one: ', a_doy[0][indice0_new[-1]+1]
            else:
                continue
        else:
            print 'the calculated index are wrong'
            i_check = -1
    print i_check
    return 0


def check_pass_hr(d_str):
    match0 = 'result_08_01/ascat_resample_all/*%d*' % d_str
    file_list = glob.glob(match0)
    for f0 in file_list:
        print f0
        h0 = h5py.File(f0)
        secs = h0['pass_utc'].value.copy().ravel()
        tup = bxy.time_getlocaltime(secs[secs>0], ref_time=[2000, 1, 1, 0])
        pause=0


def time_cal(str):
    secs0 = bxy.get_total_sec(str)
    secs_tup = bxy.time_getlocaltime([secs0], ref_time=[2000, 1, 1, 0])
    print str
    print secs_tup


def plot_ascat_ad_difference(id=3770):
    array0 = np.load('test_ascat_obd_36_%d.npy' % (id))
    sz = array0.shape[0]
    i = 9
    pass_hra, pass_hrd = bxy.time_getlocaltime(array0[0]), bxy.time_getlocaltime(array0[sz/2])
    index = (pass_hrd[0] == 2016) & (pass_hrd[-2] > 0) & (pass_hrd[-2] < 100) & (array0[4] < -13.5)
    index_a = (array0[i+1+10] > 20) & (array0[i+1+10] < 150)
    index_d = (array0[i+1+31] > 20) & (array0[i+1+31] < 150)
    i = 0  # closest to far, to mean (no. 9)
    plot_funcs.plot_subplot([[array0[0][index_a], array0[1+i][index_a]],
                             [array0[sz/2][index_d], array0[sz/2+i+1][index_d]],
                             [array0[0], array0[i+1] - array0[sz/2+1+i]],
                             [array0[0], array0[2]],
                             [array0[0], pass_hrd[-1]]
                             ],
                            [],
                            # red_dots = [array0[3][index], array0[4][index], 1],
                            main_label=['asc', 'des', 'asc - des', 'a_pass', 'd_pass'], x_unit='secs',
                            figname='test_tbd_%d.png' % (id))



def plot_ascat_ad_dict(id=3770, i=0):
    ascat_series_A = np.load('ascat_asc_%d.npz' % id)
    ascat_series_D = np.load('ascat_des_%d.npz' % id)
    # utc_line_nodes_0, sigma0_trip_aft_0, inc_angle_trip_aft_0
    xlim = bxy.get_total_sec('20160101')
    plot_funcs.plot_subplot([[ascat_series_A['utc_line_nodes_%d' % i], ascat_series_A['sigma0_trip_aft_%d' % i]],
                             [ascat_series_D['utc_line_nodes_%d' % i], ascat_series_D['sigma0_trip_aft_%d' % i]],
                             [ascat_series_A['utc_line_nodes_%d' % i],
                              ascat_series_A['sigma0_trip_aft_%d' % i] - ascat_series_D['sigma0_trip_aft_%d' % i]],
                             # [array0[0], array0[2]],
                             # [array0[0], pass_hrd[-1]]
                            ],
                            [],
                            y_lim=[[0, 1], [[-20, -8], [-20, -8]]],
                            x_lim=[bxy.get_total_sec('20160101'), bxy.get_total_sec('20161231')],
                            # red_dots = [array0[3][index], array0[4][index], 1],
                            main_label=['asc', 'des', 'asc - des', 'a_pass', 'd_pass'], x_unit='sec',
                            figname='result_txt/test_tbd_%d_%d.png' % (id, i))


def plot_pixel(id=3770, i=0):
    ascat_series_A = np.load('ascat_asc_%d.npz' % id)
    ascat_series_D = np.load('ascat_des_%d.npz' % id)
    # utc_line_nodes_0, sigma0_trip_aft_0, inc_angle_trip_aft_0
    xlim = bxy.get_total_sec('20160101')
    plot_funcs.plot_subplot([[ascat_series_A['utc_line_nodes_%d' % i], ascat_series_A['sigma0_trip_aft_%d' % i]],
                             [ascat_series_D['utc_line_nodes_%d' % i], ascat_series_D['sigma0_trip_aft_%d' % i]],
                             [ascat_series_A['utc_line_nodes_%d' % i],
                              ascat_series_A['sigma0_trip_aft_%d' % i] - ascat_series_D['sigma0_trip_aft_%d' % i]],
                             # [array0[0], array0[2]],
                             # [array0[0], pass_hrd[-1]]
                            ],
                            [],
                            y_lim=[[0, 1], [[-20, -8], [-20, -8]]],
                            x_lim=[bxy.get_total_sec('20160101'), bxy.get_total_sec('20161231')],
                            # red_dots = [array0[3][index], array0[4][index], 1],
                            main_label=['asc', 'des', 'asc - des', 'a_pass', 'd_pass'], x_unit='sec',
                            figname='result_txt/test_tbd_%d_%d.png' % (id, i))


def use_ascat_npz(id=3770):
    ascat_series = np.load('ascat_asc_%d.npz' % id)
    for f0 in sorted(ascat_series.files):
        print f0
    print 'aft_3 has shape 0f ', ascat_series['inc_angle_trip_aft_3'].shape

    return 0


def check_ascat_grid_utc_line_nodes():
    ps = ['result_08_01/ascat_resample_all/ascat_metopA_20160203_12_A.h5',
        'result_08_01/ascat_resample_all/ascat_metopB_20160203_21_D.h5']
    ps2 = ['result_08_01/ascat_resample_all/ascat_metopA_20160203_10_A.h5',
        'result_08_01/ascat_resample_all/ascat_metopA_20160203_12_A.h5',
            'result_08_01/ascat_resample_all/ascat_metopB_20160203_18_D.h5']
    p = 'result_08_01/ascat_resample_all/ascat_metopB_20160130_11_A.h5'
    ps3 = ['result_08_01/ascat_resample_all/ascat_metopB_20160214_12_A.h5',
            'result_08_01/ascat_resample_all/ascat_metopB_20160215_10_A.h5',
            'result_08_01/ascat_resample_all/ascat_metopB_20160215_19_D.h5']
    ps4 = ['/home/xiyu/PycharmProjects/R3/result_08_01/ascat_resample_all_copy/ascat_metopB_20170223_22_D.h5',
        'result_08_01/ascat_resample_all/ascat_metopA_20160130_10_A.h5',
            'result_08_01/ascat_resample_all/ascat_metopA_20160130_12_A.h5',
            'result_08_01/ascat_resample_all/ascat_metopB_20160130_20_D.h5',
            'result_08_01/ascat_resample_all/ascat_metopB_20160130_21_D.h5']
    for fname in ps4:
        print fname
        h0 = h5py.File(fname)
        for key0 in h0.keys():
            print key0
        array0 = h0['utc_line_nodes'].value
        tup = 0
        print array0[(137, 137), (200, 200)]
        tup = bxy.time_getlocaltime(array0[(137, 137), (200, 200)], ref_time=[2000, 1, 1, 0])
        print tup


def check_x_value():
    x0 = np.load('x.npy')
    pause = 0
    xy = np.load('x2_check.npy')
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.plot(xy[0], xy[1], 'k.')
    plt.savefig('2562_x_check.png')


def drawing_parameter(ids):
    table_write = 0
    tables = np.zeros([ids.shape[1], 8]) - 999
    for id0, id_name in zip(ids[0], ids[1]):
        npy_list = []
        for file in ['main1', 'main2', 'main3', 'sec1', 'sec2', 'sec3', 'vline', 'lim', 'maintb0', 'maintb2']:
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
        text_example = 'winter: %.3f $\pm$ %.3f, edge: %.3f, level: %.3f' \
                       % (sigma_melt_t1[3], sigma_melt_t1[4], sigma_melt_t1[2], sigma_melt_t1[1])
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
        # np.where(conv_tb_a[0] > tb_a_onset2)  # index after
        tb_a_t3_0_cross = conv_tb_a[0][(conv_tb_a[0]>tb_a_onset2) & (conv_tb_a[1] > -0.5)][0]
        conv_tb_d, tb_d_onset2 = \
            data_process.get_onset(npy_list[9][0], npy_list[9][1],
                                   thaw_window=[bxy.get_total_sec('20160101', reftime=[2000, 1, 1, 12]) +
                                   doy0*3600*24 for doy0 in [60, 150]],
                                   k=k0, type='tb')  # sigma_0
        # plot_funcs.plot_subplot([npy_list[8],
        #                      npy_list[9],
        #                      npy_list[2]],
        #                     [conv_tb_a, conv_tb_d, conv_sigma],
        #                     main_label=['NPR PM ($10^{-2})$', 'NPR AM ($10^{-2})$', '$\sigma^0$ (dB)'],
        #                     vline=[[tb_a_onset2, tb_a_t3_0_cross, sup_onset2],
        #                            ['k-', 'b-', 'r-'], ['tb_down', 'tb_min', 'sig_up']],  # tb down, tb min, sigma up
        #                     x_unit='sec', x_lim=npy_list[7],
        #                     figname='result_agu/result_2019/t3/20181202plot_ascat_%d_%d_tb.png' % (id_name, id0))
        # add a line to a table
        # with open('result_agu/result_2019/station_results.txt') as file0:
        # if table_write == 0:
        #     time0 = bxy.time_getlocaltime(bxy.get_time_now())
        #     heads = '###### Table generated on %s #####' % (time0.strftime("%B %d, %Y"))
        tables[table_write, 0] = id_name
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


def test_legend():
    x = np.linspace(-3, 3)
    y = np.sin(x)
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(1, 1, 1)
    line0 = ax0.plot(x, y, label='wth')
    v_obj = ax0.axvline(x=0.5, ls='--', color='b', label='x0')
    v_obj2 = ax0.axvline(x = 1,  ls=':', color='b', label='x0')
    # plt.legend((v_obj, line0), ('test', 'test2'),
    #             bbox_to_anchor=[0., 1.02, 1., .102], loc=3, ncol=2, mode='expand', borderaxespad=0.,
    #             prop={'size': 12})
    plt.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=plot_funcs.update_prop)})
    plt.savefig('test_legend.png')


def save_as_latex_table():
    # with open('result_agu/result_2019/table_result.txt', 'r') as file0:
    #     for row in file0:
    #         line = re.sub()
    #         pause = 0
    tbs = np.loadtxt('result_agu/result_2019/table_result.txt', delimiter=',')
    fmt_list = []
    for item0 in tbs[0]:
        fmt_list.append('%d')
    fmt_list[-1] = '%.3f'
    np.savetxt('result_agu/result_2019/latex_table_result.txt', tbs,
               delimiter=' & ', header='station & t1 & t1_b & t2 & t2_b & t3 & t3_b & t1_level', fmt=' & '.join(fmt_list))


def test_npz():
    all_result = np.load('20181104_result.npz')
    for fname in all_result.files:
        print fname


def input_var(x):
    x0 = x.copy()
    x0.shape = 2, -1
    return 0


def tran2doy(txtname, savename):
    txt0 = np.loadtxt(txtname, delimiter=',')
    txt0_copy = txt0.copy()
    for i in [2, 4, 6]:
        txt0_copy[:, i] = bxy.time_getlocaltime(txt0_copy[:, i], ref_time=[2016, 1, 1, 0])[3]
    for i in [1, 3, 5]:
        txt0_copy[:, i] = bxy.time_getlocaltime(txt0_copy[:, i], ref_time=[2016, 1, 1, 12])[3]

    with open(txtname, 'r') as f0:
        for row in f0:
            heads = row
            break
    np.savetxt(savename, txt0_copy, delimiter=',', header=heads, fmt='%.2f')


def smap_download_file_list(year=2017, m=3, d=18):
    delta_d = 1
    str_list = bxy.trans_doy_str(np.arange(77, 214), y=2017, form='%Y.%m.%d')
    with open('smap_folders.txt', 'w') as f0:
        for str0 in str_list:
            f0.write('%s\n' % str0)
    return 0


def check_npy():
    # './result_08_01/area/ascat/ascat_'+datez+'_metopB_alaska.npy'
    fname0 = './result_08_01/area/ascat/ascat_20180101_metopB_alaska.npy'
    x = np.load(fname0)
    return 0


def check_ascat_grid_2018():
    name0 = 'ascat_metopB_20180329_12_A.h5'
    path = '/home/xiyu/PycharmProjects/R3/result_08_01/ascat_resample_all'
    h0 = h5py.File('%s/%s' % (path, name0))
    return 0

def test_new():
    result_a = np.array([[9.68000000e+02,5.79020062e+08, 5.79816557e+08, 5.79793986e+08,
        5.69629885e+08, 5.80922284e+08, 5.78180569e+08, 1.00000000e+00]])
    i_site = 0
    y = 'Soil Moisture Percent -2in (pct)'
    window_t1_sec = [result_a[i_site, 1], result_a[i_site, 2]]
    window_t2_sec = [result_a[i_site, 3], result_a[i_site, 4]]
    window_t1t2_npr_sec = [result_a[i_site, 1], result_a[i_site, 3]]
    window_t1t2_ascat_sec = [result_a[i_site, 2], result_a[i_site, 4]]
    window00 = bxy.time_getlocaltime(window_t1_sec)[-2]
    data_process.get_period_insitu(968, result_a[i_site, 1:-1], y, window=window00)


def rm_out_liers():
    x0 = np.array([1, 2, 3, 4, 20, 3, 3, 3])
    valid_i = bxy.reject_outliers(x0, 3)
    print x0[valid_i]


def check_smap_overpass():
    date_str = []
    sno = 968
    for doy0 in np.arange(1, 365):
        date_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=2016)
        date_str.append(date_str0)
    start0 = bxy.get_time_now()
    smap_dict = data_process.smap_alaska_grid(date_str, ['cell_tb_v_aft', 'cell_tb_h_aft', 'cell_tb_time_seconds_aft'],
                                              'A', 9000)
    points = np.loadtxt("/home/xiyu/PycharmProjects/R3/result_agu/result_2019/points_num.txt", delimiter=',')
    p_id = points[0][points[1] == sno]
    print 'the pixel of interest has ID: ', p_id
    time0 = smap_dict['cell_tb_time_seconds_aft'][int(p_id)]
    t_tuple = bxy.time_getlocaltime(time0)
    doy0 = t_tuple[-2]
    pause = 0

def check_h5_ascat():
    path = '/home/xiyu/PycharmProjects/R3/result_08_01/ascat_resample_all2/ascat_metopA_20170101_12_A.h5'
    h0 = h5py.File(path)
    h0.close()

def check_mode_series(sno=947):
    path = '/home/xiyu/PycharmProjects/R3/result_08_01/plot_data'
    points_info = np.loadtxt('result_agu/result_2019/points_num.txt', delimiter=',').astype(int)
    if sno == 947:
        pixel_no = 4569
    else:
        pixel_no = points_info[0][points_info[1] == sno][0]
    air = np.loadtxt('%s/%d_air_%d.txt' % (path, pixel_no, sno))
    npr = np.loadtxt('%s/%d_npr_%d.txt' % (path, pixel_no, sno))
    sigma_f, sigma_m, sigma_af = np.loadtxt('%s/%d_ascat_f_%d.txt' % (path, pixel_no, sno)), \
                                 np.loadtxt('%s/%d_ascat_m_%d.txt' % (path, pixel_no, sno)), \
                                 np.loadtxt('%s/%d_ascat_f_%d.txt' % (path, pixel_no, sno))
    plot_funcs.plot_subplot([npr,
                             [sigma_f[0], sigma_f[1], sigma_m[0], sigma_m[1], sigma_af[0], sigma_af[1]],
                             air],
                         [[], [], air],
                        figname='result_agu/result_2019/new/%d_estimate_%d.png' % (pixel_no, sno),
                        h_line=[[2], [0], ['--']],
                        x_unit='mmdd')
    return 0

def check_dict():
    dict_npz = np.load('ascat_data0.npz')
    xx = dict_npz.files()
    return 0


def read_smap_series(num0s=[4547]):
    num01 = num0s[0]
    smp = np.load('smap_data0.npz')
    tbv, tbh = smp['cell_tb_v_aft'][num01], smp['cell_tb_h_aft'][num01]
    valid_i0 = (tbv > -90) & (tbv > tbh)
    t_valid = smp['cell_tb_time_seconds_aft'][num01][valid_i0] + 12*3600
    npr_value = (tbv[valid_i0] - tbh[valid_i0])/\
                (tbv[valid_i0] + tbh[valid_i0])
    return np.array([t_valid, npr_value])


def save_ascat_series(p_ids=np.array([3770]), t_window=[0, 210], path0='./result_08_01/series/ascat', y=2016):
    ascat_att0 = ['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes', 'sigma0_trip_fore', 'inc_angle_trip_fore',
                  'sigma0_trip_mid', 'inc_angle_trip_mid']
    # ascat_dict_2016 = data_process.get_ascat_dict_v2(np.arange(t_window[0], t_window[1]), p_ids=p_ids,
    #                                                  ascat_atts=ascat_att0, file_path='ascat_resample_all2')
    doy_array = np.arange(t_window[0], t_window[1])
    file_path='ascat_resample_all2'
    path_ascat = []
    for doy0 in doy_array:
        time_str0 = bxy.doy2date(doy0, fmt='%Y%m%d', year0=y)
        match_name = 'result_08_01/%s/ascat_*%s*.h5' % (file_path, time_str0)
        path_ascat += glob.glob(match_name)
    # read ascat_data into dictionary. each key0 corresponded to the key0 of the ascat h5 file (300, 300, time)
    start0 = bxy.get_time_now()
    ascat_dict = data_process.ascat_alaska_grid_v2(ascat_att0, path_ascat,  pid=p_ids)  # keys contain 'sate_type'
    start1 = bxy.get_time_now()
    print("----read ascat part: %s seconds ---" % (start1-start0))
    data_process.angular_effect(ascat_dict, 'inc_angle_trip_aft', 'sigma0_trip_aft')
    data_process.angular_effect(ascat_dict, 'inc_angle_trip_fore', 'sigma0_trip_fore')
    data_process.angular_effect(ascat_dict, 'inc_angle_trip_mid', 'sigma0_trip_mid')
    np.savez('%s/ascat_data0.npz' % path0, **ascat_dict)


def check_cal_distance():
    lat = np.array([65.60411758, 64.887959])
    lon = np.array([[-164.94568344, -146.61852435]])
    lats = np.array()

def check_npz2():
    nprz2 = np.load('ascat_year_correct_interpolate.npz')
    key_name = nprz2.files
    mask = np.load(('./result_05_01/other_product/mask_ease2_360N.npy'))
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]  # all land id in alaska
    i0 = np.where(land_id == 3770)[0]
    t0 = nprz2['utc_line_nodes'][i0]
    v0 = nprz2['sigma0_trip_mid_40'][i0]
    non_nan = ~np.isnan(t0)
    # check non_nan
    indice_nonnan = np.where(non_nan)
    plot_funcs.plot_subplot([[t0[non_nan], v0[non_nan]], [t0[non_nan], v0[non_nan]]],
                            [[t0[non_nan], v0[non_nan]], [t0[non_nan], v0[non_nan]]],
                            main_label=['sigma0', 'sigma0'], x_unit='doy')
    return 0


def check_distance():
    check_files = np.load('ascat_check.npz')
    data_process.distance_interpolate_v2(check_files['distance'], check_files['inc_angle_trip_mid'],
                                         check_files['sigma0_trip_mid'],
                                         check_files['utc_line_nodes'])

def check_onset():
    ckeck_file = np.load('onset_2017.npz')
    onset = ckeck_file['arr_1']
    check_file2 = np.load('result_agu/result_2019/smap_thaw_obdh_onset.npy')
    return 0


def grid_trans():
    # smap
    h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102'
    h0 = h5py.File(h5_name)
    lons_smap = h0['cell_lon'].value.ravel()
    lats_smap = h0['cell_lat'].value.ravel()
    mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]
    lon_land_s, lat_land_s = lons_smap[land_id], lats_smap[land_id]
    # ascat
    lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                                np.load('./result_05_01/other_product/lat_ease_grid.npy')
    mask_a = np.load('/home/xiyu/PycharmProjects/R3/result_05_01/other_product/mask_ease2_125N.npy')
    land_id_a = np.where(mask_a.ravel() > 0)[0]
    # all pixel in mask0>0 and its corresponded smap pixel
    lon_land_a, lat_land_a = lons_grid[mask_a>0], lats_grid[mask_a>0]
    n12_n36_array = np.zeros([2, lon_land_a.size]) - 1
    min_dis = np.zeros(lon_land_a.size)
    p0 = 0
    for lon0, lat0 in zip(lon_land_a, lat_land_a):
        pause = 0
        dis = bxy.cal_dis(lat0, lon0, lat_land_s, lon_land_s)
        n12_n36_array[0, p0] = land_id_a[p0] # ascat 125 id
        n12_n36_array[1, p0] = land_id[dis.argmin()]  # smap 360 id
        min_dis[p0] = dis.min()
        p0 += 1
    return n12_n36_array


def test_plot():
    # plot alaska 125 grid
    lons_grid, lats_grid = np.load('./result_05_01/other_product/lon_ease_grid.npy'), \
                                np.load('./result_05_01/other_product/lat_ease_grid.npy')
    mask_a = np.load('/home/xiyu/PycharmProjects/R3/result_05_01/other_product/mask_ease2_125N.npy')
    data_process.pass_zone_plot(lons_grid, lats_grid, mask_a, './',
                                fname='mask_0', z_max=180, z_min=50, odd_points=np.array([0, 0]))


def data_process_check0():
    # two series
    negative_edge_snowmelt = min_ascat[(min_ascat[:, 1] > melt_zone0[0]) &
                                       (min_ascat[:, 1] < melt_zone0[1])]
    if negative_edge_snowmelt[:, -1].size < 1:
        if l0 > 2000:
            outlier_count += 1
            print 'no negative edge was found within the window'
            ascat_outlier_index = n125_n360[0][l0].astype(int)
            ascat_outlier_index_360 = n125_n360[1][l0].astype(int)
            smap_outlier_index_360 = np.where(pid_smap==ascat_outlier_index_360)[0][0]
            ascat_outlier_lon, ascat_outlier_lat = n125_n360[2][l0], n125_n360[3][l0]
            with open('two_series_detect_v2_error.txt', 'a') as f0:
                if outlier_count < 2:
                    f0.write('# xxx\n')
                f0.write('year: %d; ' % year0)
                f0.write('doy count: %d; ' % times0.size)
                f0.write('location: %d, %.3f, %.3f\n' % (ascat_outlier_index, ascat_outlier_lon, ascat_outlier_lat))
                f0.write('doy and value: ')
                t0_all = bxy.time_getlocaltime(times0)[-2]
                f0.writelines('%d,' % (t0) for t0 in t0_all)
                f0.write('\n')
                f0.writelines('%.3f,' % (v0) for v0 in sigma0_correct)
                f0.write('\n')
                f0.write('*************************************************************************************\n')
            print 'checking the outlier %d' % ascat_outlier_index
            if ascat_outlier_index in [26305, 30476, 30493, 30512, 30793]:
                pause = 0  # add plotting
                figure_name = 'check_sigma0_%d' % ascat_outlier_index
                smap_outlier = smap_pixel[smap_outlier_index_360]
                npr_plot0 = smap_outlier[0:2]
                npr_conv_plot0 = smap_outlier[2]
                ascat_plot0 = sigma0_pack[0:2]
                ascat_conv_plot0, ascat_conv_plot1 = sigma0_pack[2], sigma0_pack[3]
                print 'outlier %d in %d was plotted' % (ascat_outlier_index, year0)
                plot_funcs.plot_subplot([npr_plot0, ascat_plot0, ascat_plot0],
                                        [npr_conv_plot0, conv_ascat_neg, conv_ascat_pos],
                                        main_label=['npr', '$\sigma^0$ mid', '$\sigma^0$'],
                                        figname=figure_name, x_unit='doy',
                                        h_line=[[-1], [0], [':']],
                                        line_type_main='k-',
                                        # vline=v_line_local, vline_label=doy_all,
                                        # annotation_sigma0=[text_qa_w, text_qa_m],
                                        # y_lim=[[1], [[-18, -4]]]
                                        )
            if outlier_count > 10:
                print '10 outliers have been checked'
                # break


def check_time_series():
    ascat_series = np.load('corrected_sigma0_32654.npy')
    conv_positive, conv_negative =np.load('positive_convolution_32654.npy'), np.load('negative_convolution_32654.npy')
    plot_funcs.plot_subplot([ascat_series, ascat_series], [conv_positive, conv_negative],
                            figname='./test_extracted_32654.png', x_unit='doy', y_lim=[[0, 1], [[-16, -8], [-16, -8]]])
    print 'the lowest value during melt is ', np.min(ascat_series[1][[(ascat_series[0] > 5.12603056e+08) &
                                                                      (ascat_series[0] < 5.17650168e+08)]])
def check_npz():
    npz0 = np.load('onset_outlier_2016.npz')
    print 'pixel in npz file, ', npz0['ascat_pixel'].shape[0]
    array0 = np.loadtxt('map_plot_check_pixel_2016.txt', delimiter=',')
    print  'the total number of outliers', array0.shape[0] + 5
    p = 0
    return p

def assign_number(array0, id, value):
    array0[id] = value


def test_multi(a, b):
    all_args = [(a0, b) for a0 in a]
    print all_args
    return all_args

def mult_wrapper(args):
    return mult_process0(*args)

def mult_process0(i, n):
    t = np.zeros(i)
    m = i*2.5
    n = i*5
    p = i*-1
    # print 'i is %d' % i
    # print 'the n: %d, %d, %d' % (n[0], n[1], n[2])
    return t, m, n, p


def file_name_formated():
    time_prefix = bxy.get_time_now()
    time_array = np.array([time_prefix.month, time_prefix.day, time_prefix.hour, time_prefix.minute])
    time_str_array = [time_array[l0].astype(str) for l0 in range(time_array.size)]
    time_str_array_formated = ['0'+item if len(item) < 2 else item for item in time_str_array]
    print('%s/smap_onset_%s_%s_%s%s%s%s.npz' % ('doc1', 'npz_name1', 'yyyy', time_str_array_formated[0],
                                                time_str_array_formated[1], time_str_array_formated[2],
                                                time_str_array_formated[3]))


def check_pixel_wise():
    array0 = np.load('npy_file/file2316.npy')
    array1 = np.load('npy_file/file2316.npy')
    ratio = array1[0:40]/array0[0: 40]
    percentage = ratio
    f_list = glob.glob('npy_file/file*.npy')
    ascat_array = np.zeros([len(f_list), array0.size])
    for i0, f0 in enumerate(f_list):
        print f0
        ascat_array[i0] = np.load(f0)
    return 0


def check_dual_onset():
    series = np.load('ascat_pixel_npy_2316.npy')
    times0, sigma0_correct = series[0], series[1]
    thaw_window = [bxy.get_total_sec('20170301'), bxy.get_total_sec('20170701')]
    melt_zone0 = [bxy.get_total_sec('20170301'), bxy.get_total_sec('20170701')]
    smap_onset = bxy.get_total_sec('20170401')
    g_max = 30
    edge_count = 10
    g1 = 7
    g2 = 7
    while (g1 < g_max) & (edge_count > 4):
        # update g1 each loop
        edge_out, g1, edge_count = \
            data_process.edge_iteration_v2(times0, sigma0_correct, g1, 3, [melt_zone0[0], thaw_window[1]],
                                           is_negative=0)
        g1 += 1
    max_ascat, min_na, conv_ascat_pos = edge_out[0], edge_out[1], edge_out[2]
    thaw_onset0 = data_process.get_positive_edge(max_ascat, thaw_window, smap_onset)
    melt_zone0[1] = thaw_onset0
    g_short = g1/2
    if g_short > 3:
        g_short = 3
    edge_count = 10
    while (g_short < 10) & (edge_count > 4) & (g_short < g1):
        # using new g1 in the last step, and update g2 each loop
        edge_out, g1, edge_count = \
            data_process.edge_iteration_v2(times0, sigma0_correct, g1, g_short, melt_zone0, is_negative=1)
        g_short += 1
    max_na, min_ascat, conv_ascat_local_min = edge_out[0], edge_out[1], edge_out[2]
    melt_onset0, conv_on_melt_date, lvl_on_melt_date, melt_array, number_onset = \
        data_process.get_negative_edge(min_ascat, 0.2, melt_zone0)
    return 0


def check_dual_onset_from_ascat_h5():
    f_list = bxy.get_yearly_files([143, 144], year0=2017)
    n125_n360 = np.load('n12_n36_array.npy').astype(int)
    ascat_all_id = np.unravel_index(n125_n360[0], (300, 300))
    dict_2016 = data_process.ascat_alaska_grid_v3(['sigma0_trip_aft', 'inc_angle_trip_aft', 'utc_line_nodes',
                                                   'sigma0_trip_fore', 'inc_angle_trip_fore', 'sigma0_trip_mid',
                                                   'inc_angle_trip_mid'], f_list, pid=ascat_all_id)
    # for f0 in f_list():
    #     h0 = h5py.File(f0)
    return 0


def check_dual_onset_0523(f0, f1):
    h0 = h5py.File(f0)
    h1 = h5py.File(f1)
    t0_array, t1_array = h0['utc_line_nodes'].value.ravel(), h1['utc_line_nodes'].value.ravel()
    n125_n360 = np.load('n12_n36_array_4cols.npy')
    ind_ascat_pixel = n125_n360[0].astype(int)
    t0_check = t0_array[ind_ascat_pixel][2317]
    t1_check = t1_array[ind_ascat_pixel][2317]
    # for f0 in f_list():
    #     h0 = h5py.File(f0)
    return t0_check, t1_check


def check_dual_onset_from_ungrided_npy():
    f0_metopA = './result_08_01/area/ascat/ascat_20170523_metopA_alaska.npy'
    f0_metopB = './result_08_01/area/ascat/ascat_20170523_metopB_alaska.npy'
    vA, vB = np.load(f0_metopA), np.load(f0_metopB)
    checkA, checkB = vA[:, 2] , vB[:, 2]
    return 0


def check_error_time():
    # 'result_08_01/ascat_resample_all3/ascat_metopB_20160103_21_D.h5'
    # i0: (array([66, 71]), array([217, 229]))
    h0 = h5py.File('result_08_01/ascat_resample_all3/ascat_metopB_20160103_21_D.h5')
    t0_secs = h0['utc_line_nodes'].value
    t0_tuple = bxy.time_getlocaltime([t0_secs[66, 217], t0_secs[71, 229]], ref_time=[2000, 1, 1, 0])
    npy0 = np.load('./result_08_01/area/ascat/ascat_20170214_metopB_alaska.npy')
    npy_secs = npy0[:, 14]
    npy_t_tuple = bxy.time_getlocaltime(npy_secs, ref_time=[2000, 1, 1, 0])
    npy_t_tuple_utc = bxy.time_getlocaltime(npy_secs, ref_time=[2000, 1, 1, 0], t_out='utc')
    y, m, d, doy, h = npy_t_tuple_utc[0], npy_t_tuple_utc[1], npy_t_tuple_utc[2], npy_t_tuple_utc[3], npy_t_tuple_utc[4]
    for hour in np.unique(h):
        index = h == hour
        d0, doy0, sec0 = d[index], doy[index], npy_secs[index]
        d0_two_set = np.unique(d0)
        d0_0_index, d0_1_index = np.where(d0 == d0_two_set[0])[0][0], np.where(d0 == d0_two_set[1])[0][0]
        r = 0
    return 0

def check_smap_npz():
    check0 = np.load('smap_all_series_A_2016.npz')
    print check0.files
    return 0


def check_new_gridded_h5():
    # ascat_metopB_20170426_7_D.h5
    h5_list = glob.glob('result_08_01/ascat_resample_all3/ascat*2017*.h5')
    for f0 in h5_list:
        # f0 = 'result_08_01/ascat_resample_all3/ascat_metopB_20170320_0_A.h5'
        h0 = h5py.File(f0)
        time_key_on_name = f0.split('/')[-1].split('_')
        utc_line_nodes = h0['utc_line_nodes'].value
        utc_line_nodes_nonzero = utc_line_nodes[utc_line_nodes>0]
        utc_line_tuple = bxy.time_getlocaltime(utc_line_nodes_nonzero, ref_time=[2000, 1, 1, 0], t_out='utc')
        # check doy
        # print 'filename time str and utc line node does not match'
        un_match0 = np.where((utc_line_tuple[3] - bxy.get_doy([time_key_on_name[2]], year0=2017)[0]) != 0)
        un_match1 = np.where((utc_line_tuple[4] - int(time_key_on_name[3])) != 0)
        if un_match0[0].size | un_match1[0].size > 0:
            print f0
            print utc_line_tuple[un_match0], utc_line_tuple[un_match1]
        # check pass hour
        end0 = 0
    h0 = h5py.File('result_08_01/ascat_resample_all3/ascat_metopB_20170118_7_D.h5')
    return 0


def check_inisitu_read():
    in_si_tu_2016 = read_site.in_situ_series(947, in_situ_measure='air', y=2016)
    in_si_tu_2017 = read_site.in_situ_series(947, in_situ_measure='air', y=2017)
    in_si_tu_2018 = read_site.in_situ_series(947, in_situ_measure='air', y=2018)
    np.hstack((in_si_tu_2016[:, :, 1], in_si_tu_2017[:, :, 1], in_si_tu_2018[:, :, 1]))
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
        npr_date = 100*24*3600 + bxy.get_total_sec('%d0101' % year0)
    if ascat_date < 0:
        ascat_date = 100*24*3600 + bxy.get_total_sec('%d0101' % year0)
    snd_name = "snow"
    print 'the %d was processing' % sno0
    sno = str(sno0)
    tair_name = "Air Temperature Observed (degC)"
    if sno0 in [2065, 2081]:
        if year0 == 2016:
            tair_name = "Air Temperature Average (degC)"
    # read measurements
    hr_list = [5, 7, 9, 14, 18, 21]
    t_air_one_year = read_site.in_situ_series(sno, y=year0, hr=hr_list)  # [:, :, 0] temperature at 7:00 (local)
    # time_above_zero_0 = data_process.zero_find(t_air_one_year[:, :, 0], w=10, th=-0.1)  #
    # time_above_zero_1 = data_process.zero_find(t_air_one_year[:, :, 1], w=10, th=-0.1)
    # time_above_zero_2 = data_process.zero_find(t_air_one_year[:, :, 3], w=10, th=-0.1)
    time_above_zero_list = [data_process.zero_find(t_air_one_year[:, :, i], w=10, th=-0.1)
                            for i in range(0, len(hr_list))]
    date_tuple = bxy.time_getlocaltime(time_above_zero_list, ref_time=[2000, 1, 1, 0], t_source='US/Alaska')
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


def test_station_air():
    plot_dict = dict()
    sno0 = 1175
    # ------------------------------------------------------- test ---------------------------------------------------
    # in situ data for 3 years
    plot_dict['air_above_0'], plot_dict['air_temperature'] = [], []
    for year0 in [2016, 2017, 2018]:
        hr_list = [5, 7, 9, 14, 18, 21]
        hr_index = 3
        t_air_one_year = read_site.in_situ_series(sno0, y=year0, hr=hr_list)
        time_above_zero_list = [data_process.zero_find(t_air_one_year[:, :, i], w=10, th=-0.1)
                                for i in range(0, len(hr_list))]
        plot_dict['air_above_0'].append(time_above_zero_list[hr_index])
        plot_dict['air_temperature'].append(t_air_one_year[:, :, hr_index])
    return 0


def test_single_station_result():
    tab = np.loadtxt('combination_detect_results_air_temperature.txt', delimiter=',')
    time_tuple = bxy.time_getlocaltime(tab[0], ref_time=[2000, 1, 1,0], t_source='US/Alaska')
    return time_tuple


def temperature_discrete():
    a = 0
    t_air_one_year = read_site.in_situ_series(1175, y=2016, hr=np.array([0]))
    discrete_temperature = 5 * (t_air_one_year[1].astype(int) / 5)
    return 0


def temporally_saved():
    for yr0 in [2016, 2017, 2018]:  # add 2016, 2018
        # the sigma0 time series of interested ascat pixel
        sigma0_on_station = np.load('prepare_files/npz/ascat/ascat_interest_pixel_series_%d.npz' % (yr0))
        id_array = sigma0_on_station['pixel_id']
        smap_all_pixels = np.load('prepare_files/npz/smap/smap_all_series_A_%d.npz' % (yr0))
        # smap measurements of the interested pixels (id = id_array[1])
        tbv_on_station, tbh_on_station, secs_smap_all_station = smap_all_pixels['cell_tb_v_aft'][id_array[1]], \
                                                          smap_all_pixels['cell_tb_h_aft'][id_array[1]], \
                                                          smap_all_pixels['cell_tb_time_seconds_aft.npy'][id_array[1]]
        series_npr_all_staions = (tbv_on_station-tbh_on_station)/(tbv_on_station+tbh_on_station)
        series_npr_all_staions[tbv_on_station < 100] = -999
        dict_year[str(yr0)] = [sigma0_on_station, series_npr_all_staions, secs_smap_all_station]
    # get results
    results_table = np.zeros([id_array[1].size, 16])
    loop_count = -1
    for i_pixel, ascat_id_1d in enumerate(id_array[0]):
        loop_count += 1
        smap_id_1d = id_array[1][i_pixel]
        if single_check:  # test one particular pixel
            smap_id_1d = site_id_smap[site_id_no == single_check][0]
            ascat_id_1d = site_id_ascat[site_id_no == single_check][0]
            i_pixel = np.where(id_array[0] == ascat_id_1d)[0][0]
            y_limit = [np.array([0, 1, 2]), np.array([[-0.02, 0.08], [-20, -6], snow_limit])]
            y_limit_2 = [[0, 1], [[0, 0.05], [-3, 9]]]
            print 'check the pixel smap id %d, ascat id %d' % (smap_id_1d, id_array[0][i_pixel])
            if loop_count > 0:
                return 0
        else:
            y_limit = [np.array([0, 1, 2]), np.array([[-0.02, 0.08], [-20, -6], snow_limit])]
            y_limit_2 = [[0, 1], [[0, 0.05], [-3, 6]]]

        # find corresponded station id, if applicable
        if ascat_id_1d in site_id_ascat:
            station_id = site_id_no[site_id_ascat==ascat_id_1d][0]
            print "the site is", station_id
            figname = 'ms_pixel_test_%d_%d' % (station_id, id_array[0][i_pixel])
        else:
            station_id = 947  # if no station in this pixel, set default station ID as 947
            figname='ms_pixel_test_%d_%d' % (id_array[0][i_pixel], smap_id_1d)

        # h5 files that saves time series, edge detection output, and in situ measurements
        plot_h5 = h5py.File('prepare_files/npz/station_measurement/station_plot_%d.h5' % station_id, 'a')
        npr0 = plot_h5.create_group('npr')
        ascat0 = plot_h5.create_group('ascat')
        # plot_h5.create_dataset('v_line', ())
        # plot_h5['ascat_conv_melt'], plot_h5['ascat_conv_thaw'] = [], []
        npr_bar = plot_h5.create_group('npr_conv_bar')
        ascat_bar_max = plot_h5.create_group('ascat_conv_thaw_bar')
        ascat_bar_min = plot_h5.create_group('ascat_conv_melt_bar')
        # dictionary save the onset
        plot_h5.create_dataset('npr_onset', (3, )), plot_h5.create_dataset('ascat_onset', (3, ))
        plot_h5.create_dataset('onset0', (3, ))
        plot_h5.create_dataset('onset1', (3, ))
        plot_h5.create_dataset('onset2', (3, ))
        for i0, yr0 in enumerate([2016, 2017, 2018]):  # 2016, 2017, 2018
            # for each pixel, estimate the onset year by year
            sigma0_one_year = dict_year[str(yr0)][0]
            npr_one_year = dict_year[str(yr0)][1]
            npr_secs_one_year = dict_year[str(yr0)][2]
            # saving npr and back scatter
            npr_array = np.array([npr_secs_one_year[i_pixel], npr_one_year[i_pixel]])  # npr t series
            valid_index = bxy.get_valid_index(npr_array, key_id=[0, 1], invalid=[-999])
            npr_save = npr_array[:, valid_index]
            npr0.create_dataset('%d' % yr0, npr_save.shape, data=npr_save)
            ascat_array = np.array([sigma0_one_year['utc_line_nodes'][i_pixel],  # ascat back scatter t seies
                                    sigma0_one_year['sigma0_trip_aft'][i_pixel],
                                    sigma0_one_year['inc_angle_trip_aft'][i_pixel]])

            sigma0_45 = data_process.angular_correct(ascat_array[1], ascat_array[2], ascat_array[0])
            ascat_array[1] = sigma0_45
            # plot check
            index0 = ascat_array[0]>0
            plot_funcs.quick_plot(ascat_array[0][index0], sigma0_45[index0])
            valid_index = bxy.get_valid_index(ascat_array, key_id=[0, 1], invalid=[0])
            ascat_save = ascat_array[:, valid_index]
            ascat0.create_dataset('%d' % yr0, ascat_save.shape, data=ascat_save)

            # detection
            m_zone, su_zone, th_zone, win_zone = data_process.zone_intiation(yr0)
            conv_npr_pixel, thaw_secs_npr, npr_local_max, all_local_min,\
            _ = data_process.smap_melt_initiation(npr_one_year[i_pixel], npr_secs_one_year[i_pixel], win_zone,
                                                  su_zone, yr0,
                                                  gk=smap_g, one_pixel_return=True)
            plot_h5['npr_onset'][i0] = thaw_secs_npr
            npr_bar_data = npr_local_max[:, [1, 2]].T
            npr_bar.create_dataset('%d' % yr0, npr_bar_data.shape, data=npr_bar_data)

            # ascat detection
            ascat_pass_time = sigma0_one_year['utc_line_nodes'][i_pixel]
            ascat_pass_hr = bxy.time_getlocaltime(ascat_pass_time, ref_time=[2000, 1, 1, 0])[-1]
            pass_index = (ascat_pass_time > 0)
            # pass_index = (ascat_pass_time > 0) & (ascat_pass_hr > 12) & (ascat_pass_hr < 20)
            convolution_series, convolution_event = \
                data_process.two_series_sigma_process(0,
                                                      sigma0_one_year['sigma0_trip_aft'][i_pixel][pass_index],
                                                      sigma0_one_year['inc_angle_trip_aft'][i_pixel][pass_index],
                                                      sigma0_one_year['utc_line_nodes'][i_pixel][pass_index],
                                                      thaw_secs_npr, m_zone, th_zone, win_zone, su_zone,
                                                      7, [7, 7, 7], False,
                                                      save_path='prepare_files/npy_ascat_one_station', is_return=True)
            thaw_bar_data = convolution_event[0][:, [1, 2]].T
            ascat_bar_max.create_dataset('%d' % yr0, thaw_bar_data.shape, data=thaw_bar_data)
            melt_bar_data = convolution_event[1][:, [1, 2]].T
            ascat_bar_min.create_dataset('%d' % yr0, melt_bar_data.shape, data=melt_bar_data)
            combine_result = np.load('prepare_files/npy_ascat_one_station/file0.npy')
            plot_h5['onset0'][i0] = thaw_secs_npr
            # re consider the melt events, if the local minimum is significantly different from winter conv std
            melt_events_minimum = combine_result[33:]
            melt_evnets_time = combine_result[25:33]
            conv_std_winter = combine_result[22]
            onset_secs = check_possible_melt(np.array([melt_evnets_time, melt_events_minimum]), conv_std_winter)
            index_events = (melt_events_minimum < -1) & (melt_events_minimum > -999)
            strong_melt_secs = combine_result[25:33][index_events][0] if sum(index_events) > 0 else onset_secs
            if strong_melt_secs == -1:
                strong_melt_secs = combine_result[11]
            plot_h5['onset1'][i0] = strong_melt_secs
            plot_h5['onset2'][i0] = combine_result[14]

        # plotting
        npr_plot = np.hstack((plot_h5['npr/2016'].value, plot_h5['npr/2017'].value, plot_h5['npr/2018'].value))
        ascat_plot = np.hstack((plot_h5['ascat/2016'].value, plot_h5['ascat/2017'].value, plot_h5['ascat/2018'].value))
        # ascat extraction
        ascat_pass_time = ascat_plot[0]
        ascat_pass_time_tuple = bxy.time_getlocaltime(ascat_pass_time, ref_time=[2000, 1, 1, 0])
        ascat_pass_hr, ascat_pass_doy = ascat_pass_time_tuple[-1], ascat_pass_time_tuple[3]
        # pass_index = ~np.isnan(ascat_pass_time)
        # # pass_index = (ascat_pass_time > 0) & (ascat_pass_hr > 12) & (ascat_pass_hr < 24)
        # # pass_index2 = (ascat_pass_time > 0) & ((ascat_pass_doy < 60) | (ascat_pass_doy > 180))
        # # pass_index = pass_index | pass_index2
        # ascat_plot = ascat_plot[:, pass_index]
        vline_secs = np.concatenate((plot_h5['onset0'].value, plot_h5['onset1'].value, plot_h5['onset2'].value))

        # insitu data for 3 years
        air_temperature, snow_data = plot_h5.create_group('air_temperature'), plot_h5.create_group('snow')
        air_onset, snow_onset = plot_h5.create_dataset('air_above_0', (3, ), data=np.zeros(3)), \
                                plot_h5.create_dataset('snow_depth_drop', (3, ), data=np.zeros(3))
        for i0, yr0 in enumerate([2016, 2017, 2018]):
            hr_list = np.array([5, 7, 12, 14, 18, 21])
            hr_index = np.where(hr_list == hr_pass)[0][0]
            swe_one_year, swe_label = read_site.in_situ_series(station_id, y=yr0, hr=np.array([0]),
                                                               in_situ_measure='snow')
            times_one_day = swe_one_year.shape[2]
            swe_2d_array = swe_one_year[:, :, 0]
            valid_index = swe_2d_array[1, :] > -99
            swe_2d_array = swe_2d_array[:, valid_index]
            swe_difference = np.array([swe_2d_array[0, 1:], -1*np.diff(swe_2d_array[1])])
            if swe_difference.size < 1:
                swe_decrease_time = bxy.get_total_sec('%d0301' % yr0)
            else:
                swe_decrease_time = data_process.zero_find(swe_difference, w=3, th=3)
            plot_h5['snow_depth_drop'][i0] = swe_decrease_time
            t_air_one_year, t_air_label = read_site.in_situ_series(station_id, y=yr0, hr=hr_list)
            # swe_value, swe_date = read_site.read_measurements(site_no, 'snow', np.arange(366, 366+365), hr=0)
            # swe_date-=365
            time_above_zero_list = np.array([])
            time_above_zero_list = [data_process.zero_find(t_air_one_year[:, :, i], w=air_window, th=-0.1)
                                    for i in range(0, len(hr_list))]  # set temperature window as 7 days
            if time_above_zero_list[hr_index] == 1:
                time_above_zero_list[hr_index] = bxy.get_total_sec('%d0101' % yr0)
            plot_h5['air_above_0'][i0] = time_above_zero_list[hr_index]
            one_year_air = t_air_one_year[:, :, hr_index]
            air_temperature.create_dataset('%d' % yr0, shape=one_year_air.shape, data=one_year_air)
            one_year_snow = swe_one_year[:, :, 0]
            snow_data.create_dataset('%d' % yr0, shape=one_year_snow.shape, data=one_year_snow)


def check_angular():
    path_pre = 'prepare_files/h5/pixel_check/pixel_plot_'
    h50 = h5py.File('%s%d.h5' % (path_pre, 44623))
    x, y = h50['ascat/2016'].value[2], h50['ascat/2016'].value[1]
    plt.plot(x, y, 'k.')
    plt.savefig('check_angular_fig.png')
    return 0

    # the convolution output not saved in h5 files


def read_new_h5(h5_name):
    # result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20161101.h5
    # result_08_01/ascat_resample_all3/ascat_metopB_20160102_3_D.h5
    h0 = h5py.File(h5_name, 'r')
    # air temperature size 2, 365, 6
    t_air = h0['air_temperature/2016'].value
    air_hour_0 = t_air[0, :, 0]
    air_hour_2 = t_air[0, :, 2]
    air_all_hour = t_air[0, 9, :]
    for a0 in [air_hour_0, air_hour_2, air_all_hour]:
        doy_list = bxy.time_getlocaltime(a0, ref_time=[2000, 1, 1, 0], t_source='US/Alaska')
    for key0 in h0.keys():
        print key0
    for key0 in h0['air_temperature'].keys():
        print key0
    h0.close()
    return 0

def read_a_txt():
    # npz_value = np.load('npz_folder_2020/ascat_onset_all_2016_05091653.npz')
    npz_value2 = np.load('npy_series_file/ascat_outlier_series_2018.npz')
    n12_n36 = np.load('n12_n36_array_4cols.npy')
    l0 = np.where(n12_n36[0] == 44324)
    npy_value = np.load('npy_file/file11388.npy')
    t_air_one_year = read_site.in_situ_series(949, y=2016, hr=np.array([15]))


def read_a_npy(fname):
    p = 1
    np0 = site_infos.get_ind_table()
    npz0 = np.load(fname)

    keys = npz0.keys()
    t0 = npz0['utc_line_nodes'][0]
    t0_val = t0[t0>0]
    t0_tuple = bxy.time_getlocaltime(t0_val, ref_time=[2000, 1, 1, 0])
    print t0_tuple[0]
    return 0


def detect_one():
    year0 = 2018
    n12_n36 = np.load('n12_n36_array_4cols.npy')
    ascat_series = np.loadtxt('check_pixle_9219.txt')
    # two_series_sigma_process(l0, sigma0, inc0, times0,
    #                          smap_onset, melt_zone0, thaw_window, winter_window, summer_window,
    #                          melt_buff=30, gk=[7, 7, 7], angular=False, save_path='npy_file', is_return=False)
    melt_str_window = bxy.initial_window(year0, 'melt')
    summer_str_window = bxy.initial_window(year0, 'summer')
    winter_str_window = bxy.initial_window(year0, 'winter')
    melt_zone, summer_zone, winter_zone = bxy.get_total_sec_array(melt_str_window), \
                                          bxy.get_total_sec_array(summer_str_window), \
                                          bxy.get_total_sec_array(winter_str_window)
    # read smap onset
    mask = np.load(('./result_05_01/other_product/mask_ease2_360N.npy'))
    mask_1d = mask.reshape(1, -1)[0]
    land_id = np.where(mask_1d != 0)[0]
    npz_value = np.load('npz_folder_final_hopefully/smap_onset_all_2018_07141703.npz')
    id_smap = np.where(n12_n36[0] == 44324)
    ind_smap = n12_n36[1][id_smap]
    smap_onset = npz_value['smap_onset'][land_id==ind_smap[0]]
    data_process.two_series_sigma_process(-1, ascat_series[0], ascat_series[1], ascat_series[2], smap_onset[0],
                                          melt_zone, melt_zone, winter_zone, summer_zone, melt_buff=30,
                                          gk=[-1, 7, 7], angular=True, save_path='temp_npy_file', is_return=True)
    # check the ascat time series in 2018 (pixel  44324)
    sigma0_correct, a, b = data_process.angular_correct(ascat_series[0], ascat_series[1], ascat_series[2],
                                                        inc_c=40, coef=True)
    x = bxy.time_getlocaltime(ascat_series[2], ref_time=[2000, 1, 1, 0])
    x_polt = x[3] + x[4]/24.0
    plt.plot(x_polt, sigma0_correct, 'k.')
    plt.savefig('detect_one.png')
    return 0


if __name__ == "__main__":
    # detect_one()
    # read_new_h5('result_08_01/ascat_resample_all3/ascat_metopA_20161101_19_A.h5')
    # read_a_npy('npy_series_file/ascat_outlier_series_2017.npz'
    h99 = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151102.h5')
    n99 = np.load('n12_n36_array_4cols.npy')
    n0, n1 = np.load('npy_series_file/npr_series_2018.npy'), \
             np.load('prepare_files/npz/smap/smap_all_series_D_2018.npz')
    n12_36 = np.load('npy_series_file/npr_series_%d.npy')
    i0 = np.where(n12_36[0] == 48815)
    status = 0
    # read_a_txt()
    # check_angular()
    # read_a_npy()
    # read_new_h5('prepare_files/h5/pixel_check/pixel_plot_%d.h5' % 44623)
    # read_new_h5('prepare_files/h5/station_measurement/station_plot_%d.h5' % 947)
    # read_new_h5('prepare_files/h5/pixel_check/pixel_plot_44623.h5')
    # read_new_h5('prepare_files/h5/station_measurement/station_plot_%d.h5' % 947)
    # temperature_discrete()
    # time_tuple = test_single_station_result()
    # test_station_air()
    # # quick 538358568.0
    # in_situ_tair_snd(1175, year0=2016, npr_date=-1, ascat_date=-1)
    # check_inisitu_read()
    # check_new_gridded_h5()
    # print bxy.time_getlocaltime([538358568], ref_time=[2000, 1, 1, 0], t_out='utc')
    # check_error_time()
    # # check_dual_onset_from_ascat_h5()
    # check_smap_npz()
    # # check_dual_onset()
    # # check_pixel_wise()
    # check_dual_onset_from_ungrided_npy()
    # f_list = ['result_08_01/ascat_resample_all3/ascat_metopA_20170523_12_A.h5',
    #           'result_08_01/ascat_resample_all3/ascat_metopB_20170523_12_A.h5']
    # t_check = check_dual_onset_0523(f_list[0], f_list[1])
    # t_check2 = check_dual_onset_0523('result_08_01/ascat_resample_all2/ascat_metopB_20170523_14_A.h5',
    #                             'result_08_01/ascat_resample_all2/ascat_metopA_20170523_14_A.h5')
    # check_dual_onset_from_ascat_h5()
    # check_dual_onset()
    # check_pixel_wise()
    # file_name_formated()
    # check multiprocess
    # i, n = np.array([0, 1, 2]), np.array([0, 0, 0])
    # all_args = test_multi(i, n)
    # pool0 = Pool(2)
    # v = pool0.map(mult_wrapper, all_args)
    # p = 0
    # print 'Processed i: ', v
    # print 'proceesed n: ', m

    # array0 = np.zeros(3)
    # print array0
    # assign_number(array0, 1, -2)
    # print array0
    # p=0
    # check_time_series()
    # points = site_infos.get_id()
    # station_z_2016 = np.load('onset_%s_%d.npz' % ('interest', 2016))
    # 47 pixels,
    # n12_n36 = grid_trans()
    # check_onset()
    # check_h5_ascat()
    # check_onset()
    # check_distance()
    # read_smap_series()
    # series1 = np.array([-1, -2, 1, -1, 2, 3, 5])
    # out0, out1 =data_process.n_convolve3(series1, 3)
    # test_new()
    # check_ascat_grid_2018()
    # check_npy()
    # smap_download_file_list(year=2017, m=3, d=18)
    # 201902 # 'result_agu/result_2019/table_result_A_doy.txt'
    # tran2doy('result_agu/result_2019/table_result_A_sec.txt',
    #          'result_agu/result_2019/table_result_A_doys.txt')
    # input = np.array([1, 2, 3, 4])
    # print input
    # input_var(input)
    # print input
    # x0 = input_var(input)
    # print input
    # 201901
    # check = -1
    # # drawing_parameter([5356])
    # # test_legend()
    # land_ids = np.loadtxt('result_agu/result_2019/points_num.txt', delimiter=',')
    # drawing_parameter(land_ids)
    # save_as_latex_table()


    # check_ad_ascat()
    # check_ascat_grid_utc_line_nodes()
    # use_ascat_npz()
    # for i0 in range(0, 10):
    #     plot_ascat_ad_dict(id=4646, i=i0)
    # plot_ascat_ad_dict(i=9)
    # plot_ascat_ad_dict(i=0)
    # site_infos.get_ascat_grid_keys()
    # site_infos.get_ascat_grid_keys()
    # check_ascat_grid()
    # plot_ascat_ad_difference(5356)
    # plot_ascat_ad_difference(3770)
    # thaw_npr, thaw_ascat, melt_ascat
    # Tri_onsets = [41817600, 43303822, 42580751]
    # print bxy.time_getlocaltime(Tri_onsets, ref_time=[2015, 1, 1, 0])
    # check_north_amsr2(gridname='alaska')
    # check_h5_info()
    # check_ascat_h5()
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