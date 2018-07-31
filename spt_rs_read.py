__author__ = 'xiyu'
import test_def, peakdetect
import csv
import numpy as np
import read_site
import sys
# import plot_test
# import site_infos
import Read_radar
# import h5py
import data_process
import os
import site_infos
# import os, re

# test_def.plot_ref('test_win_ref', [0, 1])

# test_def.read_h5(['2016.11.23', '2016.12.20'])

# sys.exit()
# tbv0, tbh0, npr0, gau0, ons0 = test_def.main('2213', ['20160101', '20161225'], sm_wind=7, mode='annual', seriestype='tb')
# print ons0
# status = 0


# save daily smap data in forms of txt
# 'smap_tb': ['cell_tb_v_aft', 'cell_tb_qual_flag_v_aft', 'cell_tb_error_v_aft',
#                      'cell_tb_h_aft', 'cell_tb_qual_flag_h_aft', 'cell_tb_error_h_aft',
#                      'cell_boresight_incidence_aft', 'cell_tb_time_seconds_aft',
#                      'cell_tb_v_fore', 'cell_tb_qual_flag_v_fore', 'cell_tb_error_v_fore',
#                      'cell_tb_h_fore', 'cell_tb_qual_flag_h_fore', 'cell_tb_error_h_fore',
#                      'cell_boresight_incidence_fore', 'cell_tb_time_seconds_fore']}
site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
orb = '_A_'
for site_no in site_nos:
    full_path = 'tp/temp_timeseries_0730/tb_'+site_no+orb+'2016.txt'
    Read_radar.read_tb2txt(site_no, orb, fname=full_path, attribute_name='smap_ta_lonlat_colrow',
                                   year_type='water', is_inter=True)
sys.exit()
site_nos_new = ['957', '958', '963', '2080', '947', '949', '950',
                '960', '962', '967', '968', '1090', '1175', '1177', '1233',
                '2065', '2081', '2210', '2211', '2212', '2213']
site_nos_new = ['957', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'ns6', 'ns7']
site_nos_new = ['957', '9001', '9002', '9003', '9004', '9005', '9006', '9007']
# read from downloaded L1C data
site_nos_new = ['9001', '9002', '9003', '9004', '9005', '9006', '9007',
                '948', '958', '963', '2080', '947', '949', '950',
                '960', '962', '967', '968', '1090', '1175', '1177', '1233',
                '2065', '2081', '2210', '2211', '2212', '2213'
                ]  # no '957'
site_nos_new = ['9001', '9002', '9003', '9004', '9005', '9006', '9007',
                '948', '958', '963', '2080',
                '947', '949', '950',
                '960', '962', '967', '968', '1090', '1175', '1177', '1233',
                '2065', '2081', '2210', '2211', '2212', '2213'
                ]  # no '957'
# site_nos_new = ['1233', '2065']
# Read_radar.radar_read_main('_A_', site_nos_new, ['2016.06.02', '2016.12.31'], 'vv')
# Read_radar.radar_read_main('_D_', site_nos_new, ['2016.06.02', '2016.12.31'], 'vv')
doy0 = range(10, 251)
for doy in doy0:
    Read_radar.getascat(site_nos_new, doy)
bcomand = "sh result_08_01/point/ascat/cpdata.sh"
# os.system(bcomand)
sys.exit()
for site_no in site_nos_new:

    for orb in ['_A_', '_D_']:
        full_path = './result_08_01/point/smap_pixel/time_series/tb_'+site_no+orb+'2016'
        Read_radar.read_tb2txt(site_no, orb, fname=full_path, attribute_name='smap_ta_lonlat_colrow',
                               year_type='water', is_inter=True, ipt_path='_08_01')
        print '%s-%s has been all extracted' % (site_no, orb)

sys.exit()

site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
doy0 = range(60, 150)
for doy in doy0:
    Read_radar.getascat(site_nos, doy)
sys.exit()
doy0 = range(1, 11)
for doy in doy0:
    Read_radar.read_ascat_alaska(doy, year0=2016)
sys.exit()
site_nos = ['947', '949', '950', '960', '962', '967', '968', '1089', '1090', '1175', '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
# site_nos = ['947']
for site_no in site_nos:
        orb = '_A_'
        full_path = './result_07_01/txtfiles/SMAP_pixel_series/tb_'+site_no+orb+'2016'
        # Read_radar.read_tb2txt(site_no, orb, fname=full_path, attribute_name='smap_tb_lonlat')
        Read_radar.read_tb2txt(site_no, orb, fname=full_path, attribute_name='smap_ta_lonlat_colrow',
                               year_type='water', prefix='result_07_01/txtfiles/site_tb/multipixels/', is_inter=False)


    # data_process.plot_tbtxt(site_no, orb, full_path, ['cell_tb_v_aft', 'cell_tb_error_v_aft',
    #                      'cell_tb_h_aft', 'cell_tb_error_h_aft',
    #                      'cell_boresight_incidence_aft', 'cell_tb_time_seconds_aft'])

# save daily pass ascat into time series file in .npy form

def process_new():
    test_def.read_alaska('2015.10.01', '2017.03.01')
    sys.exit()



def read_2tb():
    for site_no in site_nos:
        orb = '_A_'
        full_path = './result_07_01/txtfiles/site_tb/tb_'+site_no+orb+'2016.txt'
        # Read_radar.read_tb2txt(site_no, orb, fname=full_path, attribute_name='smap_tb_lonlat')
        Read_radar.read_tb2txt(site_no, orb, fname=full_path, attribute_name='smap_tb_lonlat', year_type='water')

    # read from downloaded L1C data
    Read_radar.radar_read_main('_D_', site_nos, ['2016.12.24', '2017.03.01'], 'vv')

    # read ascat 12.5km of alaska
    doy0 = range(731, 790)
    for doy in doy0:
        Read_radar.read_ascat_alaska(doy-365)
    sys.exit()

    test_def.read_alaska('2016.12.02', 18)


# test_def.trans_peroid(99, -99, 'buff')
# test_def.trans_peroid(1, -1, 'trans')
# plot_test.date_demo()

# tick_site = ['947', '949', '950', '960', '962', '1090', '967', '968',
#             '1175','2081', '2213', '2210', '2065', '1177']
# loc = []
# for site_no in tick_site:
#     x = site_infos.change_site(site_no)[2]
#     y = site_infos.change_site(site_no)[1]
#     loc.append((x, y, 12, 0, 1, 'BR', site_no))
#
# loc_np = np.array(loc)
# np.savetxt('loc_xy'+'.txt', loc_np, delimiter=' ', fmt='%s')

#data_process.h5_write('test12.h5', ['gp1'], [2.0])
# site_no = '947'
# Read_radar.readradar()
# h5_1177_list, d_list = read_site.get_h5_list('20160101', '20161225', '1177', '_A_', excep='20160915')
# count = 0
# datez = -1
# for h5_1177 in h5_1177_list:
#     datez += 1
#     h1 = h5py.File('result0901/s1177/'+h5_1177, 'r')
#     h2 = h1['North_Polar_Projection']
#     tbv_tp, tbh_tp = [], []
#     if count < 1:
#         # the initiation condition: no -9999 missing data
#         if h2['tb_cell_lat'].value.size > 1:
#             count += 1
#             loc = site_infos.change_site('1177')
#             dis2 = (h2['tb_cell_lat'].value - loc[1])**2 + (h2['tb_cell_lon'].value-loc[2])**2
#             dis_order = np.argsort(dis2)
#             TBv = np.zeros([len(h5_1177_list), h2['tb_cell_lat'].size])
#             TBh = np.zeros([len(h5_1177_list), h2['tb_cell_lat'].size])
#         else:
#             continue
#     if h2['tb_cell_lat'].value.size > 1:  # data is effective
#         print h5_1177
#         for orderi in dis_order:
#             tbv_tp.append(h2['cell_tb_v_aft'][orderi])
#             tbh_tp.append(h2['cell_tb_h_aft'][orderi])
#         TBv[datez] = np.array(tbv_tp)
#         TBh[datez] = np.array(tbh_tp)
# doy = read_site.days_2015(d_list) - 365
# var0, var1 = np.transpose(TBv), np.transpose(TBh)
# var = [[var0[i], var1[i]] for i in range(0, 6)]
# test_def.spatial_plot(doy, var[0:5])
# status = 1



