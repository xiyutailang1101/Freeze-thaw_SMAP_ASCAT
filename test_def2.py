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
# import data_process
# import os, re

# test_def.plot_ref('test_win_ref', [0, 1])

# test_def.read_h5(['2016.11.23', '2016.12.20'])
site_nos = ['1177', '1233', '947', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '2081', '2210', '1089',  '2212', '2211']
Read_radar.radar_read_main('_A_', site_nos, ['2015.04.01', '2016.12.24'], 'vv')
sys.exit()
tbv0, tbh0, npr0, gau0, ons0 = test_def.main('2213', ['20160101', '20161225'], sm_wind=7, mode='annual', seriestype='tb')
print ons0
status = 0
# test_def.trans_peroid(99, -99, 'buff')
# test_def.trans_peroid(1, -1, 'trans')
# plot_test.date_demo()
# test_def.read_alaska('2016.12.02', 18)
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



