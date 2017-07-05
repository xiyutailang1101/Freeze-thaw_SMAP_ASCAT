import spt_quick
import os, re
import data_process
import numpy as np
import test_def
import sys
# get the date string list
# data_process.ascat_onset_map('./result_05_01/onset_result/ascat_onset_3_2016.npy', ft='0')
    # 4: -151.47462, 62.65426, 107, 204
    # 3. -152.05521, 61.62864, 98, 202
    # -153.38525605, 60.80840073, 89, 204
    # -153.48717832, 62.25746204, 100, 210 area_3 03
    # 89, -151.20377421, 60.73384959, 93, 195 area03 04
    # 61, -150.67175906, 60.04628798, 89, 190 area03 05
    # 154, 167, -148.24980309, 61.72569201, 107, 188 area04 01
    # 65, -148.41475430, 61.36947935, 104, 187 area04 02
    # 83, -145.91968862, 64.71503588, 134, 194 area04 03
    # 65, -153.50964993, 70.70806989, 166, 243 area05 01
    # 146, -153.06530376, 70.50435315, 165, 241 area05 02
    # 103, -163.28627617, 68.65911367, 138, 265 area06 01
    # 113, -162.24943920, 69.13594698, 143, 263 area06 02
    # 76, -156.34578665, 67.67279153, 138, 240 area07 01
    # 67, -157.73233549, 68.15037569, 140, 246 area07 02
    # 144, -157.93985569, 69.16732504, 148, 250 area07 03
    # 142, -156.28640511, 70.27669083, 159, 249 area08 01
    # 133, -156.39981013, 71.03811102, 165, 252 area08 02
    # 149, -147.02034702, 68.58702006, 160, 216 area09 01
    # 165, -147.55250323, 69.12119637, 163, 220 area 09 02
    # 179, -143.77940611, 69.13717664, 170, 210 area 10 01
    # 136, -143.60888602, 69.79829215, 175, 213 area 10 02


# spt_quick.build_mask()

# normalized
# data_process.ascat_plot_series()
data_process.smap_alaska_onset(sig=4)
data_process.smap_alaska_onset(mode='npr', sig=4)

data_process.ascat_onset_map(['AS', 'DES'], product='smap', odd_point=[-157.05106272, 70.47259330], mask=True)
for m in ['area_8']:
    # 346, -159.98123961, 67.72333190, 45, 66
    # 319, -158.42869281, 67.48967273, 45, 64
    # 127, -159.82054134, 69.79921252, 51, 68
    # 126, -155.55604522, 68.44360398, 49, 62
    # 129, -157.05106272, 70.47259330, 54, 66
    print 'test series of pixel in %s' % m
    for ob in ['AS']:
        for mo in ['npr']:
            data_process.smap_result_test(m, key=ob, odd_rc=(54, 66), mode=mo, ft='0')

#  calling def function()
def call_data_process():
    data_process.ascat_plot_series()
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
                data_process.smap_result_test(m, key=ob, odd_rc=(47, 45), mode=mo, ft='1')

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

    ##
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

    # smap mask, calculate onset
    spt_quick.smap_mask()
    data_process.smap_alaska_onset()