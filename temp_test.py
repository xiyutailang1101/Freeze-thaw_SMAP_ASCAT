__author__ = 'xiyu'
import numpy as np
import basic_xiyu as bxy
import matplotlib.pyplot as plt
import plot_funcs
import glob

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
    fname = 'ascat_20160110_0528_9001_value.npy'
    fnamec = 'ascat_20160110_0528_9001_corner.npy'
    test0 = np.load(directory0+fname)
    test1 = test0[0, :, :]
    test_xy = test1[test1[:, -1] == 1074200, 0:2]
    print test0.shape
    print test_xy
    testc = np.load(directory0+fnamec)
    testc1 = testc[0, :, :]
    testc_xy = testc1[testc1[:, -1] == 1074200]
    print testc_xy

if __name__ == "__main__":
    check_9001()
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