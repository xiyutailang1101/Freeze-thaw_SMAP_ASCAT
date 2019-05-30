"""
Storage of site No., site info., attribute of h5 files
"""
import glob
import numpy as np
def get_type(id, all_site=False):
    site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968', '1090', '1175', '1177', '948',
                         '957', '958', '963'],
                    'scan_': ['2080', '2081', '2213', '2210', '2065', '2212', '2211', '1233']}
    if all_site is not False:
        snos = [a0 for key0 in site_dic.keys() for a0 in site_dic[key0]]
        # for key0 in site_dic.keys():
        return snos

    for key_site in site_dic.keys():
        if id in site_dic[key_site]:
            site_type = key_site
            return site_type
        else:
            continue
    print 'no site named (%s)' % id


def get_site_dict():

    site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968','1090', '1175', '1177'],
                    'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
    return site_dic


def get_id(id='947', mode='all'):
    site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081',
                    '2210', '2211', '2212', '2213']
                # '2210', '1089', '1233', '2212', '2211', '2213']
    if mode is 'all':
        return site_nos
    elif mode == 'int':
        return np.array(site_nos).astype(int)
    else:
        return [id]


def in_situ_vars(sno=947):
    if sno in [2065, 2081]:
        return ["Air Temperature Average (degC)",
                "Soil Temperature Observed -2in (degC)",
                "Soil Moisture Percent -2in (pct)",
                "Snow"]
    else:
        return ["Air Temperature Observed (degC)",
                "Soil Temperature Observed -2in (degC)",
                "Soil Moisture Percent -2in (pct)",
                "Snow"]


def ascat_site_lim(site_no):
    site_lim = {'947': [-15, -7], '949': [-14, -7], '950': [-14, -7], '960': [-15, -8], '962': [-15, -8], '967': [-13, -8], '968': [-17, -8],
                '1089': [-15, -7], '1090': [-15, -7], '1175': [-15, -8], '1177': [-19, -10],
                '1233': [-17, -9], '2065': [-16, -8], '2081': [-15, -7], '2210': [-16, -8], '2211': [-16, -8], '2212': [-16, -8],
                '2213': [-17, -9]}
    return site_lim[site_no]


def change_site(site_no, names=False):  # the No, lat and long of each site
    if names is False:
        site_list = [['947', 65.12422, -146.73390], ['948', 65.25113, -146.15133], ['949', 65.07833, -145.87067],
                     ['950', 64.85033, -146.20945], ['1090', 65.36710, -146.59200], ['960', 65.48, -145.42],
                     ['962', 66.74500, -150.66750], ['1233', 59.82001, -156.99064], ['2213', 65.40300, -164.71100],
                     ['2081', 64.68582, -148.91130], ['2210', 65.19800, -156.63500], ['2211', 63.63900, -158.03010],
                     ['2212', 66.17700, -151.74200], ['966', 60.72700, -150.47517], ['1175', 67.93333, -162.28333],
                     ['2065', 61.58337, -159.57708], ['967', 62.13333, -150.04167], ['968', 68.61683, -149.30017],
                     ['1062', 59.85972, -151.31500], ['1089', 62.63000, -150.77617], ['1177', 70.26666, -148.56666],
                     ['alaska', 63.25, -151.5], ['region_10', 64.704400, -146.00350], ['region_11', 64.887930, -146.61848],
                     ['region_12', 65.068405, -147.24229], ['region_13', 64.978016, -145.58728], ['region_14', 65.163426, -146.20591],
                     ['region_15', 65.345779, -146.83360], ['region_16', 65.250104, -145.16204], ['region_17', 65.437429, -145.78426],
                     ['region_18', 65.621694, -146.41580], ['95000', 64.551224000000005, -145.93763000000001],
                     ['95001', 64.612862000000007, -146.14027999999999], ['95002', 64.674167999999995, -146.34388999999999],
                     ['95003', 64.642555000000002, -145.80045999999999], ['95004', 64.704400000000007, -146.0035],
                     ['95005', 64.765912, -146.20751999999999], ['95006', 64.733718999999994, -145.66231999999999],
                     ['95007', 64.795771999999999, -145.86574999999999], ['95008', 64.857493000000005, -146.07015999999999],
                     ['221000', 64.88467, -156.47676000000001], ['221001', 64.928477999999998, -156.70383000000001],
                     ['221002', 64.971880999999996, -156.93167], ['221003', 64.985726999999997, -156.3777],
                     ['221004', 65.029700000000005, -156.60550000000001], ['221005', 65.073266000000004, -156.83409],
                     ['221006', 65.086686, -156.27785], ['221007', 65.130825000000002, -156.50638000000001],
                     ['221008', 65.174556999999993, -156.73571000000001],
                     ['958', 67.25, -150.18], ['2080', 63.35, -142.98], ['963', 63.94, -145.4], ['957', 68.130, -149.478],
                     ['9001', 69.16732504, -157.93985569],
                     ['9002', 66.33659274, -156.55042833], ['9003', 68.26540917, -162.97367991],
                     ['9004', 65.2680856, -161.23434328],
                     ['9005', 65.78373252, -150.86761887, ], ['9006', 62.78914008, -156.36380702],
                     ['9007', 68.58702006, -147.02034702], ['north', 70.0, -155.0], ['peninsular', ],
                     ['20000', 33.43, 79.73], ['20001', 33.77, 101.72], ['20002', 33.89, 102.13],
                     ['20003', 32.48, 80.07],
                     ['1183', 69.0+25.0/60, -148.-42.0/60], ['952', 65+6.0/60, -144-56.0/60],
                     ['961', 66+34.0/60, -145-15.0/60]] # 957 redo
        site_no_list = [row[0] for row in site_list]
        siteinfo = site_list[site_no_list.index(site_no)]
        return siteinfo
    elif names is True:
        site_nos = ['947', '949', '950', '960', '962', '967', '968', '1090', '1175', '1177', '1233', '2065', '2081',
                    '2210', '2211', '2212', '2213']
        name_list = ['Little Chena Ridge', 'Monument Creek', 'Munson Ridge', 'Eagle Summit', 'Gobblers Knob',
                     'Susitna Valley High', 'Imnaviat Creek', 'Upper Nome Creek', 'Kelly Station', 'Prudhoe Bay',
                     'Lower Mulchatna', 'Aniak', 'Nenana', 'Hozatka Lake', 'Innoko Camp', 'Kanuti Lake', 'Checkers Creek']
        if site_no in site_nos:
            return name_list[site_nos.index(site_no)]
        else:
            return site_no
    elif len(names) > 5:
        s_info = np.loadtxt(names, delimiter=',')
        return s_info


def is_new_site(site_no='1183'):
    return site_no in ['1183']


def grid_info(grid_name='erq_25'):
    grid_dict = {}
    # grid_dict['erq_25'] = np.array()
    grid_type = np.dtype({'names': ['name', 'r_unit', 'resolution', 'lat_ul', 'lon_ul', 'rows', 'cols'],
                          'formats': ['S32', 'S32', 'f', 'f', 'f', 'i', 'i']})
    grid_structure = np.array([('erq_25', 'degree', 0.25, 90.0, 0.0, 720, 1440),
                               ('pn_25', 'meter', 25000.0, -99, -99, 574, 432),
                               ('EQMA', 'degree', 0.25, 90.0, 0.0, 720, 1440),
                               ('EQMD', 'degree', 0.25, 90.0, 0.0, 720, 1440)],
                              dtype=grid_type)
    return grid_structure[grid_structure['name'] == grid_name]


def site_index(index0):
    index_list = [5356, 3770]
    site_no_list = ['968', '2213']
    return site_no_list[index_list.index(index0)]


def get_siteNo_list():
    site_no_list = ['947', '948', '949', '950', '960', '1090']
    # site_no_list = '1090'
    return site_no_list


def get_data_root(site_no, date='0702'):
    data_root = 'result' + date + '/s' + site_no + '/'
    return data_root


def get_data_path(date='_05_01'):
    path = '/home/xiyu/PycharmProjects/R3/result' + date + '/'
    if date == '_08_01':
        path = '/home/xiyu/PycharmProjects/R3/result' + date + '/point/smap_pixel/'
    return path


def get_attribute(layername='np', sublayer='sig_vv_aft'):
    """
    Return the attribute we need to read.
    :parameter
        <layername>: the projection or radar
        sublayer: A key to represent the attributes list,
        e.g. default, 'sig_vv_aft' returns ['cell_lat', 'cell_lon', 'cell_sigma0_qual_flag_vv', 'cell_sigma0_vv_aft']
    :return: a list that is [the group name (first layer), the attribute list (second layer)]
    """
    layer1 = {'gm': 'Global_Projection', 'np': 'North_Polar_Projection', 'radar': 'Sigma0_Data', 'flag': 'flag',
              'cell_tb_v_aft': 'North_Polar_Projection'}
    # all_tb = [u'cell_tb_h_aft', u'cell_tb_qual_flag_h_aft', u'cell_tb_qual_flag_v_aft',
    #           u'cell_tb_v_aft', u'site_loc', u'tb_cell_lat', u'tb_cell_lon']
    # tbh_aft = ['cell_tb_h_aft',  'tb_cell_lat', 'tb_cell_lon', 'cell_tb_qual_flag_h_aft']
    # tbv_aft = ['cell_tb_v_aft',  'tb_cell_lat', 'tb_cell_lon', 'cell_tb_qual_flag_v_aft']
    # tbv_af = ['cell_tb_v_aft',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_v_aft', 'cell_tb_error_h_aft',
    #         'cell_tb_time_utc_aft', 'cell_boresight_incidence_aft']
    # tbv_fo = ['cell_tb_v_fore',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_v_fore', 'cell_tb_error_h_fore',
    #           'cell_tb_time_utc_fore', 'cell_boresight_incidence_fore']
    # l_sig = ['cell_lat', 'cell_lon', 'cell_sigma0_qual_flag_vv', 'cell_sigma0_vv_aft']
    # h_sig = ['cell_lat', 'cell_lon', 'cell_sigma0_qual_flag_vv', 'cell_sigma0_vv_aft']
    att_dict = \
        {'tbh_aft': ['cell_tb_h_aft',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_h_aft'],
         'tbv_aft': ['cell_tb_v_aft',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_v_aft'],
         'cell_tb_v_aft': ['cell_tb_v_aft',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_v_aft', 'cell_tb_error_v_aft',
                  'cell_tb_time_utc_aft', 'cell_boresight_incidence_aft'],
         'cell_tb_h_aft': ['cell_tb_h_aft',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_h_aft', 'cell_tb_error_h_aft',
                  'cell_tb_time_utc_aft', 'cell_boresight_incidence_aft'],
         'cell_tb_v_fore': ['cell_tb_v_fore',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_v_fore', 'cell_tb_error_h_fore',
                  'cell_tb_time_utc_fore', 'cell_boresight_incidence_fore'],
         'sig_vv_aft': ['cell_lat', 'cell_lon', 'cell_sigma0_qual_flag_vv', 'cell_sigma0_vv_aft'],
         'sig_hh_aft': ['cell_lat', 'cell_lon', 'cell_sigma0_qual_flag_vv', 'cell_sigma0_vv_aft'],

         'smap_tb': ['cell_tb_v_aft', 'cell_tb_qual_flag_v_aft', 'cell_tb_error_v_aft',
                     'cell_tb_h_aft', 'cell_tb_qual_flag_h_aft', 'cell_tb_error_h_aft',
                     'cell_boresight_incidence_aft', 'cell_tb_time_seconds_aft',
                     'cell_tb_v_fore', 'cell_tb_qual_flag_v_fore', 'cell_tb_error_v_fore',
                     'cell_tb_h_fore', 'cell_tb_qual_flag_h_fore', 'cell_tb_error_h_fore',
                     'cell_boresight_incidence_fore', 'cell_tb_time_seconds_fore'],
         'smap_tb_lonlat': ['cell_lon', 'cell_lat',
                     'cell_tb_v_aft', 'cell_tb_qual_flag_v_aft', 'cell_tb_error_v_aft',
                     'cell_tb_h_aft', 'cell_tb_qual_flag_h_aft', 'cell_tb_error_h_aft',
                     'cell_boresight_incidence_aft', 'cell_tb_time_seconds_aft',
                     'cell_tb_v_fore', 'cell_tb_qual_flag_v_fore', 'cell_tb_error_v_fore',
                     'cell_tb_h_fore', 'cell_tb_qual_flag_h_fore', 'cell_tb_error_h_fore',
                     'cell_boresight_incidence_fore', 'cell_tb_time_seconds_fore'],
         'smap_ta_lonlat_colrow': ['cell_lon', 'cell_lat',
                     'cell_tb_v_aft', 'cell_tb_qual_flag_v_aft', 'cell_tb_error_v_aft',
                     'cell_tb_h_aft', 'cell_tb_qual_flag_h_aft', 'cell_tb_error_h_aft',
                     'cell_boresight_incidence_aft', 'cell_tb_time_seconds_aft',
                     'cell_tb_v_fore', 'cell_tb_qual_flag_v_fore', 'cell_tb_error_v_fore',
                     'cell_tb_h_fore', 'cell_tb_qual_flag_h_fore', 'cell_tb_error_h_fore',
                     'cell_boresight_incidence_fore', 'cell_tb_time_seconds_fore', 'cell_row', 'cell_column']}
    att_read = [layer1[layername], att_dict[sublayer]]


    # att_dict = {'sig_vv_aft': att_sig_vv_aft, 'sig_hh_aft': att_sig_hh_aft,
    # if layername == 'sigma':
    #     attributes = ['Sigma0_Data/cell_sigma0_vv_aft', 'Sigma0_Data/cell_lat', 'Sigma0_Data/cell_lon',
    #                   'Sigma0_Data/cell_sigma0_qual_flag_vv']
    # elif layername == 'tb':
    #     attributes = ['Global_Projection/cell_tb_v_aft', 'Global_Projection/tb_cell_lat',
    #                   'Global_Projection/tb_cell_lon', '/none']
    # elif layername == 'tbn':
    #     attributes = ['North_Polar_Projection/cell_tb_v_aft', 'North_Polar_Projection/tb_cell_lat',
    #                   'North_Polar_Projection/tb_cell_lon', '/none']
    # else:
    #     print 'there is no %s data' % layername
    return att_read


def get_layer(key):
    """
    Return the first layer selected on the basis of keys
    :param
        sensor: tb or sigma
    :return:
    """
    layer1 = {'gm': u'Global_Projection', 'np': u'North_Polar_Projection', 'radar': u'Sigma0_Data', 'flag': u'flag'}
    return layer1[key]


def in_situ_measurement(site_no):
    measured = 'Date,' \
               'Soil Moisture Percent -2in (pct),Soil Moisture Percent -8in (pct),Soil Moisture Percent -20in (pct),' \
               'Soil Temperature Observed -2in (degC),Soil Temperature Observed -8in (degC),Soil Temperature Observed -20in (degC),' \
               'Air Temperature Average (degC),Air Temperature Maximum (degC),Air Temperature Minimum (degC),Air Temperature Observed (degC),' \
               'Precipitation Accumulation (mm),Precipitation Increment (mm),Snow Water Equivalent (mm)'
    air0 = "Air Temperature Observed (degC)"
    if site_no in ['2065', '2081']:
        air0 = "Air Temperature Average (degC)"
    return 0


def get_site_h5(siteno, date_version='_07_01'):
    if date_version == '_07_01':
        path0 = 'result%s/methods/h5_result/*%s*.h5' % (date_version, siteno)
        f_path = glob.glob(path0)
    return f_path


def site_onset(site_no, orbit='A'):
    onset_file = 'result_07_01/txtfiles/result_txt/smap_ft_compare_%s.csv' % orbit
    onset_fromfile = np.loadtxt(onset_file, delimiter=',')
    onset_value = onset_fromfile[onset_fromfile[:, 10] == int(site_no), :]
    return onset_value


def ascat_heads(key0):

    heads = {}
    heads['ascat0'] = ['latitude, longitude, sigma0_trip0, sigma0_trip1, sigma0_trip2, f_usable0, f_usable1, ' \
        'f_usable2, inc_angle_trip0, inc_angle_trip1, inc_angle_trip2, f_land0, f_land1, f_land2, ' \
        'utc_line_nodes, abs_line_number, sat_track_azi, swath_indicator, kp0, kp1, kp2, ' \
        'azi_angle_trip0, azi_angle_trip1, azi_angle_trip2,  num_val_trip0, num_val_trip1, num_val_trip2, ' \
        'f_f0, f_f1, f_f2, f_v0, f_v1, f_v2, f_oa0, f_oa1, f_oa2, f_sa0, f_sa1, f_sa2, f_tel0, f_tel1, f_tel2,' \
        'f_ref0, f_ref1, f_ref2, as_des']

    heads['ascat'] = ['latitude', 'longitude', 'sigma0_trip', 'f_usable', 'inc_angle_trip', 'f_land'
                                      , 'utc_line_nodes', 'abs_line_number', 'sat_track_azi', 'swath_indicator',
                                      'kp', 'azi_angle_trip', 'num_val_trip', 'f_f', 'f_v', 'f_oa', 'f_sa', 'f_tel',
                                      'f_ref', 'as_des_pass']
    return heads[key0]


def all_att_smap_l1c():
    att_list = [u'cell_antenna_scan_angle_aft', u'cell_antenna_scan_angle_fore', u'cell_boresight_incidence_aft',
                u'cell_boresight_incidence_fore', u'cell_column', u'cell_lat', u'cell_lat_centroid_aft',
                u'cell_lat_centroid_fore', u'cell_lon', u'cell_lon_centroid_aft', u'cell_lon_centroid_fore',
                u'cell_number_measurements_3_aft', u'cell_number_measurements_3_fore',
                u'cell_number_measurements_4_aft', u'cell_number_measurements_4_fore', u'cell_number_measurements_h_aft',
                u'cell_number_measurements_h_fore', u'cell_number_measurements_v_aft', u'cell_number_measurements_v_fore',
                u'cell_row', u'cell_solar_specular_phi_aft', u'cell_solar_specular_phi_fore', u'cell_solar_specular_theta_aft',
                u'cell_solar_specular_theta_fore', u'cell_tb_3_aft', u'cell_tb_3_fore', u'cell_tb_4_aft', u'cell_tb_4_fore',
                u'cell_tb_error_3_aft', u'cell_tb_error_3_fore', u'cell_tb_error_4_aft', u'cell_tb_error_4_fore',
                u'cell_tb_error_h_aft', u'cell_tb_error_h_fore', u'cell_tb_error_v_aft', u'cell_tb_error_v_fore',
                u'cell_tb_h_aft', u'cell_tb_h_fore', u'cell_tb_qual_flag_3_aft', u'cell_tb_qual_flag_3_fore',
                u'cell_tb_qual_flag_4_aft', u'cell_tb_qual_flag_4_fore', u'cell_tb_qual_flag_h_aft',
                u'cell_tb_qual_flag_h_fore', u'cell_tb_qual_flag_v_aft', u'cell_tb_qual_flag_v_fore',
                u'cell_tb_time_seconds_aft', u'cell_tb_time_seconds_fore', u'cell_tb_time_utc_aft',
                u'cell_tb_time_utc_fore', u'cell_tb_v_aft', u'cell_tb_v_fore']
    return att_list


def cols_nums(name_str):
    # ascat
    ascat_dict = {'orbit': [-2]}
    return ascat_dict[name_str][0]


def get_sno_list(type_str='string'):
    site_dic = {}
    site_dic['string'] = \
        ['947', '949', '950', '960', '962', '967', '968', '1090', '1175',
         '1177', '1233', '2065', '2081', '2210', '2211', '2212', '2213']
    site_dic['int'] = [947, 949, 950, 960, 962, 967, 968, 1090, 1175,
         1177, 1233, 2065, 2081, 2210, 2211, 2212, 2213]
    site_dic['test'] = ['968', '1090']
    return site_dic[type_str]


def get_satellite_type():
    return ['metopA_A.h5', 'metopA_D.h5', 'metopB_A.h5', 'metopB_D.h5']


def get_ascat_grid_keys():
    suffix = ['_fore', '_mid', '_aft']
    keys = []
    with open('meta0_ascat_ak.txt') as f0:
        for row0 in f0:
            row = row0.split(', ')
            trip_check = int(row[2])-int(row[1])
            if trip_check < 1.5:
                keys.append(row[0])
            else:
                for i_0 in range(0, trip_check):
                    keys.append(row[0] + suffix[i_0])
    return keys


def ascat_grid_keys():
    return ['latitude', 'longitude', 'sigma0_trip_fore', 'sigma0_trip_mid', 'sigma0_trip_aft', 'f_usable_fore',
            'f_usable_mid', 'f_usable_aft', 'inc_angle_trip_fore', 'inc_angle_trip_mid', 'inc_angle_trip_aft',
            'f_land_fore', 'f_land_mid', 'f_land_aft', 'utc_line_nodes', 'abs_line_number', 'sat_track_azi',
            'swath_indicator', 'kp_fore', 'kp_mid', 'kp_aft', 'azi_angle_trip_fore', 'azi_angle_trip_mid',
            'azi_angle_trip_aft', 'num_val_trip_fore', 'num_val_trip_mid', 'num_val_trip_aft', 'f_f_fore',
            'f_f_mid', 'f_f_aft', 'f_v_fore', 'f_v_mid', 'f_v_aft', 'f_oa_fore', 'f_oa_mid', 'f_oa_aft', 'f_sa_fore',
            'f_sa_mid', 'f_sa_aft', 'f_tel_fore', 'f_tel_mid', 'f_tel_aft', 'f_ref_fore', 'f_ref_mid', 'f_ref_aft',
            'as_des_pass', 'distance']


def ascat_keys_col(att_str):
    keys_47 = ['latitude', 'longitude', 'sigma0_trip_fore', 'sigma0_trip_mid', 'sigma0_trip_aft', 'f_usable_fore',
            'f_usable_mid', 'f_usable_aft', 'inc_angle_trip_fore', 'inc_angle_trip_mid', 'inc_angle_trip_aft',
            'f_land_fore', 'f_land_mid', 'f_land_aft', 'utc_line_nodes', 'abs_line_number', 'sat_track_azi',
            'swath_indicator', 'kp_fore', 'kp_mid', 'kp_aft', 'azi_angle_trip_fore', 'azi_angle_trip_mid',
            'azi_angle_trip_aft', 'num_val_trip_fore', 'num_val_trip_mid', 'num_val_trip_aft', 'f_f_fore',
            'f_f_mid', 'f_f_aft', 'f_v_fore', 'f_v_mid', 'f_v_aft', 'f_oa_fore', 'f_oa_mid', 'f_oa_aft', 'f_sa_fore',
            'f_sa_mid', 'f_sa_aft', 'f_tel_fore', 'f_tel_mid', 'f_tel_aft', 'f_ref_fore', 'f_ref_mid', 'f_ref_aft',
            'as_des_pass', 'distance']
    if type(att_str) is str:
        return keys_47.index(att_str)
    else:
        return keys_47[att_str]


def ascat_timeseries_keys(att0_main):
    suffix = ['_'+a0 for a0 in ['9', '0', '1', '2', '3', '4', '5', '6', '7', '8']]
    return []


def empty_array_sz(name):
    dict0 = {
             'inc_angle_trip_mid': 9, 'inc_angle_trip_aft': 9, 'inc_angle_trip_fore': 9,
             'utc_line_nodes': 9,
             'sate_type': 1,
             'sigma0_trip_mid': 9, 'sigma0_trip_aft': 9, 'sigma0_trip_fore': 9,
             }
    return dict0[name]


def ascat_dictionary():
    # a structure array saving the meta data of ascat dictionary
    return 0