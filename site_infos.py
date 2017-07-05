"""
Storage of site No., site info., attribute of h5 files
"""

def get_type(id):
    site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968','1090', '1175', '1177'],
                    'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
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


def get_id(id, mode='all'):
    site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177', '2210', '1089', '1233', '2212', '2211']
    if mode is 'all':
        return site_nos
    else:
        return [id]


def change_site(site_no):  # the No, lat and long of each site
    site_list = [['947', 65.12422, -146.73390], ['948', 65.25113, -146.15133], ['949', 65.07833, -145.87067],
                 ['950', 64.85033, -146.20945],['1090', 65.36710, -146.59200], ['960', 65.48, -145.42],
                 ['962', 66.74500, -150.66750], ['1233', 59.82001, -156.99064], ['2213', 65.40300, -164.71100],
                 ['2081', 64.68582, -148.91130], ['2210', 65.19800, -156.63500], ['2211', 63.63900, -158.03010],
                 ['2212', 66.17700, -151.74200], ['966', 60.72700, -150.47517], ['1175', 67.93333, -162.28333],
                 ['2065', 61.58337, -159.57708], ['967', 62.13333, -150.04167], ['968', 68.61683, -149.30017],
                 ['1062', 59.85972, -151.31500], ['1089', 62.63000, -150.77617], ['1177', 70.26666, -148.56666],
                 ['alaska', 63.25, -151.5]]
    site_no_list = [row[0] for row in site_list]
    siteinfo = site_list[site_no_list.index(site_no)]
    return siteinfo


def get_siteNo_list():
    site_no_list = ['947', '948', '949', '950', '960', '1090']
    # site_no_list = '1090'
    return site_no_list


def get_data_root(site_no, date='0702'):
    data_root = '/home/xiyu/PycharmProjects/R3/result' + date + '/s' + site_no + '/'
    return data_root


def get_data_path(date='_05_01'):
    path = '/home/xiyu/PycharmProjects/R3/result' + date + '/'
    return path


def get_attribute(layername, sublayer='sig_vv_aft'):
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
        {'tbh_aft': ['cell_tb_h_aft',  'tb_cell_lat', 'tb_cell_lon', 'cell_tb_qual_flag_h_aft'],
         'tbv_aft': ['cell_tb_v_aft',  'tb_cell_lat', 'tb_cell_lon', 'cell_tb_qual_flag_v_aft'],
         'cell_tb_v_aft': ['cell_tb_v_aft',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_v_aft', 'cell_tb_error_h_aft',
                  'cell_tb_time_utc_aft', 'cell_boresight_incidence_aft'],
         'cell_tb_v_fore': ['cell_tb_v_fore',  'cell_lat', 'cell_lon', 'cell_tb_qual_flag_v_fore', 'cell_tb_error_h_fore',
                  'cell_tb_time_utc_fore', 'cell_boresight_incidence_fore'],
         'sig_vv_aft': ['cell_lat', 'cell_lon', 'cell_sigma0_qual_flag_vv', 'cell_sigma0_vv_aft'],
         'sig_hh_aft': ['cell_lat', 'cell_lon', 'cell_sigma0_qual_flag_vv', 'cell_sigma0_vv_aft']}
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
