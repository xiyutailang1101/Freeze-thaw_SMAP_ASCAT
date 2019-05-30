__author__ = 'xiyu'
import site_infos
import numpy as np
from mpl_toolkits.basemap import Basemap
import pytesmo.colormaps.load_cmap as smcolormaps
import pytesmo.grid.resample as res
import basic_xiyu as bs
import h5py
import read_site
import data_process
import plot_funcs
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import re
from datetime import date as dtime
from datetime import datetime as dtime2
from netCDF4 import Dataset
import h5py
import test_def
import basic_xiyu as bxy
import glob

def smap_mask(is_test=False):
    # read ascat land mask
    ascat_mask = np.load('./result_05_01/other_product/mask_ease2_125N.npy')
    ascat_lat, ascat_lon = np.load('lat_ease_grid.npy'), np.load('lon_ease_grid.npy')
    # new updated grid on 2018
    h0 = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151121.h5')
    smap_lon = h0['cell_lon'].value
    smap_lat = h0['cell_lat'].value
    # smap_lat, smap_lon = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy'), \
    #                      np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
    v = np.zeros(ascat_mask.shape, dtype=int)
    v[ascat_mask] = 1
    v_dict = {'mask': v[ascat_mask]}
    smap_mask_dict = res.resample_to_grid(v_dict, ascat_lon[ascat_mask], ascat_lat[ascat_mask], smap_lon, smap_lat)
    np.save('./result_05_01/other_product/mask_ease2_360N', smap_mask_dict['mask'].data)

    # test the mask
    if is_test == True:
        mask_t = smap_mask_dict['mask']
        fig2 = plt.figure(figsize=[8, 8])
        ax2 = fig2.add_subplot(111)
        m2 = Basemap(width=10e6, height=6e6, resolution='l', projection='laea', lat_ts=62, lat_0=52, lon_0=-160., ax=ax2)
        m2.scatter(smap_lon[~mask_t.mask], smap_lat[~mask_t.mask], 1, marker='*', color='k', latlon=True)
        m2.drawcoastlines()
        plt.savefig('spatial_smap_grid.png', dpi=120)
        plt.close()

    # save as smap land mask

    return 0


def build_mask(mode='sig'):
    if mode=='sig':
        v = np.loadtxt('/home/xiyu/Data/Snow/perennial05.txt')
        mask_vu, lon_mask0, lat_mask0 = v[:, -1], v[:, -2], v[:, -3]
        snow_id = mask_vu == 12
        mask_v, lon_mask0, lat_mask0 = mask_vu[snow_id], lon_mask0[snow_id], lat_mask0[snow_id]
        mask0 = {'snow': mask_v}
        grp0 = Dataset('/home/xiyu/Data/easegrid2/EASE2_N12.5km.geolocation.v0.9.nc', 'r', format='NETCDF4')
        ease_lat, ease_lon = grp0.variables['latitude'][:], grp0.variables['longitude'][:]
        lat_gd, lon_gd = ease_lat[400: 700, 400: 700], ease_lon[400: 700, 400: 700]
        resampled_ascat0 = res.resample_to_grid(mask0, lon_mask0, lat_mask0, lon_gd, lat_gd, search_rad=9000)
        data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat0['snow'], '/home/xiyu/Data/Snow/', fname='snow_mask01', z_min=0, z_max=15)
        np.save('./result_05_01/other_product/snow_mask_125', resampled_ascat0['snow'].data)
    elif mode=='npr':
        v = np.loadtxt('/home/xiyu/Data/Snow/perennial05.txt')
        mask_vu, lon_mask0, lat_mask0 = v[:, -1], v[:, -2], v[:, -3]
        snow_id = mask_vu == 12
        mask_v, lon_mask0, lat_mask0 = mask_vu[snow_id], lon_mask0[snow_id], lat_mask0[snow_id]
        mask0 = {'snow': mask_v}
        lon_gd = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
        lat_gd = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
        resampled_ascat0 = res.resample_to_grid(mask0, lon_mask0, lat_mask0, lon_gd, lat_gd, search_rad=20000)
        np.save('./result_05_01/other_product/snow_mask_360', resampled_ascat0['snow'].data)
        data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat0['snow'], '/home/xiyu/Data/Snow/', fname='snow_mask01', z_min=0, z_max=15)
    elif mode=='30km_snow_mask':
        mask_snow = np.loadtxt('/home/xiyu/Data/Snow/30km_snow_mask.txt', comments=';')
        mask_v, lon_mask0, lat_mask0 = mask_snow[:, -1], mask_snow[:, -2], mask_snow[:, -3]
        h5_name = 'result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_%s.h5' % '20151102'
        h0 = h5py.File(h5_name)
        lon_gd = h0['cell_lon'].value
        lat_gd = h0['cell_lat'].value
        mask0 = {'snow': mask_v}
        resampled_ascat0 = res.resample_to_grid(mask0, lon_mask0, lat_mask0, lon_gd, lat_gd, search_rad=30000)
        np.save('./result_05_01/other_product/snow_mask_360_2', resampled_ascat0['snow'].data)
        data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat0['snow'], '/home/xiyu/Data/Snow/', fname='snow_mask0726', z_min=0, z_max=15)


def smap_area_plot(datez, save_dir='./result_05_01/smap_area_result/', orbit='AS'):
    """
    The extracted h5 file have all pixels located in alaska in the "given date", we first read the cell row and col info
    from it. Then transfer this (500, 500) ease grid 36N system to a (90, 100) ease grid 36N system, covering the alaska
    regions. The transfer is: trans(row, col) = origin(row, col) - (140, 160)
    :param datez:
    :param save_dir:
    :param orbit:
    :return:
    """
    ob = orbit
    if orbit == 'A':
        fpath0 = './result_05_01/SMAP_AK/smap_ak_as/AS_'
    else:
        fpath0 = './result_05_01/SMAP_AK/smap_ak_as/DES_'
    # initial parameters
    h5_ungrid_name = 'result_08_01/area/SMAP/SMAP_alaska_%s_%s.h5' % (ob, datez)
    h5_ungrid_daily_obj = h5py.File(h5_ungrid_name)
    if 'North_Polar_Projection' not in h5_ungrid_daily_obj.keys():
        print 'no North_Polar_Projection on %s' % datez
        with open('smap_area_plot_interpolate.out', 'a') as writer0:
            writer0.write(datez)
        return -1
    smap_data = h5_ungrid_daily_obj['North_Polar_Projection']
    # read aft tb at v and h, read their coordinates
    lat_tb0, lon_tb0 = smap_data['cell_lat'].value, smap_data['cell_lon'].value

    # the required attributes
    att_list = [u'cell_antenna_scan_angle_aft', u'cell_boresight_incidence_aft', u'cell_column',
                u'cell_lat_centroid_aft', u'cell_lon_centroid_aft',
                u'cell_row', u'cell_solar_specular_phi_aft', u'cell_solar_specular_theta_aft',
                u'cell_solar_specular_theta_fore',
                u'cell_tb_error_h_aft', u'cell_tb_error_h_fore', u'cell_tb_error_v_aft', u'cell_tb_error_v_fore',
                u'cell_tb_h_aft', u'cell_tb_h_fore', u'cell_tb_qual_flag_h_aft',
                u'cell_tb_qual_flag_h_fore', u'cell_tb_qual_flag_v_aft', u'cell_tb_qual_flag_v_fore',
                u'cell_tb_time_seconds_aft', u'cell_tb_time_seconds_fore', u'cell_tb_v_aft', u'cell_tb_v_fore']
    smap_dict0 = {}
    for key0 in att_list:
        smap_dict0[key0] = smap_data[key0].value

    # if not os.path.exists(save_dir+datez):
    #     os.makedirs(save_dir+datez)
    ease_lat_un = np.fromfile('/home/xiyu/Data/easegrid2/gridloc.EASE2_N36km/EASE2_N36km.lats.500x500x1.double', dtype=float).reshape(500, 500)
    ease_lon_un = np.fromfile('/home/xiyu/Data/easegrid2/gridloc.EASE2_N36km/EASE2_N36km.lons.500x500x1.double', dtype=float).reshape(500, 500)
    ease_lat = np.ma.masked_array(data=ease_lat_un, mask=[ease_lat_un == -999])
    ease_lon = np.ma.masked_array(data=ease_lon_un, mask=[ease_lon_un == -999])
    row_range = range(140, 230)
    col_range = range(160, 260)
    # make grid
    smap_grid_dict = {}
    for key0 in att_list:
        smap_grid_dict[key0] = np.zeros([90, 100]) - 999

    if lat_tb0.size < 1:
        print 'no data in date %s' % datez
    else:
        cell_row, cell_col = smap_data[u'cell_row'].value, smap_data[u'cell_column'].value
        print 'the number of data is: ', cell_row.size
        if cell_row.size > 1000:
            pause = 0
        grid_coord = [np.array([row0-140 for row0 in cell_row]), np.array([col0-160 for col0 in cell_col])]
        for key0 in att_list:
            smap_grid_dict[key0][grid_coord] = smap_dict0[key0]
        # resample a smap_sea_mask

        # resampled_smap0 = res.resample_to_grid\
        #     (smap_dict0, lon_tb, lat_tb, ease_lon[np.ix_(row_range, col_range)],
        #      ease_lat[np.ix_(row_range, col_range)], search_rad=27000)
        resampled_smap0 = smap_grid_dict
        # resampled_smap0 = res.resample_to_grid(smap_dict0, lon_tb, lat_tb, grid_lon[~grid_lon.mask], grid_lat[~grid_lat.mask], search_rad=27000)
        v_map_un = np.zeros(ease_lon.shape)
        # SMAP_alaska_A_20160124.h5
        h5_grid_name = '%s/SMAP_alaska_%s_GRID_%s.h5' % (save_dir, ob, datez)
        # check file path and name
        pause=0
        # h5_grid_name = 'result_08_01/area/smap_area_result/SMAP_alaska_%s_GRID_%s.h5' % (ob, datez)
        h0 = h5py.File(h5_grid_name, 'a')
        for key1 in resampled_smap0.keys():
            h0[key1] = resampled_smap0[key1]  # assign value
        h0[u'cell_lon'] = ease_lon[np.ix_(row_range, col_range)]
        h0[u'cell_lat'] = ease_lat[np.ix_(row_range, col_range)]
        h0.close()
    h5_ungrid_daily_obj.close()

        # for key in ['tbv', 'tbh']:
        #     v_map_un[~ease_lon.mask] = resampled_smap0[key].data
        #     v_map = np.ma.masked_array(v_map_un, mask=[(v_map_un==-9999)|(v_map_un==0)])
        #     tp0 = [[140, 219], [160, 249]]
        #     data_process.pass_zone_plot(ease_lon[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1],
        #                                 ease_lat[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1],
        #                                 v_map[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1]
        #                                 , pre_path, fname='smap_'+key+'_'+datez, z_min=185, z_max=285)
        #     ease_y = ease_lat[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1]
        #     ease_x = ease_lon[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1]
        #     np.save('./result_05_01/other_product/lat_ease2_360N_grid', ease_y.data)
        #     np.save('./result_05_01/other_product/lon_ease2_360N_grid', ease_x.data)
        #     np.save('./result_05_01/smap_resample_'+ob+'/all/smap_'+datez+'_'+key+'_resample', v_map[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1].data)


def ascat_area_plot2(datez, lat_valid, lon_valid, save_dir='./result_05_01/test_area_result_9km_',
                     orbit_no=0, format_ascat='npy', sate=False, attributs=[3, 12, 9, 45, 6, 2, 4, 11, 13, 8, 10, 5, 7]):
    '''
    the input ascat data covers a given region, here we project them into a given grid system with coordinate
    [lon_gd, lat_gd]
    :param datez:
    :param lat_valid:
    :param lon_valid:
    :param save_dir:
    :param orbit_no: 0 for ascending
    :param format_ascat:
    :param sate:
    :param attributs: the indices ranging (0, 46), such as 'inc_angle_trip_mid', referred to R3/meta0_ascat_ak.txt
    :return: gridded .h5 files, whose name is consist of:
            1) ['metopA_A', 'metopB_A', 'metopA_D', 'metopB_D'], corresponded sate type [0, 1, 2, 3]
            2) timing of the measurements: yyyymmdd_hour
            3) orbit: A for ascending; D for descending.
    '''
    # start0 = dtime2.now()
    # print 'the orbit number is %d' % orbit_no
    sate_type = ['metopA_A', 'metopB_A', 'metopA_D', 'metopB_D']
    ascat_keys = site_infos.ascat_grid_keys()
    if orbit_no == 0:  # AS
        ob = 'A'
        tzone = [[18, 19.5], [19.5, 21], [21, 22.5], [22.5, 24], [-0.5, 1], [1, 2.5]]
    else:
        ob = 'D'
        tzone = [[2.5, 4], [4, 5.5], [5.5, 7], [7, 8.5], [8.5, 10]]
    save_dir = save_dir+ob+'/'
    # # read the grid system from nc file
    # grp0 = Dataset('/home/xiyu/Data/easegrid2/EASE2_N12.5km.geolocation.v0.9.nc', 'r', format='NETCDF4')
    # ease_lat, ease_lon = grp0.variables['latitude'][:], grp0.variables['longitude'][:]
    # lat_valid, lon_valid = (ease_lat > 54) & (ease_lat < 72), (ease_lon > -170) & (ease_lon < -130)
    # ease_valid = lat_valid & lon_valid
    # lat_gd, lon_gd = ease_lat[400: 700, 400: 700], ease_lon[400: 700, 400: 700]
    # np.save('lat_ease_grid', lat_gd.data), np.save('lon_ease_grid', lon_gd.data)
    lon_gd, lat_gd = np.load('./extra_data/lon_ease_grid_125_ak.npy'), \
                                np.load('./extra_data/lat_ease_grid_125_ak.npy')
    # lats_dim, lons_dim = np.arange(54, 72, 0.1), np.arange(-170, -130, 0.1)
    # lons_grid, lats_grid = np.meshgrid(lons_dim, lats_dim)

    # read the ascat, constraint by land% > 50%, orbit, and usable flag
    if sate is False:
        ascat_data = np.load('./result_08_01/area/ascat/ascat_'+datez+'_metopB_alaska.npy')
        sate = 'metopB'
    else:
        ascat_data = np.load('./result_08_01/area/ascat/ascat_'+datez+'_metopA_alaska.npy')
        sate = 'metopA'
    ascat_dict = {}
    meta_file = 'meta0_ascat_ak.txt'
    with open(meta_file) as meta0:
        content = meta0.readlines()
        metas = [x.strip() for x in content]
    for meta0 in metas:
        if len(meta0)>1:
            att_range = meta0.split(',')  # 0-2: name of attribute, start index, end index
            if att_range[0] == 'f_usable':
                st, en = int(att_range[1]), int(att_range[2])  # fore to after
            # ascat_dict[att_range[0]] = ascat_data[:, st: en]
        else:
            continue
    if ascat_data.size < 1:
        # empty, the orbit doesn't exist
        with open('ascat_area_plot2_metopA_empty.txt', 'a') as f0:
            f0.write('No data: Date: %s, orbit: A, B\n' % datez)
        return 0
    # read data quality, land cover, and orbit
    orbit_id = ascat_data[:, attributs[3]]
    f_use = ascat_data[:, attributs[4]]
    land_mid = ascat_data[:, attributs[1]]
    land_fore, land_aft = ascat_data[:, attributs[7]], ascat_data[:, attributs[8]]
    f_use_fore, f_use_aft = ascat_data[:, attributs[11]], ascat_data[:, attributs[12]]
    # filter the unqualified
    id_valid_aft, id_valid_fore = ascat_filter([land_aft, orbit_id, f_use_aft], orbit_no), \
                                  ascat_filter([land_fore, orbit_id, f_use_fore], orbit_no)
    id_valid_mid = ascat_filter([land_mid, orbit_id, f_use], orbit_no)
    id_valid_ak = id_valid_aft & id_valid_fore & id_valid_mid
    # read measurements
    print 'mid (%d, %d), fore (%d, %d), aft (%d, %d)' % (attributs[0], attributs[2], attributs[5], attributs[9],
                                                         attributs[6], attributs[10])
    sigma_land, inc_land = ascat_data[:, attributs[0]][id_valid_ak], \
                         ascat_data[:, attributs[2]][id_valid_ak]  # mid measurements
    sigma_fore_land, sigma_aft_land = ascat_data[:, attributs[5]][id_valid_ak], \
                                      ascat_data[:, attributs[6]][id_valid_ak]
    inc_fore_land, inc_aft_land = ascat_data[:, attributs[9]][id_valid_ak], \
                                  ascat_data[:, attributs[10]][id_valid_ak]

    # check

    # for i2 in [0, 2, 5, 6, 9, 10]:
    #     print 'the %d_th attribute' % attributs[i2]
    #     key0 = site_infos.ascat_keys_col(attributs[i2])
    #     print 'has name %s' % key0
    # for a_no in np.array(attributs).astype(int)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]:
    #     key0 = site_infos.ascat_keys_col(a_no)
    #     print 'read from given attributes: %s' % (key0)

    if sigma_land.size < 1:
        # empty, the orbit doesn't exist
        with open('ascat_area_plot2_metopA_empty.txt', 'a') as f0:
            f0.write('No data: Date: %s, orbit: %s\n' % (datez, ob))
        return 0
    lon_land, lat_land, pass_time = ascat_data[:, 1][id_valid_ak], \
                                    ascat_data[:, 0][id_valid_ak], \
                                    ascat_data[:, 14][id_valid_ak]
    pass_array = bxy.time_getlocaltime(pass_time, ref_time=[2000, 1, 1, 0])
    pass_hr = pass_array[-1]
    # pass_hr = pass_time
    # print("----fist part: %s seconds ---" % (dtime2.now()-start0))
    # start0 = dtime2.now()
    ############################
    daily_hour = pass_hr
    # print("----2nd part: %s seconds ---" % (dtime2.now()-start0))
    # start0 = dtime2.now()
    # daily_hour = (daily_sec/3600).astype(int)
    u_v, u_i = np.unique(daily_hour, return_index=True)  # seconds integral hour
    for hour0 in u_v:
        id_loc_time0 = daily_hour == hour0
        sigma_land0, inc_land0, pass_hr0, pass_time0 = \
        sigma_land[id_loc_time0], inc_land[id_loc_time0], pass_hr[id_loc_time0], pass_time[id_loc_time0]

        sigma_fore0, sigma_aft0, inc_fore0, inc_aft0 = sigma_fore_land[id_loc_time0], sigma_aft_land[id_loc_time0], \
                                                       inc_fore_land[id_loc_time0], inc_aft_land[id_loc_time0]
        lon_land0, lat_land0 = lon_land[id_loc_time0], lat_land[id_loc_time0]

        # add the name of keys
        # sigma_dict0 = {'sigma': sigma_land0, 'incidence': inc_land0,
        #                'pass_utc': pass_time0}  # pass_hr0 --> pass_time0, 0829/2018
    #         sigma_fore_land, sigma_aft_land = ascat_data[:, attributs[5]][id_valid_fore], \
    #                                   ascat_data[:, attributs[6]][id_valid_aft]
    # inc_fore_land, inc_aft_land = ascat_data[:, attributs[9]][id_valid_fore], \
    #                               ascat_data[:, attributs[10]][id_valid_aft]
        sigma_dict0 = {'sigma0_trip_mid': sigma_land0, 'inc_angle_trip_mid': inc_land0,
                       ascat_keys[14]: pass_time0,
                       'sigma0_trip_fore': sigma_fore0, 'sigma0_trip_aft': sigma_aft0,
                       'inc_angle_trip_fore': inc_fore0, 'inc_angle_trip_aft': inc_aft0}  # pass_hr0 --> pass_time0, 11/29/2018, attributs=[3, 12, 9, -1, 6]
        print 'the extracted keys are', sigma_dict0.keys()
        resampled_ascat0 = res.resample_to_grid(sigma_dict0, lon_land0, lat_land0, lon_gd, lat_gd, search_rad=9000)
        # print("----3rd part: %s seconds ---" % (dtime2.now()-start0))
        # start0 = dtime2.now()
        if format_ascat == 'h5':
            # save as h5 file, running speed is too low
            # print sate_type.index(sate+'_'+ob)
            h5_name = 'result_08_01/ascat_resample_all2/ascat_%s_%s_%d_%s.h5' % (sate, datez, hour0, ob)
            h50 = h5py.File(h5_name, 'a')
            for key0 in sigma_dict0.keys():
                if key0 in h50.keys():
                    del h50[key0]
                    h50[key0] = resampled_ascat0[key0].data  # delete and update
                    # dset = h50.create_dataset(key0, data=resampled_ascat0[key0].data)
                else:
                    h50[key0] = resampled_ascat0[key0].data
            if 'sate_type' not in h50.keys():
                h50['sate_type'] = sate_type.index(sate+'_'+ob)
            if 'latitude' not in h50.keys():
                h50['latitude'], h50['longitude'] = lat_gd, lon_gd
        elif format_ascat == 'npy':
            # save as npy file
            if sate is False:
                resample_name = './result_08_01/ascat_resample_npy/ascat_%s_%d_resample_%s' % (datez, hour0, ob)
                incidence_name = './result_08_01/ascat_resample_npy/ascat_%s_%d_incidence_%s' % (datez, hour0, ob)
                pass_name = './result_08_01/ascat_resample_npy/ascat_%s_%d_pass_utc_%s' % (datez, hour0, ob)
            else:
                sate_name = 'metop%s' % sate
                resample_name = './result_08_01/ascat_resample_npy/ascat_%s_%d_resample_%s_%s' \
                                % (datez, hour0, sate_name, ob)
                incidence_name = './result_08_01/ascat_resample_npy/ascat_%s_%d_incidence_%s_%s' \
                                 % (datez, hour0, sate_name, ob)
                pass_name = './result_08_01/ascat_resample_npy/ascat_%s_%d_pass_utc_%s_%s' \
                            % (datez, hour0, sate_name, ob)
            np.save(resample_name, resampled_ascat0['sigma'].data)
            np.save(incidence_name, resampled_ascat0['incidence'].data)
            np.save(pass_name, resampled_ascat0['pass_utc'].data)
    ###########################
    # for tz in tzone:
    #     n0 += 1
    #     id_loc_time0 = (pass_hr > tz[0]) & (pass_hr < tz[1])
    #     if sum(id_loc_time0) < 1:  # no data interval
    #         continue
    #     else:
    #         n1 += 1
    #         sigma_land0, inc_land0, pass_hr0, pass_time0 = \
    #             sigma_land[id_loc_time0], inc_land[id_loc_time0], pass_hr[id_loc_time0], pass_time[id_loc_time0]
    #         lon_land0, lat_land0 = lon_land[id_loc_time0], lat_land[id_loc_time0]
    #         # select attributes that we want to save !! 08/29/2018
    #         sigma_dict0 = {'sigma': sigma_land0, 'incidence': inc_land0,
    #                        'pass_utc': pass_time0}  # pass_hr0 --> pass_time0, 0829/2018
    #         resampled_ascat0 = res.resample_to_grid(sigma_dict0, lon_land0, lat_land0, lon_gd, lat_gd, search_rad=9000)
    #         mean_pass = np.mean(pass_hr0)
    #
    #         resample_name = './result_08_01/ascat_resample_%s/new/ascat_%s_%d_resample' % (ob, datez, 0.5*(tz[0]+tz[1]))
    #         incidence_name = './result_08_01/ascat_resample_%s/new/ascat_%s_%d_incidence' % (ob, datez, 0.5*(tz[0]+tz[1]))
    #         pass_name = './result_08_01/ascat_resample_%s/new/ascat_%s_%d_pass_utc' % (ob, datez, 0.5*(tz[0]+tz[1]))
    #         np.save(resample_name, resampled_ascat0['sigma'].data)
    #         np.save(incidence_name, resampled_ascat0['incidence'].data)
    #         np.save(pass_name, resampled_ascat0['pass_utc'].data)
                # fig_name = 'zone_%s_%.2f' % (str(tzone[n0][0]*10), mean_pass)
                # data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat0['sigma'], pre_path, fname=fig_name)
    # resampled_ascat0['sigma'] -= (resampled_ascat0['incidence']-45.0)*-0.11  # normalization
    # np.save('./result_05_01/ascat_resample_norms/ascat_resample_'+ob+'/ascat_'+datez+'_resample', resampled_ascat0['sigma'].data)
    # add the station location
    # for i_tp in ['merc', 'laea']:
    #     fig = plt.figure(figsize=[8, 8])
    #     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #     if i_tp == 'merc':
    #         m = Basemap(llcrnrlon=-170, llcrnrlat=54, urcrnrlon=-130, urcrnrlat=72,
    #                     resolution='l', area_thresh=1000., projection='merc',
    #                     lat_ts=50., ax=ax)
    #     else:
    #         m = Basemap(width=4e6, height=4e6, resolution='l', projection='laea', lat_ts=62, lat_0=62, lon_0=-150., ax=ax)
    #     im = m.pcolormesh(lon_gd, lat_gd, resampled_ascat0['sigma'], latlon=True, vmax=-4, vmin=-20)
    #     m.drawcoastlines()
    #     cb = m.colorbar(im)
    #     site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177', '2210',
    #                 '1089', '1233', '2212', '2211']
    #     s_lat, s_lon, s_name = [], [], []
    #     for no in site_nos:
    #         s_name.append(site_infos.change_site(no)[0])
    #         s_lat.append(site_infos.change_site(no)[1])
    #         s_lon.append(site_infos.change_site(no)[2])
    #     m.scatter(s_lon, s_lat, 5, marker='*', color='k', latlon=True)
    #     plt.savefig(pre_path+'spatial_ascat0_'+i_tp+'.png', dpi=120)
    #     plt.close()
    # MOVED FROM MID
    # n, bins, patches = plt.hist(pass_hr, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.minorticks_on()
    # plt.tick_params(which='minor', color='r')
    # plt.savefig(pre_path+'spatial_ascat0_timehist.png')
    # sigma_dict = {'sigma': sigma_land, 'incidence': inc_land, 'pass': pass_hr}
    # plt.close()

    # plot the result
    # fig = plt.figure(figsize=[8, 8])
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # m = Basemap(llcrnrlon=-170, llcrnrlat=54, urcrnrlon=-130, urcrnrlat=72,
    #             resolution='l', area_thresh=1000., projection='merc',
    #             lat_ts=50., ax=ax)
    # im = m.pcolormesh(lons_grid, lats_grid, resampled_ascat['sigma'], latlon=True)
    # m.drawcoastlines()
    # cb = m.colorbar(im)
    # plt.savefig(pre_path+'spatial_ascat0_allpass.png', dpi=120)
    # plt.close()
    # fig = plt.figure(figsize=[8, 8])
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # m = Basemap(llcrnrlon=-170, llcrnrlat=5ascat_area_plot24, urcrnrlon=-130, urcrnrlat=72,
    #             resolution='l', area_thresh=1000., projection='merc',
    #             lat_ts=50., ax=ax)
    # im = m.pcolormesh(lons_grid, lats_grid, resampled_ascat['pass'], latlon=True, vmin=0, vmax=24)
    # m.drawcoastlines()
    # cb = m.colorbar(im)
    # plt.savefig(pre_path+'spatial_ascat3_timemap.png', dpi=120)
    # plt.close()
    # fig2 = plt.figure(figsize=[8, 8])
    # ax2 = fig2.add_subplot(111)
    # m2 = Basemap(llcrnrlon=-170, llcrnrlat=54, urcrnrlon=-130, urcrnrlat=72,
    #              resolution='l', area_thresh=1000., projection='merc',
    #             lat_ts=50., ax=ax)
    # sc = ax2.scatter(lon_land, lat_land, c=pass_hr)
    # plt.colorbar(sc)
    # m2.drawcoastlines()
    # plt.savefig(pre_path+'spatial_ascat3_timemap2.png', dpi=120)
    # plt.close()
    # separate the two time window
    # id_loc_time = (pass_hr>17.5) & (pass_hr<19.5)
    # if sum(id_loc_time) < 1:
    #     print 'No 9:30 pm pass in %s' % datez
    #     return 0
    # else:
    #     sigma_land0, inc_land0, pass_hr0 = sigma_land[id_loc_time], inc_land[id_loc_time], pass_hr[id_loc_time]
    #     lon_land0, lat_land0 = lon_land[id_loc_time], lat_land[id_loc_time]
    #     sigma_dict0 = {'sigma': sigma_land0, 'incidence': inc_land0, 'pass': pass_hr0}
    #     resampled_ascat0 = res.resample_to_grid(sigma_dict0, lon_land0, lat_land0, lons_grid, lats_grid)
    #     data_process.pass_zone_plot(lons_grid, lats_grid, resampled_ascat0['sigma'], pre_path, fname='_zone175')
    #
    # # get a normalized linear equation
    # id_value = ~resampled_ascat0['sigma'].mask
    # backscatter = resampled_ascat0['sigma'][id_value]
    # incidence = resampled_ascat0['incidence'][id_value]
    # a, b = np.polyfit(incidence, backscatter, 1)
    # f = np.poly1d([a, b])


def ascat_filter(vars, orb, threshold=[0.5, 1]):
    # thershold[2]: 0: good, 1: acceptable, 2: unusable
    id_valid_ak = (vars[0] > threshold[0]) & (vars[1] == orb) & (vars[2] < threshold[1])  # land > 0.5, the orbit
    return id_valid_ak


def build_land_mask(date_list):
    """
    build land mask using ascat f_land
    date_list contain 30 days to generate mask cover the alaska
    :return:
    """
    sigma_dict0 = {}
    'mask_land'
    value, lat, lon = [], [], []
    for datez in date_list:
        ascat_data = np.load('./result_08_01/area/ascat/ascat_'+datez+'_alaska.npy')
        value.append(ascat_data[:, 12])
        lat.append(ascat_data[:, 0])
        lon.append(ascat_data[:, 1])
    sigma_dict0['mask_land'] = np.array(value)
    lon = np.array(lon)
    lat = np.array(lat)
    # save ascat mask
    resampled_ascat0 = res.resample_to_grid(sigma_dict0, lon, lat, lon_gd, lat_gd, search_rad=9000)
    # save smap mask
    return 0


def refined_read_series():
    peroid = ['20160101', '20161225']
    site_no = '947'
    tbn, dtr2, dis = read_site.read_series(peroid[0], peroid[1], site_no, '_A_', data='np', dat='cell_tb_v_aft')
    tbnd, dtr2d, disd = read_site.read_series(peroid[0], peroid[1], site_no, '_D_', data='np', dat='cell_tb_v_aft')
    status = 1


def creat_h5(site_id, groupname):  # 20170521
    hf_objs = []
    for s0 in site_id:
        hf = h5py.File('./h5_l1c/smap_'+s0, 'a')
        hf.create_group('North_Polar_Projection/File_Loc')
        hf.close()
    return hf_objs


def save_site_location_as_txt():
    site_no_list = ['2065', '947', '967', '2213', '949', '950', '960', '962', '968', '1090', '1175', '1177',  '2081', '2210', '1089', '1233', '2212', '2211']
    site_loc = []
    for site_no in site_no_list:
        site_loc.append(site_infos.change_site(site_no))
    with open('site_location.txt', 'w') as output:
        for site in site_loc:
            output.write('%s, %.5f, %.5f\n' % (str(site[0]), site[1], site[2]))
    #np.savetxt('site_loction.txt', site_loc, fmt='|S10, %.5f, %.5f', header='name, Y_lat, X_lon'
    return 0


def resample_test():
    # try mpl
    initials = 0
    lat_dim, lon_dim = np.arange(50, 75, 0.1), np.arange(-170, -125, 0.1)
    lon_grid, lat_grid = np.meshgrid(lon_dim, lat_dim)
    file_ak = 'result_03_4/alaska02/AS_ascat_20160502_alaska.npy'
    t_ak = np.load(file_ak)
    lat_ak, lon_ak = t_ak[:, 0], t_ak[:, 1]
    v_ak = t_ak[:, 2]
    v_ak[v_ak < -50] = np.nan
    dict_ak = dict()
    dict_ak['sigma'] = v_ak
    r_dara = res.resample_to_grid(dict_ak, lon_ak, lat_ak, lon_grid, lat_grid)
    conti = 0


def txt_saved():
    x1 = np.load('result_03_4/alaska02/ascend/AS_ascat_20160101_alaska.npy')
    x2 = np.load('result_03_4/alaska02/ascend/AS_ascat_20160102_alaska.npy')
    x3 = np.load('result_03_4/alaska02/ascend/AS_ascat_20160103_alaska.npy')
    x4 = np.load('result_03_4/alaska02/ascend/AS_ascat_20160104_alaska.npy')
    no = 0
    for x in [x1, x2, x3, x4]:
        no += 1
        print x.shape[0]
        np.savetxt('test_AK_2016010' + str(no) + '.txt', x, delimiter=',')


def test_overlay(id, dict_base, dict_input, x, y):

    return 0


def ascat_alaka_area():  # 20170604
    # get the date string list
    file_list = os.listdir('/home/xiyu/PycharmProjects/R3/result_05_01/ASCAT_AK')
    date_list = []
    p_underline = re.compile('_')
    for file in file_list:
        date_list.append(p_underline.split(file)[1])
    date_list = sorted(date_list)
    for da in date_list:

        print 'Processing Alaska regional ASCAT data at %s' % da
        '''
        uncomment command below
        '''
        # spt_quick.ascat_area_plot2(da, orbit_no=2)  # give savedir and orbit (2: all, 1: ascend only)


def ascat_test_nn(id_as, id_des, site_file):
    '''
    return the lat and lon of nn at an orbit.
    :return:
    '''
    nn_as, nn_des = site_file[id_as], site_file[id_des]
    nn_as_loc, nn_des_loc = nn_as[:, 0:2], nn_des[:, 0:2]
    print 'location of ascending: \n', nn_as_loc, '\n'
    # print 'location of dscending: \n', nn_des_loc
    return nn_as_loc


def ascat_point_plot(center=False, dis0=19,
                     site_nos=['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968', '1090', '1175',
                               '1177', '2210', '1233', '2212', '2211'], sate='all', site_loc='ak'):

    site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968', '1090', '1175', '1177'],
                    'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
    if center is not False:
        all_subcenter = np.loadtxt(center, delimiter=',').T # the subcenters
        site_nos = all_subcenter[0].astype(int)
    for site_no in site_nos:
        main_sid = site_nos[0]
        site_no = str(site_no)
        ascat_point = []
        loc_as = np.array([])  # Location of nn for ascending pass
        x_time, sigma0, date_list, out_ascat, inc45_55 = [], [], [], [], []
        if center is not False:
            i_tb = all_subcenter[2] == float(site_no)
            site_subcenter = all_subcenter[:, i_tb]
            center_tb = [site_subcenter[0], site_subcenter[1]]
            txt_path = '/home/xiyu/PycharmProjects/R3/result_05_01/site_ascat/' + 's' + str(main_sid)+ '/'
        else:
            if site_loc == 'ak':
                txt_path = '/home/xiyu/PycharmProjects/R3/result_08_01/point/ascat/' + 's' + str(site_no) + '/'
                out_path = \
                    './result_08_01/point/ascat/ascat_site_series/ascat_s' + str(site_no)+'_2016'+sate
            elif site_loc == 'tibet':
                txt_path = 'result_08_01/20181101/ascat_series/' + 's' + str(site_no) + '/'
                out_path = 'result_08_01/20181101/ascat_series/ascat_s' + str(site_no)+'_2016'+sate
        n = 0
        n_warning, n_inc = 0, 0
        if sate == 'all':
            file_list = sorted(os.listdir(txt_path))
        else:
            matchname = '%s*metop%s*.npy' % (txt_path, sate)
            path_list = glob.glob(matchname)
            file_list = [path0.split('/')[-1] for path0 in path_list]
            file_list = sorted(file_list)
        ascat_all_neighbor1d = np.array([])
        for txt_file in file_list:
            # get dt
            n+=1
            p_uline = re.compile('_')
            datei = p_uline.split(txt_file)[1]
            txt_i = np.load(txt_path + txt_file)
            if txt_i.size < 5:
                n_warning += 1
                # print 'no data is %s' % datei
                continue
            if txt_i.size > 5:
                date_list.append(datei)
                if len(txt_i.shape) < 2:  # only 1- d
                    locs, sig = np.array([txt_i[0:2]*1]), np.array([txt_i[2:5]*1])
                    f_u, inc = np.array([txt_i[5:8]]), np.array([txt_i[8:11]])
                    orb = np.array([txt_i[-1]])
                elif txt_i.shape[1] > 10:  # with triplets, flag, and inc angles
                    locs, sig = txt_i[:,  0:2], txt_i[:, 2:5]
                    f_u, inc = txt_i[:,  5:8]*1, txt_i[:, 8:11]
                    f_land = txt_i[:, 11:14]
                    t_utc = txt_i[:, 14]
                    orb = txt_i[:, -1]
                # print 'DATE:', datei
                id_nn, dis = data_process.ascat_nn\
                    (locs.T[1], locs.T[0], sig.T, orb.T, site_no, f=f_u.T, disref=dis0, pass_sec=t_utc)
                # test
                as_loc = ascat_test_nn(id_nn[0], id_nn[1], txt_i)
                loc_as = np.append(loc_as, as_loc)
                # pass_hr = 24*np.modf(t_utc/3600/24)[0]
                pass_hr = t_utc
                items = ['NN as_pass', 'NN des_pass', 'DOY', 'passing hours']
                idn = np.concatenate((id_nn[0], id_nn[1])).astype(int)
                dis_all = np.concatenate((dis[0], dis[1]))
                ascat_all_neighbor = np.zeros((txt_i[idn].shape[0], txt_i[idn].shape[1]+1))
                ascat_all_neighbor[:, 0:-1] = txt_i[idn]
                ascat_all_neighbor[:, -1] = dis_all
                ascat_all_neighbor1d = np.append(ascat_all_neighbor1d, ascat_all_neighbor.ravel())

                # sig_daily = data_process.ascat_ipt(dis, txt_i[:, [2, 3, 4, 8, 9, 10, 11, 12, 13, 0, 1]][idn].T, pass_hr[idn], orb[idn])
                # doy = np.fix(t_utc/3600/24 - (dtime(2016, 1, 1) - dtime(2000, 1, 1)).days)[0]
                # sig_table = [[doy]+t for t in sig_daily]
                # for v in sig_table:
                #     ascat_point.append(v)
                # tt = 0

        # np.save('./result_08_01/point/ascat/ascat_site_series/ascat_s'+str(site_no)+'_2016', np.array(ascat_point))
        np.save(out_path, ascat_all_neighbor1d.reshape(-1, 47))
        # np.save('./result_05_01/ascat_point/loc_a_s'+site_no+'_2016', loc_as.reshape(-1, 2))
    st = 0
    return 0


def ascat_sub_pixel(sub_info, center=False, dis0=19,
                    site_nos=['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968', '1090', '1175',
                               '1177', '2210', '1089', '1233', '2212', '2211']):
    """

    :param sub_info: 3*N: 3 for lon, lat, id; N is the number of point
    :param center:
    :param dis0:
    :param site_nos:
    :return:
    """

    # site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968', '1090', '1175', '1177'],
    #                 'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
    if center is not False:
        all_subcenter = np.loadtxt(center, delimiter=',').T # the subcenters
        site_nos = all_subcenter[0].astype(int)
    for site_no in site_nos:
        txt_path = './result_08_01/point/ascat/s%s/' % site_no
        file_list = sorted(os.listdir(txt_path))
        x_time, sigma0, date_list, out_ascat, inc45_55 = [], [], [], [], []
        # 0: num of sub_pixels, 1: 1num of days, 2: num of attributes (-1 was the pixel_id)
        all_subpixel = np.zeros([9, len(file_list)*3, 47]) - 999
        ascat_corner = np.zeros([9, len(file_list)*3, 9]) - 999  # 2: lat*4, lon*4
        corner_headers = ['lat0', 'lat1', 'lat2', 'lat3', 'lon', 'lon', 'lon', 'lon', '1_doy_hr_id']
        if ascat_corner.shape[0] != len(corner_headers):
            print 'the headers for corner data should be updated, from current number of headers: %d to %d' \
                  % (len(corner_headers), ascat_corner.shape[0])
            return -1
        i_day = 0  # the id of days
        i_subpixel_day = np.zeros(9)
        for i, file0 in enumerate(file_list):
            date0 = file0.split('_')[1]
            value0 = np.load(txt_path+file0)
            if value0.size < 46:
                t_now = bxy.current_time()
                t_str = "%d_%d_%d_%d:%d" % (t_now.tm_year, t_now.tm_mon, t_now.tm_mday, t_now.tm_hour, t_now.tm_min)
                bcomand = "echo 'no data recorded in station %s on %s (%s)' 1> missing_data.txt" \
                          % (site_no, date0, t_str)
                os.system(bcomand)
                continue
            date_list.append(date0)
            for i2 in range(0, sub_info.shape[1]):
                lat0i, lon0i = sub_info[1, i2], sub_info[0, i2]
                dis_list0i = bxy.cal_dis(lat0i, lon0i, value0[:, 0], value0[:, 1])  # find the nearest distance
                value0i = value0[np.where(dis_list0i<dis0)]
                i3 = value0i.shape[0]  # number of nearest pixels, i3 = 0--2, (=2 if two tracks overlap)
                all_subpixel[i2, i_subpixel_day[i2]: i_subpixel_day[i2]+i3, 0:-1] = value0i
                if i3>0:
                    # extract pixel info, the coordinates of cornner
                    for i4 in range(0, i3):
                        lat_c, lon_c, u_time, line_no_c, orbit_c = \
                            value0i[i4][0], value0i[i4][1], value0i[i4][14], value0i[i4][15], value0i[i4][-1]
                        if u_time == 506126353.0:
                            pause = 0
                        local_time_c = bxy.time_getlocaltime([u_time], ref_time=[2000, 1, 1, 0])
                        local_hr_c = local_time_c[-1, :]
                        # add pixel id
                        id_daily = 1e6+local_time_c[-2, :]*1e3+local_hr_c*10+i4  # 1xxx(doy)x(nearest_id)
                        if id_daily == 1013201:
                            pause = 1
                        # all_subpixel[i2, i_subpixel_day[i2]: i_subpixel_day[i2]+i4, -1] = id_daily
                        all_subpixel[i2, i_subpixel_day[i2]+i4, -1] = id_daily
                        dis_list_i4 = bxy.cal_dis(lat_c, lon_c, value0[:, 0], value0[:, 1])
                        dis_indx = np.where((dis_list_i4 < 14) & (dis_list_i4 > 0.01) & (value0[:, -1] == orbit_c))
                        nn_pixels = value0[dis_indx]
                        utc_seconds = nn_pixels[:, 14]  # get local time to use the same track (passing time)
                        local_time = bxy.time_getlocaltime(utc_seconds, ref_time=[2000, 1, 1, 0])
                        local_hr = local_time[-1]
                        uni_hr = np.unique(local_hr)  # get all pass hrs (0-2, e.g.: pass at 11:00, 13:00)
                        nn_pixels_track = nn_pixels[local_hr == local_hr_c, :]
                        print 'there is %d pixels around the target, %d of them are in the same track' \
                              % (nn_pixels.shape[0], nn_pixels_track.shape[0])
                        if nn_pixels_track.shape[0] < 4:
                           # olny 3 points nearby was founded
                            azi = value0[i4][16]
                            r = 6370
                            theta, phi = (90-lat_c)/180*np.pi, lon_c/180*np.pi
                            xc, yc, zc = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
                            d = 12.5/2.0
                            # np.array([[xc-d, yc+d], [xc+d, yc+d], [xc+d, yc-d], [xc-d, yc-d]])
                            orig = np.array([[-d, d], [d, d], [d, -d], [-d, -d]])
                            r_matrix = np.array([[np.cos(azi), np.sin(azi)], [-np.sin(azi), np.cos(azi)]])
                            r_xy = np.dot(orig, r_matrix) + np.array([xc, yc])
                            theta0 = np.arctan(np.sqrt(r_xy[:, 0]**2+r_xy[:, 1]**2)/zc)
                            phi0 = np.arctan(r_xy[:, 1]/r_xy[:, 0])
                            lat_corner = 90 - theta0*180/np.pi
                            lon_corner = phi0*180/np.pi
                            for i1, lon1 in enumerate(lon_corner):
                                if lon1 < 60:
                                    lon_corner[i1]-=180
                        else:
                            line_nos = nn_pixels_track[:, 15]
                            i_node1 = line_nos > line_no_c
                            i_node2 = line_nos < line_no_c  # two away pixel center
                            i_node0 = line_nos == line_no_c # two pixels on the same line
                            lat1, lon1 = nn_pixels_track[i_node1, 0], nn_pixels_track[i_node1, 1]
                            lat2, lon2 = nn_pixels_track[i_node2, 0], nn_pixels_track[i_node2, 1]
                            lat_side, lon_side = nn_pixels_track[i_node0, 0], nn_pixels_track[i_node0, 1]
                            lat_corner = 0.5 * np.array([lat1+lat_side[0], lat1+lat_side[1],
                                                         lat2+lat_side[0], lat2+lat_side[1]]).ravel()
                            lon_corner = 0.5 * np.array([lon1+lon_side[0], lon1+lon_side[1],
                                                         lon2+lon_side[0], lon2+lon_side[1]]).ravel()
                        i_seq = np.argsort(lon_corner)
                        i_seq0132 = [i_seq[i] for i in [0, 1, 3, 2]]
                        ascat_corner[i2, i_subpixel_day[i2]+i4, 0:4] = lat_corner[i_seq0132]
                        ascat_corner[i2, i_subpixel_day[i2]+i4, 4:8] = lon_corner[i_seq0132]
                        ascat_corner[i2, i_subpixel_day[i2]+i4, -1] = id_daily
                elif i3>2:
                    pas = 0
            i_subpixel_day[i2] = i_subpixel_day[i2] + i3
        # OUTPUT
        prefix00 = 'result_08_01/point/ascat/time_series'
        fname0 = '%s/ascat_%s_%s_%s_value' % (prefix00, date_list[0], date_list[-1][-4:], site_nos[0])
        fname1 = '%s/ascat_%s_%s_%s_corner' % (prefix00, date_list[0], date_list[-1][-4:], site_nos[0])
        np.save(fname0, all_subpixel)
        np.save(fname1, ascat_corner)
        txtname = '%s/corner_meta0.txt' % prefix00
        np.savetxt(txtname, ascat_corner[0, 1, :], delimiter=',', header=','.join(corner_headers))
        print 'saved at directory: ', prefix00
        return 0

        #id_nn, dis = data_process.ascat_nn(locs.T[1], locs.T[0], sig.T, orb.T, site_no, f=f_u.T, disref=dis0)


        main_sid = site_nos[0]/100
        site_no = str(site_no)
        ascat_point = []
        loc_as = np.array([])  # Location of nn for ascending pass
        x_time, sigma0, date_list, out_ascat, inc45_55 = [], [], [], [], []
        if center is not False:
            i_tb = all_subcenter[2] == float(site_no)
            site_subcenter = all_subcenter[:, i_tb]
            center_tb = [site_subcenter[0], site_subcenter[1]]
            txt_path = '/home/xiyu/PycharmProjects/R3/result_05_01/site_ascat/' + 's' + str(main_sid)+ '/'
        else:
            txt_path = '/home/xiyu/PycharmProjects/R3/result_05_01/site_ascat/' + 's' + str(site_no) + '/'
        n = 0
        n_warning, n_inc = 0, 0
        file_list = sorted(os.listdir(txt_path))
        for txt_file in file_list:
            # get dt
            n+=1
            p_uline = re.compile('_')
            datei = p_uline.split(txt_file)[1]
            txt_i = np.load(txt_path + txt_file)
            if txt_i.size < 5:
                n_warning += 1
                # print 'no data is %s' % datei
                continue
            if txt_i.size > 5:
                date_list.append(datei)
                if len(txt_i.shape) < 2:  # only 1- d
                    locs, sig = np.array([txt_i[0:2]*1]), np.array([txt_i[2:5]*1])
                    f_u, inc = np.array([txt_i[5:8]]), np.array([txt_i[8:11]])
                    orb = np.array([txt_i[-1]])
                elif txt_i.shape[1] > 10:  # with triplets, flag, and inc angles
                    locs, sig = txt_i[:,  0:2], txt_i[:, 2:5]
                    f_u, inc = txt_i[:,  5:8]*1, txt_i[:, 8:11]
                    f_land = txt_i[:, 11:14]
                    t_utc = txt_i[:, 14]
                    orb = txt_i[:, -1]
                # print 'DATE:', datei
                id_nn, dis = data_process.ascat_nn(locs.T[1], locs.T[0], sig.T, orb.T, site_no, f=f_u.T, disref=dis0)
                # test
                as_loc = ascat_test_nn(id_nn[0], id_nn[1], txt_i)
                loc_as = np.append(loc_as, as_loc)
                # pass_hr = 24*np.modf(t_utc/3600/24)[0]
                pass_hr = t_utc
                pass_time = [pass_hr[id_nn[0]], pass_hr[id_nn[1]],
                             [np.fix(t_utc/3600/24 - (dtime(2016, 1, 1) - dtime(2000, 1, 1)).days)],
                             pass_hr]
                items = ['NN as_pass', 'NN des_pass', 'DOY', 'passing hours']
                for i in range(0, len(pass_time)):
                    isprint = False
                    # print items[i], ':', pass_time[i], '\n'
                idn = np.concatenate((id_nn[0], id_nn[1])).astype(int)
                sig_daily = data_process.ascat_ipt(dis, txt_i[:, [2, 3, 4, 8, 9, 10, 11, 12, 13, 0, 1]][idn].T, pass_hr[idn], orb[idn])
                doy = np.fix(t_utc/3600/24 - (dtime(2016, 1, 1) - dtime(2000, 1, 1)).days)[0]
                sig_table = [[doy]+t for t in sig_daily]
                for v in sig_table:
                    ascat_point.append(v)
                tt = 0
        np.save('./result_05_01/ascat_point/ascat_s'+str(site_no)+'_2016', np.array(ascat_point))
        # np.save('./result_05_01/ascat_point/loc_a_s'+site_no+'_2016', loc_as.reshape(-1, 2))
    st = 0
    return 0

def get_grid():
    grp0 = Dataset('/home/xiyu/Data/easegrid2/EASE2_N12.5km.geolocation.v0.9.nc', 'r', format='NETCDF4')
    ease_lat, ease_lon = grp0.variables['latitude'][:], grp0.variables['longitude'][:]
    lat_valid, lon_valid = (ease_lat > 54) & (ease_lat < 72), (ease_lon > -170) & (ease_lon < -130)
    ease_valid = lat_valid & lon_valid
    lat_gd, lon_gd = ease_lat[400: 700, 400: 700], ease_lon[400: 700, 400: 700]
    return  lat_gd, lon_gd


def amsr2_area_resample(attributes, save_path, format='h5', raw_path='', date_str=False, grid_name='north'):
    # load grid
    if grid_name == 'north':
        grid_lon = np.load('/home/xiyu/Data/easegrid2/ease_alaska_north_lon.npy')
        grid_lat = np.load('/home/xiyu/Data/easegrid2/ease_alaska_north_lat.npy')
    else:
        h_t = h5py.File('result_08_01/area/smap_area_result/SMAP_alaska_A_GRID_20151121.h5')
        grid_lon = h_t['cell_lon'].value
        grid_lat = h_t['cell_lat'].value
    # find files
    if date_str is not False:
        for str0 in date_str:
            match0 = 0
    else:
        match0 = '%s/*.h5' % raw_path
        file_list = glob.glob(match0)
    for f0 in file_list:
        h0_un_re = h5py.File(f0)
        if len(h0_un_re.keys()) > 0:
            h5_new ='%s/%s_%s.h5' % (save_path, f0.split('/')[-1].split('.')[0], grid_name)
            lat_land0, lon_land0 = h0_un_re['latitude_36GHz'].value, h0_un_re['longitude_36GHz'].value
            dict_un_re = {}
            for att0 in attributes:
                dict_un_re[att0] = h0_un_re[att0].value
            resampled_ascat0 = res.resample_to_grid(dict_un_re, lon_land0, lat_land0, grid_lon, grid_lat, search_rad=10000)
            h0_re = h5py.File(h5_new, 'a')
            for att0 in attributes:
                if att0 in h0_re.keys():
                    continue
                else:
                    h0_re[att0] = resampled_ascat0[att0]
            h0_un_re.close()
            h0_re.close()
        else:
            continue


def amsr2_detection(res_path='result_08_01/area/amsr2_resample', match_type='alaska', orbit='', odd_plot=False,
                    att=[], extract_point=[], is_plot=False):
    """

    :param res_path:
    :param match_type:
    :param orbit:
    :param odd_plot: dtype: nd_array, the indexes of specific regions (e.g., the north region) if true
    :return:
    """
    # read the resample daily data to get time series
    match0 = ('%s/*' % res_path)+match_type+orbit+'*.h5'
    h5_list = sorted(glob.glob(match0))
    # get the x time
    x_time_array = np.zeros([len(h5_list)])
    for i_day, fname0 in enumerate(h5_list):
        d_str = fname0.split('/')[-1].split('_')[2]
        x_time_array[i_day] = bxy.get_total_sec(fname0.split('/')[-1].split('_')[2])  # try secs
        # bxy.get_secs()
    # initial a 3-d array
    array_3d = np.zeros([len(h5_list), len(att), len(odd_plot)])  # date, attribute, location
    for i_day, h5_file0 in enumerate(h5_list):
        h0 = h5py.File(h5_file0)
        for i_att, att0 in enumerate(att):
            daily_att0_id = h0[att0].value.ravel()  # data of a given att0, daily, 1 dimension
            array_3d[i_day, i_att, :] = daily_att0_id[odd_plot]
        h0.close()
    # index in target region
    # only loops the land area
    if odd_plot is False:
        mask = np.load('./result_05_01/other_product/mask_ease2_360N.npy')
        # onset0 = np.ma.masked_array(onset0, mask=[(onset0==0)|(mask==0)])
        mask_1d = mask.reshape(1, -1)[0]
        land_id = np.where(mask_1d != 0)[0]
    elif type(odd_plot) is list:
        land_id = odd_plot
    else:
        land_id = [odd_plot]
    s=0
    sz_date, sz_att = len(h5_list), len(att)

    amsr2_detection_out = []
    for i0 in land_id:
        pause = 0
        if i0 in extract_point:
            print 'the extracted pixel is', i0
            # check the value of a sepcific point
            # only one pixel is matched
            n3 = np.where(np.array(land_id) == i0)
            array_series = np.zeros([sz_date, sz_att])
            # print 'source: ', array_3d.shape, 'extract: ', array_series.shape
            print 'extract: ', array_series.shape
            for i_att, att0 in enumerate(att):
                temp_v = array_3d[:, att.index(att0), n3]  # land_id==i0
                print 'n3, ', n3
                print 'temp_v: ', temp_v.shape
                print 'received array shape', array_series[:, i_att].shape
                array_series[:, i_att] = temp_v.ravel()*0.01
            amsr2_detection_out.append([i0, x_time_array, array_series])
            if is_plot is True:
                figname = 'result_08_01/amsr2_test_%d.png' % i0
                print 'array_series shape ', array_series.shape
                a0, a2 = 0, 2
                y_label = [att0.split('(')[-1] for att0 in [att[a0], att[a2]]]
                plot_funcs.plot_subplot([[x_time_array, array_series[:, a0]], [x_time_array, array_series[:, a2]]], [],
                                        text='test01', main_label=[y_label[0], y_label[1]],
                                        figname=figname, x_unit='doy')

    return amsr2_detection_out  # the last interested point
