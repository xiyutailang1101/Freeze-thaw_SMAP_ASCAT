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


def smap_mask():
    # read ascat land mask
    ascat_mask = np.load('./result_05_01/other_product/mask_ease2_125N.npy')
    ascat_lat, ascat_lon = np.load('lat_ease_grid.npy'), np.load('lon_ease_grid.npy')
    smap_lat, smap_lon = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy'), \
                         np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
    v = np.zeros(ascat_mask.shape, dtype=int)
    v[ascat_mask] = 1
    v_dict = {'mask': v[ascat_mask]}
    smap_mask_dict = res.resample_to_grid(v_dict, ascat_lon[ascat_mask], ascat_lat[ascat_mask], smap_lon, smap_lat)
    np.save('./result_05_01/other_product/mask_ease2_360N', smap_mask_dict['mask'].data)

    # test the mask
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
    v = np.loadtxt('/home/xiyu/Data/Snow/perennial05.txt')
    mask_vu, lon_mask0, lat_mask0 = v[:, -1], v[:, -2], v[:, -3]
    snow_id = mask_vu == 12
    mask_v, lon_mask0, lat_mask0 = mask_vu[snow_id], lon_mask0[snow_id], lat_mask0[snow_id]
    mask0 = {'snow': mask_v}
    if mode=='sig':
        grp0 = Dataset('/home/xiyu/Data/easegrid2/EASE2_N12.5km.geolocation.v0.9.nc', 'r', format='NETCDF4')
        ease_lat, ease_lon = grp0.variables['latitude'][:], grp0.variables['longitude'][:]
        lat_gd, lon_gd = ease_lat[400: 700, 400: 700], ease_lon[400: 700, 400: 700]
        resampled_ascat0 = res.resample_to_grid(mask0, lon_mask0, lat_mask0, lon_gd, lat_gd, search_rad=9000)
        data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat0['snow'], '/home/xiyu/Data/Snow/', fname='snow_mask01', z_min=0, z_max=15)
        np.save('./result_05_01/other_product/snow_mask_125', resampled_ascat0['snow'].data)
    elif mode=='npr':
        lon_gd = np.load('./result_05_01/other_product/lon_ease2_360N_grid.npy')
        lat_gd = np.load('./result_05_01/other_product/lat_ease2_360N_grid.npy')
        resampled_ascat0 = res.resample_to_grid(mask0, lon_mask0, lat_mask0, lon_gd, lat_gd, search_rad=20000)
        np.save('./result_05_01/other_product/snow_mask_360', resampled_ascat0['snow'].data)
        data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat0['snow'], '/home/xiyu/Data/Snow/', fname='snow_mask01', z_min=0, z_max=15)

def smap_area_plot(datez, save_dir='./result_05_01/smap_area_result/', orbit_no=0):
    ob = 'AS'
    smap_data = np.load('./result_05_01/SMAP_AK/txt_xyz'+datez+'.npy')
    datez = datez[0:4] + datez[5:7] + datez[8:10]
    pre_path = save_dir+datez+'/'
    if not os.path.exists(save_dir+datez):
        os.makedirs(save_dir+datez)
    # hf0 = h5py.File('/home/xiyu/Data/easegrid2/SMAP_L3_SM_P_20160717_R14010_001.h5', 'r')
    # ease_lat_un = hf0['Soil_Moisture_Retrieval_Data_AM']['latitude'].value
    # ease_lat = np.ma.masked_array(data=ease_lat_un, mask=[ease_lat_un == -9999])
    # ease_lon_un = hf0['Soil_Moisture_Retrieval_Data_AM']['longitude'].value
    # ease_lon = np.ma.masked_array(data=ease_lon_un, mask=[ease_lon_un == -9999])
    ease_lat_un = np.fromfile('/home/xiyu/Data/easegrid2/gridloc.EASE2_N36km/EASE2_N36km.lats.500x500x1.double', dtype=float).reshape(500, 500)
    ease_lon_un = np.fromfile('/home/xiyu/Data/easegrid2/gridloc.EASE2_N36km/EASE2_N36km.lons.500x500x1.double', dtype=float).reshape(500, 500)
    ease_lat = np.ma.masked_array(data=ease_lat_un, mask=[ease_lat_un == -999])
    ease_lon = np.ma.masked_array(data=ease_lon_un, mask=[ease_lon_un == -999])
    # row, col = data_process.bbox_find(ease_lat, ease_lon, [72, 30], [-170, -100])
    # lat_gd, lon_gd = ease_lat[min(row): max(row), min(col): max(col)], ease_lon[min(row): max(row), min(col): max(col)]
    ease_valid = ((ease_lat > 54) & (ease_lat < 72)&(ease_lon > -170) & (ease_lon < -130))
    # fig2 = plt.figure(figsize=[8, 8])
    # ax2 = fig2.add_subplot(111)
    # # m2 = Basemap(llcrnrlon=-175, llcrnrlat=30, urcrnrlon=-70, urcrnrlat=72,
    # #                 resolution='l', area_thresh=1000., projection='merc',
    # #                 lat_ts=30., ax=ax2)
    # m2 = Basemap(width=10e6, height=6e6, resolution='l', projection='laea', lat_ts=62, lat_0=52, lon_0=-160., ax=ax2)
    # # ease_lon[ease_lon<-900] = 0
    # # ease_lat[ease_lat<-900] = 0
    # m2.scatter(ease_lon[ease_valid], ease_lat[ease_valid], 1, marker='*', color='k', latlon=True)
    # # m2.scatter(ease_lon[~ease_lon.mask], ease_lat[~ease_lat.mask], 1, marker='.', color='k', latlon=True)
    # # plt.colorbar(sc)
    # m2.drawcoastlines()
    # plt.savefig(save_dir+'spatial_smap_grid.png', dpi=120)
    # plt.close()

    tb_v0, tb_h0, lat_tb0, lon_tb0 = smap_data[:, 2], smap_data[:, 3], smap_data[:, 1], smap_data[:, 0]
    if lat_tb0.size < 1:
        print 'no data in date %s' % datez
    elif (lat_tb0 == -1).all():
        print '-1 data in date %s' % datez
    elif (lat_tb0 > 10).any():
        tb_h, tb_v = tb_h0[tb_h0 != -1], tb_v0[tb_v0 != -1]
        lon_tb, lat_tb = lon_tb0[lon_tb0 != -1], lat_tb0[[lat_tb0 != -1]]
        smap_dict0 = {'tbv': tb_v, 'tbh': tb_h}
        grid_lon, grid_lat = ease_lon[ease_valid], ease_lat[ease_valid]
        resampled_smap0 = res.resample_to_grid(smap_dict0, lon_tb, lat_tb, ease_lon[~ease_lon.mask], ease_lat[~ease_lat.mask], search_rad=27000)
        # resampled_smap0 = res.resample_to_grid(smap_dict0, lon_tb, lat_tb, grid_lon[~grid_lon.mask], grid_lat[~grid_lat.mask], search_rad=27000)
        v_map_un = np.zeros(ease_lon.shape)
        for key in ['tbv', 'tbh']:
            v_map_un[~ease_lon.mask] = resampled_smap0[key].data
            v_map = np.ma.masked_array(v_map_un, mask=[(v_map_un==-9999)|(v_map_un==0)])
            tp0 = [[140, 219], [160, 249]]
            # data_process.pass_zone_plot(ease_lon[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1],
            #                             ease_lat[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1],
            #                             v_map[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1]
            #                             , pre_path, fname='smap_'+key+'_'+datez, z_min=185, z_max=285)
            ease_y = ease_lat[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1]
            ease_x = ease_lon[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1]
            np.save('./result_05_01/other_product/lat_ease2_360N_grid', ease_y.data)
            np.save('./result_05_01/other_product/lon_ease2_360N_grid', ease_x.data)
            np.save('./result_05_01/smap_resample_'+ob+'/smap_'+datez+'_'+key+'_resample', v_map[min(tp0[0]): max(tp0[0])+1, min(tp0[1]): max(tp0[1])+1].data)


def ascat_area_plot2(datez, save_dir='./result_05_01/test_area_result_9km_', orbit_no=0):
    if orbit_no == 0: # AS
        ob = 'AS'
        tzone = [[18, 19.5], [19.5, 21], [21, 22.5], [22.5, 24]]
    else:
        ob = 'DES'
        tzone = [[2.5, 4], [4, 5.5], [5.5, 7], [7, 8.5], [8.5, 10]]
    save_dir = save_dir+ob+'/'
    # initialize a grid system
    if not os.path.exists(save_dir+datez):
        os.makedirs(save_dir+datez)
    pre_path = save_dir+datez+'/'
    grp0 = Dataset('/home/xiyu/Data/easegrid2/EASE2_N12.5km.geolocation.v0.9.nc', 'r', format='NETCDF4')
    ease_lat, ease_lon = grp0.variables['latitude'][:], grp0.variables['longitude'][:]
    lat_valid, lon_valid = (ease_lat > 54) & (ease_lat < 72), (ease_lon > -170) & (ease_lon < -130)
    ease_valid = lat_valid & lon_valid
    lat_gd, lon_gd = ease_lat[400: 700, 400: 700], ease_lon[400: 700, 400: 700]
    np.save('lat_ease_grid', lat_gd.data), np.save('lon_ease_grid', lon_gd.data)
    lats_dim, lons_dim = np.arange(54, 72, 0.1), np.arange(-170, -130, 0.1)
    lons_grid, lats_grid = np.meshgrid(lons_dim, lats_dim)

    # read the ascat, constraint by land% > 50%, orbit, and usable flag
    ascat_data = np.load('./result_05_01/ASCAT_AK/ascat_'+datez+'_alaska.npy')
    sigma_mid, land_mid, inc_mid, orbit_id, f_use = ascat_data[:, 3], ascat_data[:, 12], ascat_data[:, 9], ascat_data[:, -1], ascat_data[:, 6]
    id_land_orb = (land_mid > 0.5) & (orbit_id == orbit_no) & (f_use < 2)
    sigma_land, inc_land = sigma_mid[id_land_orb], inc_mid[id_land_orb]
    lon_land, lat_land, passtime = ascat_data[:, 1][id_land_orb], ascat_data[:, 0][id_land_orb], ascat_data[:, 14][id_land_orb]
    pass_hr = 24 * ((passtime/3600/24) - (dtime2.strptime(datez, "%Y%m%d").date() - dtime(2000, 1, 1)).days)
    print np.min(pass_hr)


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

    id_loc_time00 = data_process.time_mosaic(tzone, pass_hr)
    n0 = -1
    n1 = -1
    for tz in tzone:
        n0 += 1
        id_loc_time0 = (pass_hr > tz[0]) & (pass_hr < tz[1])
        if sum(id_loc_time0) < 1:  # no data interval
            continue
        else:
            n1 += 1
            sigma_land0, inc_land0, pass_hr0 = sigma_land[id_loc_time0], inc_land[id_loc_time0], pass_hr[id_loc_time0]
            lon_land0, lat_land0 = lon_land[id_loc_time0], lat_land[id_loc_time0]
            sigma_dict0 = {'sigma': sigma_land0, 'incidence': inc_land0, 'pass': pass_hr0}
            if n1 < 1:
                resampled_ascat0 = res.resample_to_grid(sigma_dict0, lon_land0, lat_land0, lon_gd, lat_gd, search_rad=9000)
                # data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat0['sigma'], pre_path, fname='_zone'+str(tzone[n0][0]*10))
            else:
                resampled_ascat_tp = res.resample_to_grid(sigma_dict0, lon_land0, lat_land0, lon_gd, lat_gd, search_rad=9000)
                #data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat_tp['sigma'], pre_path, fname='_zone'+str(tzone[n0][0]*10))
                # get the mosaic part
                id_same_mask = resampled_ascat0['sigma'].mask == resampled_ascat_tp['sigma'].mask
                id_mosaic = id_same_mask == resampled_ascat_tp['sigma'].mask
                # the intersect part of unmasked area
                id_overlay2 = (resampled_ascat0['sigma'].mask == False) & (resampled_ascat_tp['sigma'].mask == False)
                # resampled_ascat0['sigma'].mask[id_overlay2] = True
                ## data_process.pass_zone_plot(lons_grid, lats_grid, resampled_ascat0['sigma'], pre_path, fname='_overlay'+str(tzone[n0][0]*10))
                for key in ['sigma', 'incidence']:
                    resampled_ascat0[key][id_mosaic] = resampled_ascat_tp[key][id_mosaic]
                    resampled_ascat0[key].mask[id_mosaic] = resampled_ascat_tp[key].mask[id_mosaic]
                    if all(sum(id_overlay2) < 1):
                        print 'No overlay occurs from hour %.1f to %.1f' % (tz[0], tz[1])
                    else:
                        resampled_ascat0[key][id_overlay2] = 0.5*(resampled_ascat0[key][id_overlay2] + resampled_ascat_tp[key][id_overlay2])
                    # data_process.pass_zone_plot(lon_gd, lat_gd, resampled_ascat0['sigma'], pre_path, fname='_zone'+str(tzone[n0][0]*10)+'added')
    # apply normalization
    if n1 < 0:
        print 'No morning data in %s' % datez
        return 0
    # save not normalized data
    np.save('./result_05_01/ascat_resample_'+ob+'/ascat_'+datez+'_resample', resampled_ascat0['sigma'].data)
    np.save('./result_05_01/ascat_resample_'+ob+'/ascat_'+datez+'_incidence', resampled_ascat0['incidence'].data)
    resampled_ascat0['sigma'] -= (resampled_ascat0['incidence']-45.0)*-0.11  # normalization
    np.save('./result_05_01/ascat_resample_norms/ascat_resample_'+ob+'/ascat_'+datez+'_resample', resampled_ascat0['sigma'].data)
    np.save('./result_05_01/ascat_resample_'+ob+'/ascat_'+datez+'_mask', resampled_ascat0['sigma'].mask)
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


def ascat_point_plot():
    site_nos = ['947', '2081', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '1177', '2210', '1089', '1233', '2212', '2211']
    site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968', '1090', '1175', '1177'],
                    'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
    for site_no in site_nos:
        ascat_point = []
        loc_as = np.array([])  # Location of nn for ascending pass
        x_time, sigma0, date_list, out_ascat, inc45_55 = [], [], [], [], []
        txt_path = '/home/xiyu/PycharmProjects/R3/result_05_01/site_ascat/' + 's' + site_no + '/'
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
                print 'no data is %s' % datei
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
                print 'DATE:', datei
                id_nn, dis = data_process.ascat_nn(locs.T[1], locs.T[0], sig.T, orb.T, site_no, disref=0, f=f_u.T, incidence=inc.T)
                # test
                as_loc = ascat_test_nn(id_nn[0], id_nn[1], txt_i)
                loc_as = np.append(loc_as, as_loc)
                pass_hr = 24*np.modf(t_utc/3600/24)[0]
                pass_time = [pass_hr[id_nn[0]], pass_hr[id_nn[1]],
                             [np.fix(t_utc/3600/24 - (dtime(2016, 1, 1) - dtime(2000, 1, 1)).days)],
                             pass_hr]
                items = ['NN as_pass', 'NN des_pass', 'DOY', 'passing hours']
                # for i in range(0, len(pass_time)):
                #     print items[i], ':', pass_time[i], '\n'
                idn = np.concatenate((id_nn[0], id_nn[1])).astype(int)
                sig_daily = data_process.ascat_ipt(dis, txt_i[:, [2, 3, 4, 8, 9, 10, 11, 12, 13]][idn].T, pass_hr[idn], orb[idn])
                doy = np.fix(t_utc/3600/24 - (dtime(2016, 1, 1) - dtime(2000, 1, 1)).days)[0]
                sig_table = [[doy]+t for t in sig_daily]
                for v in sig_table:
                    ascat_point.append(v)
                tt = 0
        np.save('./result_05_01/ascat_point/ascat_s'+site_no+'_2016', np.array(ascat_point))
        np.save('./result_05_01/ascat_point/loc_a_s'+site_no+'_2016', loc_as.reshape(-1, 2))
    st = 0
    return 0


# def time_mosaic(intervals, pass_h):
#     id_time = []
#     for interv in intervals:
#         id_time.append((pass_hr > interv[0]) & (pass_hr < interv[1]))
#     return id_time


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