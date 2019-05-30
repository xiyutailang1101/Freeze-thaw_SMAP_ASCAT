"""
DESCRIPTION:
    Read data from SMAP product
"""
import time
import h5py
import numpy as np
import time
import datetime
import os, re, sys
import log_write
import matplotlib.pyplot as plt
import site_infos
import data_process
import read_site
import gdal
from gdalconst import *
from netCDF4 import Dataset
import basic_xiyu as bs
import test_def
import glob
import basic_xiyu as bxy

# def initlog():
#     import logging
#     logger = logging.getLogger()
#     LOG_FILE = 'debug.log'
#     hdlr = logging.FileHandler(LOG_FILE)
#     formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#     hdlr.setFormatter(formatter)
#     logger.addHandler(hdlr)
#     logger.setLevel(logging.NOTSET)
#     return logger


def return_region(filenamez, areas, sensor, prj='North_Polar_Projection'):
    return 0


def return_ind(filenamez, siten, sensor, prj='North_Polar_Projection',thsig=[0.4, 0.4], thtb=[1, 1], orbz=1, fname=None,
               center=False, atts=[], cube=1):
    """
    return the index of site value in the hdf5 files, which is array with element(s)
    :param filenamez:
    :param siten:
    :param sensor: ascat: save as .npy form.
    :param prj:
    :return:
    """
    # return the index of site value in the hdf5 files, which is array with element(s)
    # sensor :'sigma', 'TB', 'other'
      # input 1, file name
      # input 2, location
    if sensor == 'sigma':
        lat_t = siten[1]
        lon_t = siten[2]
        hf = h5py.File(filenamez, 'r')
        dset_lon = np.array(hf['Sigma0_Data/cell_lon'])  # ##
        dset_lon.shape = -1,
        dset_lat = np.array(hf['Sigma0_Data/cell_lat'])  # ##
        dset_lat.shape = -1,
        lon_ind = np.where(np.abs(dset_lon[...] - lon_t) < thsig[0])
        lat_ind = np.where(np.abs(dset_lat[...] - lat_t) < thsig[1])
        region_ind = np.intersect1d(lon_ind, lat_ind)
        # dis = (dset_lat[region_ind] - lat_t)**2 + (dset_lon[region_ind] - lon_t)**2
        # if np.any(dis):
        #     region_ind = region_ind[np.argmin(dis)]
        print 'the radar index of this region is'
        print region_ind, dset_lat[region_ind], dset_lon[region_ind]
        return region_ind, dset_lat[region_ind], dset_lon[region_ind]
        logging1.info(region_ind)
    elif sensor == 'tbs':  # TBs, return only 1 pixels
        # initialize
        i_dic = 0
        site_dict = {site_no0: dict() for site_no0 in siten}
        lon_dic = {site_no0: [] for site_no0 in siten}
        # a dict for each site, keys of the dict is the same with the smap_h5
        for f0 in filenamez:
            i_dic += 1
            hf = h5py.File(f0, 'r')
            dset_lon = np.array(hf[prj + '/cell_lon'])
            dset_lat = np.array(hf[prj + '/cell_lat'])
            for site_no0 in site_dict.keys():
                ki = 0
                #  hf_site = h5py.File('./h5_l1c/smap_'+site_no0[0]+'.h5', 'a')
                site_info0 = site_infos.change_site(site_no0)
                lat_t, lon_t = site_info0[1], site_info0[2]
                loc_ind = np.where((np.abs(dset_lon[...] - lon_t) < thtb[0])&(np.abs(dset_lat[...] - lat_t) < thtb[1]))
                lon_dic[site_no0].append(loc_ind)
                if loc_ind[0].size > 0:
                    #read_tb_site(hf, site_dict[site_no0], loc_ind, prj)
                    att_keys = hf[prj].keys()
                    for key in att_keys:
                        if ki<1:
                            site_dict[site_no0][prj+'/'+key] = (hf[prj+'/'+key].value[loc_ind])
                        else:
                            site_dict[site_no0][prj+'/'+key] = np.concatenate((site_dict[site_no0][prj+'/'+key], hf[prj+'/'+key].value[loc_ind]))
                    ki += 1
        return lon_dic, site_dict
    elif sensor == 'tbak':
        lat_t = siten[1]
        lon_t = siten[2]
        hf = h5py.File(filenamez, 'r')
        dset_lon = np.array(hf[prj + '/cell_lon'])
        dset_lat = np.array(hf[prj + '/cell_lat'])
        lon_ind = np.where(np.abs(dset_lon[...] - lon_t) < thtb[0])
        lat_ind = np.where(np.abs(dset_lat[...] - lat_t) < thtb[1])
        region_ind = np.intersect1d(lon_ind, lat_ind)
        # print '============', lon_t, lat_t, lon_ind, lat_ind, region_ind
        return region_ind, 1
    elif sensor == 'grid':
        # initialize
        i_dic = 0
        ki = 0
        site_dict = {site_no0: dict() for site_no0 in siten}
        lon_dic = {site_no0: [] for site_no0 in siten}
        # a dict for each site, keys of the dict is the same with the smap_h5
        for f0 in filenamez:
            i_dic += 1
            hf = h5py.File(f0, 'r')
            dset_col = np.array(hf[prj + '/cell_column'])
            dset_row = np.array(hf[prj + '/cell_row'])
            for site_no0 in site_dict.keys():
                #  hf_site = h5py.File('./h5_l1c/smap_'+site_no0[0]+'.h5', 'a')
                site_info0 = site_infos.change_site(site_no0)
                lat_t, lon_t = site_info0[1], site_info0[2]
                loc_ind = np.where((dset_col>159) & (dset_col<251) & (dset_row>139) & (dset_row<221))
                lon_dic[site_no0].append(loc_ind)
                if loc_ind[0].size > 0:
                    #read_tb_site(hf, site_dict[site_no0], loc_ind, prj)
                    att_keys = hf[prj].keys()
                    for key in att_keys:
                        if ki<1:
                            site_dict[site_no0][prj+'/'+key] = hf[prj+'/'+key].value[loc_ind]
                        else:
                            site_dict[site_no0][prj+'/'+key] = np.concatenate((site_dict[site_no0][prj+'/'+key],
                                                                               hf[prj+'/'+key].value[loc_ind]))
                    ki += 1
            hf.close()
        return lon_dic, site_dict

    elif sensor == 'amsr2':
         # initialize
        i_dic = 0
        site_dict = {site_no0: dict() for site_no0 in siten}
        lon_dic = {site_no0: np.array([]) for site_no0 in siten}
        lat_dic = {site_no0: np.array([]) for site_no0 in siten}
        ki = 0
        for f0 in filenamez:
            i_dic += 1
            hf = h5py.File(f0, 'r')
            dset_lat89 = hf['Latitude of Observation Point for 89A'].value
            dset_lon89 = hf['Longitude of Observation Point for 89A'].value
            ind_36 = np.arange(0, dset_lat89.shape[1], 2)
            dset_lat = dset_lat89[:, ind_36]
            dset_lon = dset_lon89[:, ind_36]

            for site_no0 in site_dict.keys():
                #  hf_site = h5py.File('./h5_l1c/smap_'+site_no0[0]+'.h5', 'a')
                site_info0 = site_infos.change_site(site_no0)
                lat_t, lon_t = site_info0[1], site_info0[2]
                loc_ind = np.where((np.abs(dset_lon[...] - lon_t) < thtb[0])&(np.abs(dset_lat[...] - lat_t) < thtb[1]))
                # h5_newfile = 'AMSR2_l2r_%s_%s_%s.h5' % (datestr, site0, orb)
                if loc_ind[0].size > 0:
                    lon_dic[site_no0] = np.concatenate((lon_dic[site_no0], dset_lon[loc_ind]))
                    lat_dic[site_no0] = np.concatenate((lat_dic[site_no0], dset_lat[loc_ind]))
                    #read_tb_site(hf, site_dict[site_no0], loc_ind, prj)
                    att_keys = hf.keys()
                    for key in att_keys:
                        # TB dataset
                        if 'Brightness Temperature' in key:
                            if key in site_dict[site_no0].keys():
                                site_dict[site_no0][key] = np.concatenate((site_dict[site_no0][key], hf[key].value[loc_ind]))
                            else:
                                # the first call
                                if key in ['Brightness Temperature (res06,10.7GHz,H)',
                                       'Brightness Temperature (res06,10.7GHz,V)',
                                        'Brightness Temperature (res06,18.7GHz,H)',
                                        'Brightness Temperature (res06,18.7GHz,V)',
                                        'Brightness Temperature (res06,23.8GHz,H)',
                                        'Brightness Temperature (res06,23.8GHz,V)',
                                        'Brightness Temperature (res06,36.5GHz,H)',
                                        'Brightness Temperature (res06,36.5GHz,V)',
                                        'Brightness Temperature (res06,6.9GHz,H)',
                                        'Brightness Temperature (res06,6.9GHz,V)',
                                        'Brightness Temperature (res06,89.0GHz,H)',
                                        'Brightness Temperature (res06,89.0GHz,V)',
                                        'Brightness Temperature (res23,18.7GHz,V)',
                                        'Brightness Temperature (res23,36.5GHz,V)',
                                        'Brightness Temperature (res23,18.7GHz,H)',
                                        'Brightness Temperature (res23,36.5GHz,H)']:
                                    site_dict[site_no0][key] = hf[key].value[loc_ind]
                        # non-TB data set  # size
                        elif key in ['Earth Azimuth',  # (2040, 243)
                                    'Earth Incidence',  # (2040, 243)
                                    'Land_Ocean Flag 6 to 36',  #(4, 2040, 243)
                                    'Land_Ocean Flag 89',  #(2, 2040, 486)
                                    'Latitude of Observation Point for 89A',  # (2040, 486)
                                    'Latitude of Observation Point for 89B',  # (2040 , 486)
                                    'Longitude of Observation Point for 89A',  # (2040, 486)
                                    'Longitude of Observation Point for 89B',  # (2040, 486)
                                    # 'Navigation Data',  # (2040, 6)  # useless
                                    'Pixel Data Quality 6 to 36',  # (2040, 486)
                                    'Pixel Data Quality 89',  # (2040, 486)
                                    'Position in Orbit',  # (2040,)
                                    'Scan Data Quality',  # (2040, 512)
                                    'Scan Time',  # (2040,)
                                    'Sun Azimuth', # (2040, 243)
                                    'Sun Elevation']:  #(2040, 243)]:
                            dims = len(hf[key].value.shape)
                            if key in site_dict[site_no0].keys():
                                if dims == 3:
                                    site_dict[site_no0][key] = np.concatenate((site_dict[site_no0][key], hf[key].value[:, loc_ind[0], loc_ind[1]].ravel()))
                                elif dims == 2:
                                    site_dict[site_no0][key] = np.concatenate((site_dict[site_no0][key], hf[key].value[loc_ind]))
                                elif dims == 1:
                                    site_dict[site_no0][key] = np.concatenate((site_dict[site_no0][key], hf[key].value[loc_ind[0]]))
                            else:
                                if dims == 3:
                                    site_dict[site_no0][key] = hf[key].value[:, loc_ind[0], loc_ind[1]].ravel()
                                elif dims == 2:
                                    site_dict[site_no0][key] = hf[key].value[loc_ind]
                                elif dims == 1:
                                    print key
                                    print hf[key].value.shape
                                    print dims
                                    site_dict[site_no0][key] = hf[key].value[loc_ind[0]]
                        ki += 1
            hf.close()
        return [lon_dic, lat_dic], site_dict

        # return_ind(filenamez, siten, sensor, prj='North_Polar_Projection',thsig=[0.4, 0.4], thtb=[1, 1], orbz=1, fname=None, center=False)
    elif sensor in ['ease_n25', 'erq_25', 'EQMA', 'EQMD']:
        # read amsr2_l3 global data
        # find rows and cols of interested pixles
        pause = 0
        site_dict = {}
        site_name_int = []
        m_no = len(filenamez)  # the total number of data, including the unvalid
        row_list, col_list = [], []
        # alaska region 63.25, -151.5 lat: 8.58, lon: 17.54
        for site in siten:
            site_dict[site] = np.zeros([m_no, 3]) - 999
            if site == 'alaska':
                print 'read amsr2 l3 data in whole AK'
                site_name_int = 0  # 0 for AK
                # all rows/cols of pixel
            else:
                site_name_int.append(int(site))  # save the site name into a list
                s_info = site_infos.change_site(site)
                grid_info0 = site_infos.grid_info(sensor)
                grid_size = grid_info0['resolution']
                row_no = (grid_info0['lat_ul'] - s_info[1])/grid_size
                col_no = (180 + s_info[2] - grid_info0['lon_ul'])/grid_size
                row_list.append(int(round(row_no)))
                col_list.append(int(round(col_no)))
        pixel_info = np.array([site_name_int, row_list, col_list])  # name, row, col of interested pixels

        # read measurements from h5 file
        daily_temp = np.zeros([len(atts), len(row_list), cube]) - 999  # 2 dimensions: attributes & pixels
        # total_value dimensions: ('date', 'atts', 'location', 'variables') e.g., variable: SND & SWE
        total_value = np.zeros([len(filenamez), len(atts), len(row_list), cube]) - 999
        date_sec = np.zeros(len(filenamez)) - 999  # saving the date str
        for i_f0, f0 in enumerate(filenamez):  # sort the file list
            #  file name example: GW1AM2_20160102_01D_PNMD_L3SGSNDLG2210210.h5
            time_str = f0.split('/')[-1].split('_')[1]
            time_sec = bxy.get_total_sec(time_str)
            date_sec[i_f0] = time_sec
        time_order = np.argsort(date_sec)
        date_sec = date_sec[time_order]
        filenamez_sorted = []
        for i0 in time_order:
            filenamez_sorted.append(filenamez[i0])
        for i_f0, f0 in enumerate(filenamez_sorted):
            hf_in = h5py.File(f0, 'r')
            for i_att, att0 in enumerate(atts):
                temp_value = hf_in[att0].value[row_list, col_list].reshape(len(row_list), -1)
                n00 = temp_value.size/temp_value.shape[0]
                daily_temp[i_att, :, 0:n00] = temp_value
            total_value[i_f0, :, :, :] = daily_temp
            hf_in.close()
        return pixel_info, total_value, date_sec

    elif sensor == 'ascat':
        # read all ascat swath in one date
        # nc_dict = read_netcdf(filenamez, ['latitude', 'longitude', 'sigma0_trip', 'f_usable', 'inc_angle_trip', 'f_land', 'as_des_pass'], anx=None)
        # nc_dict = read_netcdf(filenamez, ['latitude', 'longitude', 'sigma0_trip', 'f_usable', 'inc_angle_trip', 'f_land', 'utc_line_nodes', 'abs_line_number', 'as_des_pass'], anx=None)
        # headers = site_infos[sensor]
        headers = site_infos.ascat_heads(sensor)
        nc_dict = read_netcdf(filenamez, headers, anx=None)
        # check_ak_ascat(nc_dict)
        # sys.exit()
        for site in siten:
            info = site_infos.change_site(site)
            lat_t = info[1]
            lon_t = info[2]
            if center is not False:
                center_tb = test_def.main(site, [], sm_wind=7, mode='annual', seriestype='tb', tbob='_A_', sig0=7, center=True)
                lat_t = center_tb[1]
                lon_t = center_tb[0]
            table = np.array([])
            ind_count = 0
            for file_no in range(0, len(nc_dict['latitude'])):
                region_ind = np.where((np.abs(nc_dict['latitude'][file_no] - lat_t) < thsig[0]) & (np.abs(nc_dict['longitude'][file_no] - lon_t) < thsig[1]))
                if np.any(region_ind[0]):  # region_ind 2-d coordinate
                    band_no = region_ind[0]
                    node_no = region_ind[1]
                    # statck them all
                    stacks = np.array([nc_dict['latitude'][file_no][region_ind], nc_dict['longitude'][file_no][region_ind]])
                    col_no = 0
                    for key00 in headers:
                        with open('meta_ascat_ak.txt', 'a') as meta0:
                            incrm = 1 if nc_dict[key00][file_no].ndim < 3 else 3
                            meta0.write('%s, %d, %d \n' % (key00, col_no, col_no+incrm))
                            col_no+=incrm
                        if key00 == 'latitude' or key00 == 'longitude':
                            continue
                        if nc_dict[key00][file_no].ndim==3:
                            for i in range(0, 3):
                                #stacks = np.vstack([stacks, nc_dict[keyi][file_no][(band_no, node_no, i)]])
                                stacks = np.append(stacks, [nc_dict[key00][file_no][(band_no, node_no, i)]], axis=0)
                        elif nc_dict[key00][file_no].ndim == 2:
                            key_indicator = nc_dict[key00][file_no][band_no, node_no]
                            stacks=np.append(stacks, [key_indicator], axis=0)
                        elif nc_dict[key00][file_no].ndim == 1:
                            key_indicator = nc_dict[key00][file_no][band_no]
                            stacks=np.append(stacks, [key_indicator], axis=0)

                    # # statcks: attributes read in alaska region
                    # stacks = np.array([nc_dict['latitude'][file_no][region_ind], nc_dict['longitude'][file_no][region_ind]])  # 2
                    # for keyi in ['sigma0_trip', 'f_usable', 'inc_angle_trip', 'f_land']:  # 3+3+3+3
                    #     for i in range(0, 3):
                    #         #stacks = np.vstack([stacks, nc_dict[keyi][file_no][(band_no, node_no, i)]])
                    #         stacks = np.append(stacks, [nc_dict[keyi][file_no][(band_no, node_no, i)]], axis=0)
                    # # stack the orbit, and passing time
                    # for keyi in ['utc_line_nodes', 'abs_line_number', 'as_des_pass']:  # 1+1+1
                    #     key_indicator = nc_dict[keyi][file_no][band_no]
                    #     stacks=np.append(stacks, [key_indicator], axis=0)
                    if ind_count < 1:
                        ind_count += 1
                        # table is the each stack concatenate together along rows
                        table = stacks
                    else:
                        table = np.append(table, stacks, axis=1)
        # save this site
            if table.size < 12:
                np.save(fname+'_'+site+'.npy', np.transpose(table))
                continue
            np.save(fname+'_'+site+'.npy', np.transpose(table))
                       #fmt='%.6f, %.6f, %.6f, %.6f, %.6f, %d, %d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %d')


                #else:

                    # for i in range(0, region_ind[0].size, 1):
                    #     #sig_tp = o_sig.GetRasterBand(region_ind[0][i]+1).ReadAsArray()
                    #     for j in range(0, 3):
                    #         sig0[j].append(sig_tp[band_no[i], node_no[i], j])  #
                    #     #sig_tp = None
                    # sigma0 = sig0
                    # center_lat = dset_lat[region_ind]
                    # center_lon = dset_lon[region_ind]
            #hf.close()
            #return [np.array([[sig_tp.shape[0]-1], [sig_tp.shape[1]-1]])-region_ind, sigma0], center_lat, center_lon, orb_indicator, 1
    else:
            #hf.close()
        return -1, -1, -1, -1, -1  # fail to read data


def return_ind2(filenamez, siten, sensor, prj='North_Polar_Projection',thsig=[0.4, 0.4], thtb=[1, 1], orbz=1, fname=None, center=False):
    """
    return the index of site value in the hdf5 files, which is array with element(s)
    :param filenamez:
    :param siten:
    :param sensor:
    :param prj:
    :return:
    """
    # return the index of site value in the hdf5 files, which is array with element(s)
    # sensor :'sigma', 'TB', 'other'
      # input 1, file name
      # input 2, location
    if sensor == 'sigma':
        lat_t = siten[1]
        lon_t = siten[2]
        hf = h5py.File(filenamez, 'r')
        dset_lon = np.array(hf['Sigma0_Data/cell_lon'])  # ##
        dset_lon.shape = -1,
        dset_lat = np.array(hf['Sigma0_Data/cell_lat'])  # ##
        dset_lat.shape = -1,
        lon_ind = np.where(np.abs(dset_lon[...] - lon_t) < thsig[0])
        lat_ind = np.where(np.abs(dset_lat[...] - lat_t) < thsig[1])
        region_ind = np.intersect1d(lon_ind, lat_ind)
        # dis = (dset_lat[region_ind] - lat_t)**2 + (dset_lon[region_ind] - lon_t)**2
        # if np.any(dis):
        #     region_ind = region_ind[np.argmin(dis)]
        print 'the radar index of this region is'
        print region_ind, dset_lat[region_ind], dset_lon[region_ind]
        return region_ind, dset_lat[region_ind], dset_lon[region_ind]
        logging1.info(region_ind)
    elif sensor == 'tbs':  # TBs, return only 1 pixels
        # initialize
        i_dic = 0
        site_dict = {site_no0: dict() for site_no0 in siten}
        lon_dic = {site_no0: [] for site_no0 in siten}
        # a dict for each site, keys of the dict is the same with the smap_h5
        for f0 in filenamez:
            i_dic += 1
            hf = h5py.File(f0, 'r')
            dset_lon = np.array(hf[prj + '/cell_lon'])
            dset_lat = np.array(hf[prj + '/cell_lat'])
            for site_no0 in site_dict.keys():
                ki = 0
                #  hf_site = h5py.File('./h5_l1c/smap_'+site_no0[0]+'.h5', 'a')
                site_info0 = site_infos.change_site(site_no0)
                lat_t, lon_t = site_info0[1], site_info0[2]
                loc_ind = np.where((np.abs(dset_lon[...] - lon_t) < thtb[0])&(np.abs(dset_lat[...] - lat_t) < thtb[1]))
                lon_dic[site_no0].append(loc_ind)
                if loc_ind[0].size > 0:
                    #read_tb_site(hf, site_dict[site_no0], loc_ind, prj)
                    att_keys = hf[prj].keys()
                    for key in att_keys:
                        if ki<1:
                            site_dict[site_no0][prj+'/'+key] = (hf[prj+'/'+key].value[loc_ind])
                        else:
                            site_dict[site_no0][prj+'/'+key] = np.concatenate((site_dict[site_no0][prj+'/'+key], hf[prj+'/'+key].value[loc_ind]))
                    ki += 1
        return lon_dic, site_dict
    elif sensor == 'tbak':
        lat_t = siten[1]
        lon_t = siten[2]
        hf = h5py.File(filenamez, 'r')
        dset_lon = np.array(hf[prj + '/cell_lon'])
        dset_lat = np.array(hf[prj + '/cell_lat'])
        lon_ind = np.where(np.abs(dset_lon[...] - lon_t) < thtb[0])
        lat_ind = np.where(np.abs(dset_lat[...] - lat_t) < thtb[1])
        region_ind = np.intersect1d(lon_ind, lat_ind)
        # print '============', lon_t, lat_t, lon_ind, lat_ind, region_ind
        return region_ind, 1
    elif sensor == 'grid':
        # initialize
        i_dic = 0
        ki = 0
        site_dict = {site_no0: dict() for site_no0 in siten}
        lon_dic = {site_no0: [] for site_no0 in siten}
        # a dict for each site, keys of the dict is the same with the smap_h5
        for f0 in filenamez:
            i_dic += 1
            hf = h5py.File(f0, 'r')
            dset_col = np.array(hf[prj + '/cell_column'])
            dset_row = np.array(hf[prj + '/cell_row'])
            for site_no0 in site_dict.keys():
                #  hf_site = h5py.File('./h5_l1c/smap_'+site_no0[0]+'.h5', 'a')
                site_info0 = site_infos.change_site(site_no0)
                lat_t, lon_t = site_info0[1], site_info0[2]
                loc_ind = np.where((dset_col>159) & (dset_col<251) & (dset_row>139) & (dset_row<221))
                lon_dic[site_no0].append(loc_ind)
                if loc_ind[0].size > 0:
                    #read_tb_site(hf, site_dict[site_no0], loc_ind, prj)
                    att_keys = hf[prj].keys()
                    for key in att_keys:
                        if ki<1:
                            site_dict[site_no0][prj+'/'+key] = hf[prj+'/'+key].value[loc_ind]
                        else:
                            site_dict[site_no0][prj+'/'+key] = np.concatenate((site_dict[site_no0][prj+'/'+key],
                                                                               hf[prj+'/'+key].value[loc_ind]))
                    ki += 1
            hf.close()
        return lon_dic, site_dict

    elif sensor == 'amsr2':
         # initialize
        i_dic = 0
        site_dict = {site_no0: dict() for site_no0 in siten}
        lon_dic = {site_no0: np.array([]) for site_no0 in siten}
        lat_dic = {site_no0: np.array([]) for site_no0 in siten}
        ki = 0
        for f0 in filenamez:
            i_dic += 1
            hf = h5py.File(f0, 'r')
            dset_lat89 = hf['Latitude of Observation Point for 89A'].value
            dset_lon89 = hf['Longitude of Observation Point for 89A'].value
            ind_36 = np.arange(0, dset_lat89.shape[1], 2)
            dset_lat = dset_lat89[:, ind_36]
            dset_lon = dset_lon89[:, ind_36]

            for site_no0 in site_dict.keys():
                #  hf_site = h5py.File('./h5_l1c/smap_'+site_no0[0]+'.h5', 'a')
                site_info0 = site_infos.change_site(site_no0)
                lat_t, lon_t = site_info0[1], site_info0[2]
                loc_ind = np.where((np.abs(dset_lon[...] - lon_t) < thtb[0])&(np.abs(dset_lat[...] - lat_t) < thtb[1]))
                # h5_newfile = 'AMSR2_l2r_%s_%s_%s.h5' % (datestr, site0, orb)
                if loc_ind[0].size > 0:
                    lon_dic[site_no0] = np.concatenate((lon_dic[site_no0], dset_lon[loc_ind]))
                    lat_dic[site_no0] = np.concatenate((lat_dic[site_no0], dset_lat[loc_ind]))
                    #read_tb_site(hf, site_dict[site_no0], loc_ind, prj)
                    att_keys = hf.keys()
                    for key in att_keys:
                        # TB dataset
                        if 'Brightness Temperature' in key:
                            if key in site_dict[site_no0].keys():
                                site_dict[site_no0][key] = np.concatenate((site_dict[site_no0][key], hf[key].value[loc_ind]))
                            else:
                                # the first call
                                if key in ['Brightness Temperature (res06,10.7GHz,H)',
                                       'Brightness Temperature (res06,10.7GHz,V)',
                                        'Brightness Temperature (res06,18.7GHz,H)',
                                        'Brightness Temperature (res06,18.7GHz,V)',
                                        'Brightness Temperature (res06,23.8GHz,H)',
                                        'Brightness Temperature (res06,23.8GHz,V)',
                                        'Brightness Temperature (res06,36.5GHz,H)',
                                        'Brightness Temperature (res06,36.5GHz,V)',
                                        'Brightness Temperature (res06,6.9GHz,H)',
                                        'Brightness Temperature (res06,6.9GHz,V)',
                                        'Brightness Temperature (res06,89.0GHz,H)',
                                        'Brightness Temperature (res06,89.0GHz,V)',
                                        'Brightness Temperature (res23,18.7GHz,V)',
                                        'Brightness Temperature (res23,36.5GHz,V)',
                                        'Brightness Temperature (res23,18.7GHz,H)',
                                        'Brightness Temperature (res23,36.5GHz,H)']:
                                    site_dict[site_no0][key] = hf[key].value[loc_ind]
                        # non-TB data set  # size
                        elif key in ['Earth Azimuth',  # (2040, 243)
                                    'Earth Incidence',  # (2040, 243)
                                    'Land_Ocean Flag 6 to 36',  #(4, 2040, 243)
                                    'Land_Ocean Flag 89',  #(2, 2040, 486)
                                    'Latitude of Observation Point for 89A',  # (2040, 486)
                                    'Latitude of Observation Point for 89B',  # (2040 , 486)
                                    'Longitude of Observation Point for 89A',  # (2040, 486)
                                    'Longitude of Observation Point for 89B',  # (2040, 486)
                                    # 'Navigation Data',  # (2040, 6)  # useless
                                    'Pixel Data Quality 6 to 36',  # (2040, 486)
                                    'Pixel Data Quality 89',  # (2040, 486)
                                    'Position in Orbit',  # (2040,)
                                    'Scan Data Quality',  # (2040, 512)
                                    'Scan Time',  # (2040,)
                                    'Sun Azimuth', # (2040, 243)
                                    'Sun Elevation']:  #(2040, 243)]:
                            dims = len(hf[key].value.shape)
                            if key in site_dict[site_no0].keys():
                                if dims == 3:
                                    site_dict[site_no0][key] = np.concatenate((site_dict[site_no0][key], hf[key].value[:, loc_ind[0], loc_ind[1]].ravel()))
                                elif dims == 2:
                                    site_dict[site_no0][key] = np.concatenate((site_dict[site_no0][key], hf[key].value[loc_ind]))
                                elif dims == 1:
                                    site_dict[site_no0][key] = np.concatenate((site_dict[site_no0][key], hf[key].value[loc_ind[0]]))
                            else:
                                if dims == 3:
                                    site_dict[site_no0][key] = hf[key].value[:, loc_ind[0], loc_ind[1]].ravel()
                                elif dims == 2:
                                    site_dict[site_no0][key] = hf[key].value[loc_ind]
                                elif dims == 1:
                                    print key
                                    print hf[key].value.shape
                                    print dims
                                    site_dict[site_no0][key] = hf[key].value[loc_ind[0]]
                        ki += 1
            hf.close()
        return [lon_dic, lat_dic], site_dict
    elif sensor in ['ease_n25', 'erq_25']:
        # read amsr2_l3 global data
        # find rows and cols of interested pixles
        pause = 0
        atts = []
        site_dict = {}
        site_name_int = []
        m_no = len(filenamez)  # the total number of data, including the unvalid
        row_list, col_list = [], []
        # alaska region 63.25, -151.5 lat: 8.58, lon: 17.54
        for site in siten:
            site_dict[site] = np.zeros([m_no, 3]) - 999
            if site == 'alaska':
                print 'read amsr2 l3 data in whole AK'
                site_name_int = 0  # 0 for AK
                # all rows/cols of pixel
            else:
                site_name_int.append(int(site))  # save the site name into a list
                s_info = site_infos.change_site(site)
                grid_info0 = site_infos.grid_info(sensor)
                grid_size = grid_info0['resolution']
                row_no = (grid_info0['lat_ul'] - s_info[1])/grid_size
                col_no = (180 + s_info[2] - grid_info0['lon_ul'])/grid_size
                row_list.append(row_no)
                col_list.append(col_no)
        pixel_info = np.array([site_name_int, row_list, col_list])  # name, row, col of interested pixels

        # read measurements from h5 file
        daily_temp = np.zeros([len(atts), len(row_list)]) - 999  # 2 dimensions: attributes & pixels
        total_value = np.zeros([len(filenamez), len(atts), len(row_list)]) - 999  # dates, attributes, pixels
        for i_f0, f0 in enumerate(filenamez):
            hf_in = h5py.File(f0, 'r')
            for i_att, att0 in enumerate(atts):
                daily_temp[i_att] = hf_in[att0].value()[row_list, col_list]  # array shape: 1 * N, N is no. of pixels
            total_value[i_f0] = daily_temp
            hf_in.close()
        return pixel_info, total_value

        # return_ind(filenamez, siten, sensor, prj='North_Polar_Projection',thsig=[0.4, 0.4], thtb=[1, 1], orbz=1, fname=None, center=False)

    elif sensor == 'ascat':
        # read all ascat swath in one date
        # nc_dict = read_netcdf(filenamez, ['latitude', 'longitude', 'sigma0_trip', 'f_usable', 'inc_angle_trip', 'f_land', 'as_des_pass'], anx=None)
        # nc_dict = read_netcdf(filenamez, ['latitude', 'longitude', 'sigma0_trip', 'f_usable', 'inc_angle_trip', 'f_land', 'utc_line_nodes', 'abs_line_number', 'as_des_pass'], anx=None)
        # headers = site_infos[sensor]
        headers = site_infos.ascat_heads(sensor)
        nc_dict = read_netcdf(filenamez, headers, anx=None)
        # check_ak_ascat(nc_dict)
        # sys.exit()
        for site in siten:
            info = site_infos.change_site(site)
            lat_t = info[1]
            lon_t = info[2]
            if center is not False:
                center_tb = test_def.main(site, [], sm_wind=7, mode='annual', seriestype='tb', tbob='_A_', sig0=7, center=True)
                lat_t = center_tb[1]
                lon_t = center_tb[0]
            table = np.array([])
            ind_count = 0
            for file_no in range(0, len(nc_dict['latitude'])):
                region_ind = np.where((np.abs(nc_dict['latitude'][file_no] - lat_t) < thsig[0]) & (np.abs(nc_dict['longitude'][file_no] - lon_t) < thsig[1]))
                if np.any(region_ind[0]):  # region_ind 2-d coordinate
                    band_no = region_ind[0]
                    node_no = region_ind[1]
                    # statck them all
                    stacks = np.array([nc_dict['latitude'][file_no][region_ind], nc_dict['longitude'][file_no][region_ind]])
                    col_no = 0
                    for key00 in headers:
                        with open('meta_ascat_ak.txt', 'a') as meta0:
                            incrm = 1 if nc_dict[key00][file_no].ndim < 3 else 3
                            meta0.write('%s, %d, %d \n' % (key00, col_no, col_no+incrm))
                            col_no+=incrm
                        if key00 == 'latitude' or key00 == 'longitude':
                            continue
                        if nc_dict[key00][file_no].ndim==3:
                            for i in range(0, 3):
                                #stacks = np.vstack([stacks, nc_dict[keyi][file_no][(band_no, node_no, i)]])
                                stacks = np.append(stacks, [nc_dict[key00][file_no][(band_no, node_no, i)]], axis=0)
                        elif nc_dict[key00][file_no].ndim == 2:
                            key_indicator = nc_dict[key00][file_no][band_no, node_no]
                            stacks=np.append(stacks, [key_indicator], axis=0)
                        elif nc_dict[key00][file_no].ndim == 1:
                            key_indicator = nc_dict[key00][file_no][band_no]
                            stacks=np.append(stacks, [key_indicator], axis=0)

                    # # statcks: attributes read in alaska region
                    # stacks = np.array([nc_dict['latitude'][file_no][region_ind], nc_dict['longitude'][file_no][region_ind]])  # 2
                    # for keyi in ['sigma0_trip', 'f_usable', 'inc_angle_trip', 'f_land']:  # 3+3+3+3
                    #     for i in range(0, 3):
                    #         #stacks = np.vstack([stacks, nc_dict[keyi][file_no][(band_no, node_no, i)]])
                    #         stacks = np.append(stacks, [nc_dict[keyi][file_no][(band_no, node_no, i)]], axis=0)
                    # # stack the orbit, and passing time
                    # for keyi in ['utc_line_nodes', 'abs_line_number', 'as_des_pass']:  # 1+1+1
                    #     key_indicator = nc_dict[keyi][file_no][band_no]
                    #     stacks=np.append(stacks, [key_indicator], axis=0)
                    if ind_count < 1:
                        ind_count += 1
                        # table is the each stack concatenate together along rows
                        table = stacks
                    else:
                        table = np.append(table, stacks, axis=1)
        # save this site
            if table.size < 12:
                np.save(fname+'_'+site+'.npy', np.transpose(table))
                continue
            np.save(fname+'_'+site+'.npy', np.transpose(table))
                       #fmt='%.6f, %.6f, %.6f, %.6f, %.6f, %d, %d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %d')


                #else:

                    # for i in range(0, region_ind[0].size, 1):
                    #     #sig_tp = o_sig.GetRasterBand(region_ind[0][i]+1).ReadAsArray()
                    #     for j in range(0, 3):
                    #         sig0[j].append(sig_tp[band_no[i], node_no[i], j])  #
                    #     #sig_tp = None
                    # sigma0 = sig0
                    # center_lat = dset_lat[region_ind]
                    # center_lon = dset_lon[region_ind]
            #hf.close()
            #return [np.array([[sig_tp.shape[0]-1], [sig_tp.shape[1]-1]])-region_ind, sigma0], center_lat, center_lon, orb_indicator, 1
    else:
            #hf.close()
        return -1, -1, -1, -1, -1  # fail to read data


def read_nc(file, att, anx=None):
    nc_dict = dict()
    #
    for var in att:  # initial the output as dict, key is the att name.
        nc_dict[var] = []
    for f in file:
        for keyi in nc_dict.keys():  # value for each key
            o_key = gdal.Open(f+':'+keyi)
            v_key = o_key.ReadAsArray()
            nc_dict[keyi].append(v_key)
            o_key = None
            v_key = None
    # orbit
    if anx is not None:
        for keyi in anx:
            nc_dict[keyi] = []
            for f in file:
                ob_path = '/media/327A50047A4FC379/ASCAT/orbits/'
                p_slash = re.compile('/')
                ncfile = p_slash.split(f)[5]
                obfile = ob_path+ncfile[: -2]+'txt'
                obs = np.loadtxt(obfile, delimiter=',')
                nc_dict['orbit'].append(obs)
    return nc_dict


def read_netcdf(file, att, anx=None):
    """
    file contain all (n) files available per day, function returns a dictionary file, each key of the dict refers to a
    n elements list, each element refers data of one nc file.
    :param file:
    :param att: the list contains the variables of nc file.
    :param anx:
    :return:
    """
    nc_dict = dict()
    #
    for var in att:  # initial the output as dict, key is the att name.
        nc_dict[var] = []
    for f in file:
        rootgrp = Dataset(f, 'r', format='NETCDF4')
        for keyi in nc_dict.keys():  # value for each key
            #o_key = gdal.Open(f+':'+keyi)
            net_key = rootgrp.variables.keys()
            # check key latitude
            if keyi == 'longitude':
                pause = 0
            if (keyi == 'longitude') & (keyi not in net_key):
                v_key = rootgrp.variables['lon'][:]
            elif (keyi == 'latitude') & (keyi not in net_key):
                v_key = rootgrp.variables['lat'][:]
            else:
                v_key = rootgrp.variables[keyi][:]
            nc_dict[keyi].append(v_key)
            o_key = None
            v_key = None
        rootgrp.close()
    # orbit
    if anx is not None:
        for keyi in anx:
            nc_dict[keyi] = []
            for f in file:
                ob_path = '/media/327A50047A4FC379/ASCAT/orbits/'
                p_slash = re.compile('/')
                ncfile = p_slash.split(f)[5]
                obfile = ob_path+ncfile[: -2]+'txt'
                obs = np.loadtxt(obfile, delimiter=',')
                nc_dict['orbit'].append(obs)
    return nc_dict


def read_series(indices, filename):  # filename = root + files
    if np.any(indices):
        hf = h5py.File(filename, 'r')
        region_sig0 = np.array(hf['Sigma0_Data/cell_sigma0_vv_aft'])
        hf.close()
        region_sig0.shape = -1,
        # print(region_sig0.shape)
        temp_sig0 = region_sig0[indices]
        sig0 = np.mean(temp_sig0)
        return sig0
    else:
        return 0
    # print("++++++++++++++++reading++++++++++++++++++")


def read_region(indices, filename, scales):  # gridsize 1 is coarser
    if np.any(indices != -1):
        hf = h5py.File(filename, 'r')
        sig0 = np.array(hf['Sigma0_Data/cell_sigma0_vv_aft'])
        lat = np.array(hf['Sigma0_Data/cell_lat'])
        lon = np.array(hf['Sigma0_Data/cell_lon'])
        hf.close()
        [row, col] = ind2xy(indices, sig0.shape[1])
        region_sig = sig0[row - 0.5 * scales: row + 0.5 * scales, col - 0.5 * scales: col + 0.5 * scales]
        region_lat = lat[row - 0.5 * scales: row + 0.5 * scales, col - 0.5 * scales: col + 0.5 * scales]
        region_lon = lon[row - 0.5 * scales: row + 0.5 * scales, col - 0.5 * scales: col + 0.5 * scales]
    return region_sig, region_lat, region_lon


def read_region2(filename, corners, field):
        # corner is the lower left and up right of pixels
        # filename is the h5 file for finer resolution data, e.g., radar 36 *36
        # field includes the attributes of h5 files:
        # sigma0 for back scatter coefficient, lat for latitude, lon for longitude.
        if np.any(corners != -1):
            hf2 = h5py.File(filename, 'r')
            value = np.array(hf2[field[0]])
            lat = np.array(hf2[field[1]])
            lon = np.array(hf2[field[2]])
            hf2.close()
            value.shape = -1,
            lat.shape = -1,
            lon.shape = -1,
            # sig0_list, lat_list, lon_list = np.array([]), np.array([]), np.array([])
            # for i in range(0, corners[0].size):
            # lat_ind = np.where(corners[0][0] < lat < corners[1][0])
            print 'the corner coordinate is'
            print(corners)
            logging1.info(corners)
            lat_ind = np.where(np.logical_and(lat < corners[1][0], lat > corners[0][0]))
            print 'LAT INDEX IS'
            print(lat_ind)
            lat_ind2 = lat_ind
            lon_ind = np.where(np.logical_and(lon < corners[1][1], lon > corners[0][1]))
            print 'the longitude region is', corners[1][1], corners[0][1]
            print lon_ind
            print('*******************')
            logging1.info('*******************')
            # print(lon_ind)
            lon_ind2 = lon_ind
            # print(lon_ind2)
            # lon_ind = np.where(corners[0][1] < lon < corners[1][1])
            region_ind = np.intersect1d(lon_ind2, lat_ind2)
            # print(region_ind.shape)
            print('there is %d * 36 pixels were picked' % int(np.size(region_ind)/36))
            sig0_list, lat_list, lon_list = value[region_ind], lat[region_ind], lon[region_ind]
            print('the shape and type of output region data are')
            print sig0_list.shape, type(sig0_list)
            print np.min(lat_list), 'to', np.max(lat_list), np.max(lon_list), 'to', np.min(lon_list)
            logging1.info('there is %d * 36 pixels were picked' % int(np.size(region_ind)/36))
            logging1.info('the shape and type of output region data are')
            logging1.info(sig0_list.shape)
            return sig0_list, lat_list, lon_list
        else:
            return -1, -1, -1


def read_region3(filename, corners, field, prj='n'):
        """
        Only return indices
        :param filename:
        :param corners:
        :param field:
        :return:
        """
        # corner is the lower left and up right of pixels
        # filename is the h5 file for finer resolution data, e.g., radar 36 *36
        # field includes the attributes of h5 files:
        # sigma0 for back scatter coefficient, lat for latitude, lon for longitude.
        if np.any(corners != -1):
            hf2 = h5py.File(filename, 'r')
            lat = np.array(hf2[field[0]])
            lon = np.array(hf2[field[1]])
            hf2.close()
            lat.shape = -1,
            lon.shape = -1,
                # sig0_list, lat_list, lon_list = np.array([]), np.array([]), np.array([])
                # for i in range(0, corners[0].size):
                # lat_ind = np.where(corners[0][0] < lat < corners[1][0])
            if prj == 'gm':
                print 'the corner coordinate is'
                print(corners)
                logging1.info(corners)
                lat_ind = np.where(np.logical_and(lat < corners[1][0], lat > corners[0][0]))
                print 'LAT INDEX IS'
                print(lat_ind)
                lat_ind2 = lat_ind
                lon_ind = np.where(np.logical_and(lon < corners[1][1], lon > corners[0][1]))
                print 'the longitude region is', corners[1][1], corners[0][1]
                print lon_ind
                print('*******************')
                logging1.info('*******************')
                # print(lon_ind)
                lon_ind2 = lon_ind
                # print(lon_ind2)
                # lon_ind = np.where(corners[0][1] < lon < corners[1][1])
                region_ind = np.intersect1d(lon_ind2, lat_ind2)
                # print(region_ind.shape)
                print('there is %d * 36 pixels were picked' % int(np.size(region_ind)/36))
                logging1.info('there is %d * 36 pixels were picked' % int(np.size(region_ind)/36))
                return 1, region_ind
            elif prj == 'n':
                #  corners contains [distance, [lat, lon]]
                dis = np.sqrt((lat - corners[1][0]) * (lat - corners[1][0]) + (lon - corners[1][1]) * (lon - corners[1][1]))
                region_ind = np.where(dis < 1)
                return 1, region_ind
        else:
            return -1, -1


def ind2xy(ind, xsize):
    col = ind % xsize
    row = np.int(ind/xsize)
    return row, col


def find_nearest(lat_array, lon_array, target_loc, prj='n', id='947', fig_name='test'):
    """
    N projection (default): select the nearest pixel of site, calculate the dis. between it and site
    :param
        <lat_array: lon_array: > ~ of pixels near the sites
    :param
        <target_loc:> lat lon of site
    :param fig_name:
    :return:
    """
    dis = (lat_array - target_loc[0])**2 + (lon_array - target_loc[1])**2
    near_ind = np.argmin(dis)
    near_lat, near_lon = lat_array[near_ind], lon_array[near_ind]  # retrieve the nearest pixel
    lat_uni = np.unique(np.sort(lat_array))
    lon_uni = np.unique(np.sort(lon_array))
    lat_ind = np.where(lat_uni == near_lat)
    lon_ind = np.where(lon_uni == near_lon)
    if prj == 'gm':
        if lat_ind[0] == lat_uni.size - 1:
            interval0 = abs(near_lat - lat_uni[lat_ind[0] - 1])/2
        else:
            interval0 = abs(near_lat - lat_uni[lat_ind[0] + 1])/2

        if lon_ind[0] == lon_uni.size - 1:
            interval1 = abs(near_lon - lon_uni[lon_ind[0] - 1])/2
        else:
            interval1 = abs(near_lon - lon_uni[lon_ind[0] + 1])/2
        if lat_uni.size == 1:
            interval0 = 0.33443
        if lon_uni.size == 1:
            interval1 = 0.18672
    elif prj == 'n':  # need to find the way approaches right or left
        refer_dis = 0.1
        if lat_ind[0] == lat_uni.size - 1:  # only one latitude
            lat2 = lat_uni[lat_ind[0] - 1]
            lon2 = lon_array[np.where(lat_array == lat2)]
            dis2 = ((near_lat-lat2)**2 + (near_lon - lon2)**2)/4
        else:
            lat2 = lat_uni[lat_ind[0] + 1]
            lon2 = lon_array[np.where(lat_array == lat2)]
            dis2 = ((near_lat-lat2)**2 + (near_lon - lon2)**2)/4
        if lon_ind[0] == lon_uni.size - 1:
            lon3 = lon_uni[lon_ind[0] - 1]
            lon3 = lon_array[np.where(lat_array == lat2)]
            dis3 = ((near_lat-lat2)**2 + (near_lon - lon2)**2)/4
        else:
            lon3 = lon_uni[lon_ind[0] + 1]
            lat3 = lat_array[lon_array == lon3]
            dis3 = ((near_lat-lat3)**2 + (near_lon - lon3)**2)/4
        if lat_uni.size == 1 or lon_uni.size == 1:
            dis_n = 0.062
        else:
            refer_dis = 0.1
    # modify the coordinate if only single lat/lon


    # view the location
    fig, ax1 = plt.subplots()
    ax1.scatter(lon_array, lat_array)
    ax1.plot(near_lon, near_lat, 'ro')
    if prj == 'gm':
        corners = [[near_lat - interval0, near_lon - interval1], [near_lat + interval0, near_lon + interval1]]  # ll, ur
        ax1.plot(corners[0][1], corners[0][0], 'r*')
        ax1.plot(corners[1][1], corners[1][0], 'r*')
        ax1.plot(target_loc[1], target_loc[0], 'b*')
        # plt.savefig('test_nearest' + prj + id + '.png', dpi=120)
        plt.savefig(fig_name  + '.png', dpi=120)
        plt.close()
            # cal the minimum of dis, check the latitude and longitude of selected tb pixel
        mindis = dis[near_ind]
        refer_dis = interval0**2 + interval1**2
        if mindis > refer_dis:
            print 'site was not within a pixel'
            return -1, corners
        else:
            return 1, corners
    elif prj == 'n':
        '''
        try to draw a circle
        '''
        ax1.plot(target_loc[1], target_loc[0], 'b*')
        # plt.savefig('test_nearest' + prj + id + '.png', dpi=120)
        plt.savefig(fig_name + '.png', dpi=120)
        plt.close()
        real_dis = (near_lat - target_loc[0])**2 + (near_lon - target_loc[1])**2
        if real_dis > refer_dis:
            print 'site was out of pixel'
            return -1, refer_dis
        else:
            return 1, [refer_dis, target_loc]
    #  retrieve the corner coordinates
    # d_id = np.linspace(0, lat_array.size - 1, lat_array.size)
    # d_table = np.zeros(lat_array.size, dtype=[('id', int), ('lat', float), ('lon', float)])
    # d_table['id'] = d_id
    # d_table['lat'] = lat_array
    # d_table['lon'] = lon_array
    # lat_sort = np.sort(d_table, order='lat')
    # lon_sort = np.sort(d_table, order='lon')
    # lat_uni = np.unique(lat_sort['lat'])
    # lon_uni = np.unique(lon_sort['lon'])
    # print 'selected tb pixels in a region of: '
    # print lat_uni, lon_uni
    # lat_ind = np.argwhere(lat_uni == near_lat)
    # lon_ind = np.argwhere(lon_uni == near_lon)
    # ll = [(lat_uni[lat_ind - 1] + lat_uni[lat_ind])/2, (lon_uni[lon_ind - 1] + lon_uni[lon_ind])/2]
    # ur = [(lat_uni[lat_ind + 1] + lat_uni[lat_ind])/2, (lon_uni[lon_ind + 1] + lon_uni[lon_ind])/2]
    # corners = [ll, ur]


def change_site(site_no):
    site_list = [['947', 65.12422, -146.73390], ['948', 65.25113, -146.15133],
['949', 65.07833, -145.87067], ['950', 64.85033, -146.20945],
['1090', 65.36710, -146.59200], ['960', 65.48, -145.42]]
    site_no_list = [row[0] for row in site_list]
    siteinfo = site_list[site_no_list.index(site_no)]
    return siteinfo


def read_tb(h5_in, attr_name, inds, prj='North_Polar_Projection'):
    """
    Read tb based on index, then give the val to h5_in file with specified attribute name.
    :param h5_in:
    :param attr_name: the name of one attribute that we assign value
    :param inds:
    :param h5_out:
    :return:
        ndarray type tb
        -1: no pixels was found
    """
    values = []  # what is the type of inds?
    if inds.size:  # inds is empty if data are not found in this h5 files
        values_np = h5_in[prj + attr_name].value[inds]
        return values_np, 1
    else:
        print 'can not read tb data'
        return -1, -1


def read_sig(h5_in, attr_name, inds):
    values = []  # what is the type of inds?
    if np.any(inds != -1):  # -1 means not found with specified lat and lon
        for index in inds:
            values.append(h5_in[attr_name][index])
        values_np = np.array(values)
        return values_np, 1
    else:
        print 'can not read tb data'
        return -1, -1


def save_radar(ind, h5_in, field, ini):
    hf_in = h5py.File(h5_in, 'r')
    save_array = np.zeros([len(field), ind.size])
    n = 0
    for att in field:
        value = np.array(hf_in[att])
        value.shape = -1,
        save_array[n, :] = value[ind]
        n += 1
    if ini.size > 0:
        save_array = np.append(ini, save_array, axis=1)
    hf_in.close()
    return save_array



logging1 = log_write.initlog('test1')


def radar_read_main(orbit, site_number, peroid, polars, r_range=False):
    """
    Radiometer saving form:
        Based on date, each date contains all the global orbits and numbers of observing time
    Radar saving form:
        All files within study area saved in one document, specified based on date.
    Data Selection:
        Based on date, generating a list of date from radar file, iterating each date in the list, searching corresponding TB file.
    Radar area data:
        Obtain the corners of TB pixel, using the corners to obtain Radar data.
    orbit: _A_ or _D_
    polars: vv, vh, hh
    """
    n = 1
    # orbit = '_D_'

    rootdir = '/media/Seagate Expansion Drive/Data_Xy/CloudFilm/'  # directory: sigma
    TBdir = '/media/327A50047A4FC379/SMAP/SPL1CTB.003/'  # directory: TB, listed by date document
    TB_date_list = sorted(os.listdir(TBdir))
    TB_date = []
    TB_date2, TB_ind = [], -1  # use to find yyyy.mm.dd format documents
    # Time list of radar data
    i = 0
    tb_attr_name = ['/cell_tb_v_aft', '/cell_tb_h_aft', '/cell_tb_qual_flag_v_aft', '/cell_tb_qual_flag_h_aft',
                    "/cell_lat", "/cell_lon"]  # /cell_lon here is for tb cell

    # list of additional tb data
    time_ini = 0
    for time_dot in TB_date_list:
        if time_dot == peroid[0]:
            time_ini = 1
        elif time_dot == peroid[1]:
            time_ini = 0
        if time_ini > 0:
            TB_date.append(time_dot.replace(".", ""))
            TB_date2.append(time_dot)
    for time_str in TB_date:  # loop day by day
        TB_ind += 1
    # for time_str in ['20150421', '20150422']:
        print('At the date of %s and %s' % (time_str, TB_date2[TB_ind]))
        logging1.info('At the date of %s' % time_str)
        '''
        ********************************************************************************************************************
    1\\\Find the TB data filename list and read the data around the station,
        for each site the radius in degrees is set as 1 degree, several pixels around the Site will be found.
        1a. Generate the file list;
        1b. Read each file in the list.
        ********************************************************************************************************************
        '''
        # 01 list with radiometer files
        tb_file_list = []
        tb_ob_list = []
        tb_folder = TB_date2[TB_ind]
        print 'the order of tb_file is :', tb_folder
        for tbs in os.listdir(TBdir + tb_folder):
            if tbs.find('.iso') == -1 and tbs.find(orbit) != -1 and tbs.find('.qa') == -1:  # Using ascend orbit data, iso and is metadata we don't need
                tb_file_list.append(tbs)
                p_underline = re.compile('_')
                tb_ob_list.append(p_underline.split(tbs)[3])
        obset = set(tb_ob_list)
        tb_name_list = []
        for orb_no in obset:  # ob_set is a set of orbit numbers
            list_orb = []
            for tb_file in tb_file_list:  # tb_file_list is a list of all tb files (Line 175)
                if re.search(orb_no, tb_file):
                    list_orb.append(tb_file)
            if len(list_orb) > 1:
                looks = sorted([p_underline.split(lkt)[-1] for lkt in list_orb])
                last_look = looks[-1]
                for lk in list_orb:
                    if lk.find(last_look) != -1:
                        tb_name_list.append(lk)
            else:
                tb_name_list.append(sorted(list_orb)[-1])

        # 02 read data from list created in 01
        proj_g = 'Global_Projection'
        proj_n = 'North_Polar_Projection'
        count_tb = 0  # remains 1 if no tb file was found
        d_att = {'_key': 'dict for global tb'}  # create dictionary for store tb data temporally
        for proj in [proj_n]:  # two projection method:
            count_tb = 0
            for namez in tb_attr_name:
                d_att[namez] = np.array([])
            tb_full_path = [TBdir + tb_folder + '/' + tb_name for tb_name in tb_name_list]
            # fname = TBdir + tb_folder + '/' + tb_name
            print site_number
            if r_range is not False:
                site_ind, hdf5_slice = return_ind(tb_full_path, site_number, 'tbs', prj=proj, thtb=r_range)
            else:
                site_ind, hdf5_slice = return_ind(tb_full_path, site_number, 'tbs', prj=proj)
            for sn in hdf5_slice.keys():
                h5_path = '%s/SMAP_%s%s%s.h5' % (pre_path, sn, orbit, time_str)
                hf0 = h5py.File(h5_path, 'a')
                for key in hdf5_slice[sn]:
                    hf0[key] = hdf5_slice[sn][key]
                hf0.close()
            continue
            # print 'site index in tb is: ', Site_ind


            # tb1, status = read_tb(hf, '/cell_tb_v_aft', Site_ind, prj=proj)  # !! [not used] check with the global projection
            # tb2, status2 = read_tb(hf, '/cell_tb_v_aft', Site_ind_n, prj=proj)  # [not used] with north pole projection
            if status1 == -1:  # -1 means not found with specified lat and lon
                a00 = 1
            else:
                count_tb = 1
                print 'the station is inside of this swath: ', tb_name
                # hf3 = h5py.File("Site_"+site_info[0] + orbit + time_str + ".h5", "a")  # new a hdf5 file when the site was found
                # new_h5_name = hf3.filename
                hf = h5py.File(fname, 'r')
                for att in tb_attr_name:
                    tb_att, status = read_tb(hf, att, Site_ind, prj=proj)
                    d_att[att] = np.append(d_att[att], tb_att)
                        # hf3[proj_g + att] = tb_att
                    # hf3["Global_Projection/site_loc"] = [site_info[1], site_info[2]]
                    # hf3[proj_g + "/tb_cell_lat"] = c_lat_g
                    # hf3[proj_g + "/tb_cell_lon"] = c_lon_g
                    # hf3.close()

                # if status2 == -1:
                #     print tb_name
                # else:
                #     if status == -1 and status2 == 1:  # find in northpole prj but not in global prj
                #         hf3 = h5py.File("Site_"+site_info[0] + orbit + time_str + ".h5", "a")
                #         hf3["Global_Projection/site_loc"] = [site_info[1], site_info[2]]
                #         new_h5_name = hf3.filename
                #     hf3 = h5py.File(new_h5_name, "a")  # new a hdf5 file
                #     for att in tb_attr_name:
                #         tb_att, status = read_tb(hf, att, Site_ind_n, prj=proj_n)
                #         hf3[proj_n + att] = tb_att
                #     hf3[proj_n + "/site_loc"] = [site_info[1], site_info[2]]
                #     hf3[proj_n + "/tb_cell_lat"] = c_lat_n
                #     hf3[proj_n + "/tb_cell_lon"] = c_lon_n
                #     count_tb += 1
                #     break
            # write data to h5 file

            if count_tb == 0:  # not pixels was found
                print 'in ths date %s and site No. %s, no tb file was founded' % (time_str, site_number)
                hf3 = h5py.File("Site_"+site_info[0] + orbit + time_str + ".h5", "a")
                miss_data = np.array([-9999.0])
                for att in tb_attr_name:
                    hf3[proj + att] = miss_data
                hf3[proj + "/tb_cell_lat"] = miss_data
                hf3[proj + "/tb_cell_lon"] = miss_data
                new_h5_name = hf3.filename
                hf3.close()
            else:
                hf3 = h5py.File("Site_"+site_info[0] + orbit + time_str + ".h5", "a")
                for att in tb_attr_name:
                    hf3[proj + att] = d_att[att]
                hf3[proj + "/tb_cell_lat"] = d_att["/cell_lat"]
                hf3[proj + "/tb_cell_lon"] = d_att["/cell_lon"]
                lt_n = hf3[proj + "/tb_cell_lat"].value
                lg_n = hf3[proj + "/tb_cell_lon"].value
                new_h5_name = hf3.filename
                hf3.close()
            # else:
            #     continue
        # read data from list: north pole prj
        '''
        ********************************************************************************************************************
    2\\\Read the 36 km region with a site in it.
        ********************************************************************************************************************
        '''
        # print '=============error test2==========================='
        # print 'attribute of h5'
        # print hf3.keys()
        # Read the 36 km region with a site in it.
        count_img = 1  # if the corresponding img cannot be found, this num. remains 1
        continue
        for files in os.listdir(rootdir):  # all radar data for this region saved together
            if re.search(time_str, files) and re.search(orbit, files) and ('001', files):  # matched by date
                Sig_name = rootdir + files
                # read the sigma near the site
                [Site_ind_sig, c1_lat_radar, c1_lon_radar, status2] = return_ind(Sig_name, site_info, 'sigma')
                if np.any(Site_ind_sig == -1):
                    print('For the radar file %s, site can not be found' % files)
                    logging1.info('For the radar file %s, site can not be found' % files)
                    continue
                else:
                    print 'data has been found in: ', files
                    logging1.info('data has been found in: %s' % files)
                sig0 = read_series(Site_ind_sig, Sig_name)
                Site_sig0[i] = sig0
                if sig0:  # Read the radar data region of 36 km, 36 * 36 pixels
                    r = 36  # r for the size of region, 36 km
                    print(Site_ind)
                    print(Sig_name)

                    # stat_c_gm, tb_corners = find_nearest(c_lat_g, c_lon_g, [site_info[1], site_info[2]], prj='gm', id=site_number, fig_name='GM_nearest' + site_number + time_str)  # retrieve the corner of tb
                    stats_c, max_dis = find_nearest(lt_n, lg_n, [site_info[1], site_info[2]], id=site_number, fig_name='N_nearest' + site_number + time_str)
                    print max_dis
                    # if type(max_dis) is float or len(max_dis) < 2:
                    #     print 'position warning'
                    # else:
                    stats, sig_ind = read_region3(Sig_name, max_dis, ['Sigma0_Data/cell_lat', 'Sigma0_Data/cell_lon'])
                    if stats != -1:
                        print new_h5_name
                        print time_str
                        save_radar(sig_ind, Sig_name, new_h5_name,
                                  ['Sigma0_Data/cell_sigma0_vv_aft', 'Sigma0_Data/cell_sigma0_hh_aft',
                                   'Sigma0_Data/cell_sigma0_qual_flag_vv', 'Sigma0_Data/cell_sigma0_qual_flag_hh',
                                   'Sigma0_Data/cell_lat', 'Sigma0_Data/cell_lon'])
                    # set flag
                    # if stats_c == -1:  # station was outside the pixel
                    #     hf3['flag'] = 1
                    # else:
                    #     hf3['flag'] = 0
                    # print('hf3 done for %d' % n)
                    # print('-----------------------------------------------------------------')
                    # logging1.info('-----------------------------------------------------------------')
                    # n += 1
                    # count_img += 1

                    break
        # if count_img == 1:
        #     hf3.close()
                    # print(Sig_region.shape)
                # print("++++++++++++++++reading++++++++++++++++++")
        # i +
        # sys.exit()
# print(time_list)
# print(Site_sig0)


def radar_read_alaska(orbit, site_number, peroid, polars, r_range=False, year=2016):
    n = 1
    # orbit = '_D_'
    rootdir = '/media/Seagate Expansion Drive/Data_Xy/CloudFilm/'  # directory: sigma
    TBdir = '/media/327A50047A4FC379/SMAP/SPL1CTB.003/'  # directory: TB, listed by date document
    # if year == 2018:
    #     TBdir = '/media/327A50047A4FC379/SMAP/smap_2018/'
    TB_date_list = sorted(os.listdir(TBdir))
    TB_date = []
    TB_date2, TB_ind = [], -1  # use to find yyyy.mm.dd format documents
    # Time list of radar data
    i = 0
    tb_attr_name = ['/cell_tb_v_aft', '/cell_tb_h_aft', '/cell_tb_qual_flag_v_aft', '/cell_tb_qual_flag_h_aft',
                    "/cell_lat", "/cell_lon"]  # /cell_lon here is for tb cell

    # list of additional tb data
    time_ini = 0
    for time_dot in TB_date_list:
        if time_dot == peroid[0]:
            time_ini = 1
        elif time_dot == peroid[1]:
            time_ini = 0
        if time_ini > 0:
            TB_date.append(time_dot.replace(".", ""))
            TB_date2.append(time_dot)
    for time_str in TB_date:  # loop day by day
        TB_ind += 1
    # for time_str in ['20150421', '20150422']:
        print('At the date of %s and %s' % (time_str, TB_date2[TB_ind]))
        logging1.info('At the date of %s' % time_str)
        '''
        ********************************************************************************************************************
    1\\\Find the TB data filename list and read the data around the station,
        for each site the radius in degrees is set as 1 degree, several pixels around the Site will be found.
        1a. Generate the file list;
        1b. Read each file in the list.
        ********************************************************************************************************************
        '''
        # 01 list with radiometer files
        tb_file_list = []
        tb_ob_list = []
        tb_folder = TB_date2[TB_ind]
        print 'the order of tb_file is :', tb_folder
        for tbs in os.listdir(TBdir + tb_folder):
            if tbs.find('.iso') == -1 and tbs.find(orbit) != -1 and tbs.find('.qa') == -1:  # Using ascend orbit data, iso and is metadata we don't need
                tb_file_list.append(tbs)
                p_underline = re.compile('_')
                tb_ob_list.append(p_underline.split(tbs)[3])
        obset = set(tb_ob_list)
        tb_name_list = []
        for orb_no in obset:  # ob_set is a set of orbit numbers
            list_orb = []
            for tb_file in tb_file_list:  # tb_file_list is a list of all tb files (Line 175)
                if re.search(orb_no, tb_file):
                    list_orb.append(tb_file)
            if len(list_orb) > 1:
                looks = sorted([p_underline.split(lkt)[-1] for lkt in list_orb])
                last_look = looks[-1]
                for lk in list_orb:
                    if lk.find(last_look) != -1:
                        tb_name_list.append(lk)
            else:
                tb_name_list.append(sorted(list_orb)[-1])

        # 02 read data from list created in 01
        proj_g = 'Global_Projection'
        proj_n = 'North_Polar_Projection'
        count_tb = 0  # remains 1 if no tb file was found
        d_att = {'_key': 'dict for global tb'}  # create dictionary for store tb data temporally
        for proj in [proj_n]:
            count_tb = 0
            for namez in tb_attr_name:
                d_att[namez] = np.array([])
            tb_full_path = [TBdir + tb_folder + '/' + tb_name for tb_name in tb_name_list]
            # fname = TBdir + tb_folder + '/' + tb_name
            print site_number
            if r_range is not False:
                site_ind, hdf5_slice = return_ind(tb_full_path, site_number, 'grid', prj=proj, thtb=r_range)
            else:
                site_ind, hdf5_slice = return_ind(tb_full_path, site_number, 'grid', prj=proj)
            for sn in hdf5_slice.keys():
                hf0 = h5py.File('./result_08_01/area/SMAP/SMAP_'+sn+orbit+time_str+'.h5', 'a')
                for key in hdf5_slice[sn]:
                    hf0[key] = hdf5_slice[sn][key]
                hf0.close()
            continue


def readradar(orbit='_A_'):
    # set h5 list
    site_no = '947'
    site_list = ['947', '949']
    radar_field = ['cell_sigma0_vv_aft', 'cell_sigma0_vv_fore', 'cell_sigma0_hh_aft', 'cell_sigma0_hh_fore',
                   'cell_sigma0_xpol_aft', 'cell_sigma0_xpol_fore', 'cell_lon', 'cell_lat']  # the string name of certain attributes, e.g. Sigma0_Data
    site_info = site_infos.change_site(site_no)
    peroid = ['20150414', '20150416']
    h5_list, date_list = read_site.get_h5_list(peroid[0], peroid[1], site_no, orbit)
    # search by date, locate by lat/lon, write data in h5 file
    h5_path = site_infos.get_data_path('_05_01') + 's' + site_no + '/'
    radar_path = '/home/xiyu/Downloads/SMAP/CloudFilm/'
    for h5_file in h5_list:
        fname = h5_path + h5_file
        hf = h5py.File(fname, 'r+')
        if 'Sigma0_Data' not in hf.keys():
            hf.create_group('Sigma0_Data')
        hf1 = hf['Sigma0_Data']  # saved in  h5 format
        radar_data = np.array([])
        for files in os.listdir(radar_path):
            if re.search(h5_file[11: 19], files) and re.search(orbit, files) and ('001', files):
                print files
                radarname = radar_path + files
                i_sig, c1_lat_radar, c1_lon_radar = return_ind(radarname, site_info, sensor='sigma')
                if i_sig.size > 0:
                    radar_data = save_radar(i_sig, radarname, ['Sigma0_Data/'+att for att in radar_field], radar_data)
        for att in radar_field:
            n = 0
            if att in hf1.keys():
                del hf1[att]
            if radar_data.size>0:
                hf1[att] = radar_data[n, :]
            else:
                hf1.create_dataset(att, data=np.array([]))
            n += 1
        hf.close()
    return 0


def read_nc_ds(ind, fname, att=[':f_usable', ':inc_angle_trip']):
    np_ds = []
    col = 0
    for var in att:
        var_obj = gdal.Open(fname+var)
        var_value = [[], [], []]
        for i in range(0, ind[0].size, 1):
            sig_tp = var_obj.GetRasterBand(ind[0][i]).ReadAsArray()
            for j in range(0, 3):
                var_value[j].append(sig_tp[ind[1][i], j])
        np_ds.append(var_value)
        col += 3
    return np_ds


def read_netcdf_ds(ind, fname, att=[':f_usable', ':inc_angle_trip']):
    np_ds = []
    col = 0
    rootgrp = Dataset(fname, 'r', format='NETCDF')
    for var in att:
        #var_obj = gdal.Open(fname+var)
        var_obj = rootgrp.variables[var][:]
        var_value = [[], [], []]
        for i in range(0, ind[0].size, 1):
            sig_tp = var_obj.GetRasterBand(ind[0][i]).ReadAsArray()
            for j in range(0, 3):
                var_value[j].append(sig_tp[ind[1][i], j])
        np_ds.append(var_value)
        col += 3
    rootgrp.close()
    return np_ds


def readascat(file_daily, site_no, orbit, timestr, center=False, satellite='B'):  # from 16 to 17
    '''

    :param file_daily: files available of a specified date
    :param site_no:
    :param orbit:
    :return:
    '''
    #siten = site_infos.change_site(site_no)
    # create np arrays
    value, lat, lon, axillary, orb_all = [], np.array([]), np.array([]), [], np.array([])
    stat = 0
    file_ncread = file_daily
    return_ind(file_ncread, site_no, 'ascat', thsig=[2, 2], orbz=orbit, fname='result_08_01/point/ascat/ascat_'+timestr+'_metop'+satellite, center=center)
    return 0
    for filenc in file_daily:
        file_ncread = 'NETCDF:'+ filenc
        r_ind, c_lat, c_lon, orb, statu = return_ind(file_ncread, siten, 'ascat', thsig=[15e4, 15e4], orbz=orbit)
        if statu > 0:
            # and save
            stat += 1
            if len(value) > 0:
                tri = 0  # refer the triplets of ascat
                for valuei in value:
                    #valuei.append(r_ind[1][tri])
                    valuei += r_ind[1][tri]
                    #np.append(valuei, r_ind[1][tri])
                    tri += 1
                axil_tp = read_nc_ds(r_ind[0], file_ncread)
                att_count = 0
                for ax_att in axillary:
                    tri = 0
                    for ax_att_tri in ax_att:
                        #np.append(ax_att_tri, axil_tp[att_count][tri])
                        ax_att_tri += axil_tp[att_count][tri]
                        tri += 1
                    att_count += 1
            else:
                axillary = read_nc_ds(r_ind[0], file_ncread)
                value=r_ind[1]
            lat=np.append(lat, c_lat)
            lon=np.append(lon, c_lon)
            orb_all=np.append(orb_all, orb)
            print filenc
            print r_ind
    if stat > 0:
        return value, lon, lat, axillary, orb_all, 1
    else:
        return -9999, -9999, -9999, -9999, -9999, -1


def getascat(site_no, doy, year0=2015, orbit=1, center=False, sate='B'):
    '''

    :param site_no: a list of stations
    :param doy: interested time range
    :param year0:
    :param orbit:
    :return:
    '''
    if sate=='B':
        path_ascat = '/media/327A50047A4FC379/ASCAT/ascat_l1/'
    else:
        path_ascat = '/media/Seagate Expansion Drive/ASCAT/ascat_1a/'
    filelist = os.listdir(path_ascat)
    yr = year0 + 1 + doy/365
    dayz = doy - (doy/365 * 365)  # get the year and j
    dtime0 = datetime.datetime(yr, 01, 01) + datetime.timedelta(dayz-1)
    dtime_str = dtime0.strftime('%Y%m%d')  # get yyyymmdd formated string, matching it in file list
    print 'Now data at %s is searching' % (dtime_str)
    file_daily = []  # daily nc list
    for ncfile in filelist:
        if re.search(dtime_str, ncfile):
            print '     file was being loading: %s' % ncfile
            file_daily.append(path_ascat+ncfile)
    # read all daily nc files
    sigma0 = readascat(file_daily, site_no, orbit, dtime_str, center=center, satellite=sate)
    return 0
    # txt_name = 'ascat_'+site_no+'_'+dtime_str+'.txt'
    #     #h1 = h5py.File('ASCAT_'+site_no+'_'+dtime_str+'.h5', 'w')
    #     #h1["site_info"] = 1.0
    #    # h1["site_info"] = np.array([siten[1], siten[2]])
    #     #hf = h1.create_group('12_5km')
    #     #hf['sigma0'] = sigma0
    #     #hf['lat'] = lon
    #     #hf['lon'] = lat
    #     #h1.close()
    # x = np.array(sigma0)
    # if status > 0:
    #     y = np.array(other).reshape(-1, lon.size)
    #     value = np.concatenate((np.array([lon, lat]), x, y), axis=0)
    #     value = np.vstack([value, orb_indicator])
    #     np.savetxt(txt_name, np.transpose(value), delimiter=',',
    #                fmt='%.6f, %.6f, %.6f, %.6f, %.6f, %d, %d, %d, %.2f, %.2f, %d')
    # else:
    #     np.savetxt(txt_name, np.transpose(np.array([lon, lat, sigma0, other])), delimiter=',', fmt='%d')
    # return 0


def read_ascat_alaska(doy, year0=2015, sate='B'):
    '''
    :param doy:  day of year based on year0
    :param year0:
    :param sate: metopB and metop A saved in different pathes
    :return: the .npy, include 47 fileds, see in '../meta0_ascat_ak.txt'
    '''
    site_no = ['alaska']
    if sate == 'B':
        path_ascat = '/media/327A50047A4FC379/ASCAT/ascat_l1/'
    elif sate == 'A':
        path_ascat = '/media/Seagate Expansion Drive/ASCAT/ascat_1a/'
    filelist = os.listdir(path_ascat)
    # yr = year0 + doy/365  # start from 2016
    # dayz = doy - (doy/365 * 365)  # get the year and j
    # dtime0 = datetime.datetime(yr, 01, 01) + datetime.timedelta(dayz-1)
    # dtime_str = dtime0.strftime('%Y%m%d')  # get yyyymmdd formated string, matching it in file list
    dtime_str = bxy.doy2date(doy, fmt='%Y%m%d', year0=year0)
    print 'Now data at %s is searching' % (dtime_str)
    file_daily = []  # daily nc list
    for ncfile in filelist:
        if re.search(dtime_str, ncfile):
            print '     file was being loading: %s' % ncfile
            file_daily.append(path_ascat+ncfile)
    # read all daily nc files
    file_ncread = file_daily
    return_ind(file_ncread, site_no, 'ascat', thsig=[8.58, 17.54], orbz=None, fname='./result_08_01/area/ascat/ascat_'+dtime_str+'_metop'+sate)
    save_name_npy = './result_08_01/area/ascat/ascat_'+dtime_str+'_metop'+sate+'_alaska.npy'
    # sigma0 = readascat(file_daily, site_no, orbit, dtime_str)
    # readascat(file_daily, site_no, orbit, timestr)
    if os.path.isfile(save_name_npy):
        return 0
    else:
        return -1


def get_peroid(st, en):
    doy_st = read_site.days_2015(st)
    doy_en = read_site.days_2015(en)
    doy_list = range(doy_st, doy_en+1, 1)
    return doy_list


def creat_h5(site_id, groupname):
    hf_objs = []
    for s0 in site_id:
        hf = h5py.File('./h5_l1c/smap_'+s0, 'a')
        hf.create_group('North_Polar_Projection/File_Loc')
        hf.close()
    return hf_objs


def read_tb_site(h5_smap, dic_site, location, prj):
    att_keys = h5_smap[prj].keys()
    for key in att_keys:
        dic_site[prj+'/'+key]=h5_smap[prj+'/'+key][location]
    return dic_site


def read_tb2txt(site_no, ob,
                attribute_name='smap_tb', fname=[], year_type = 'water', is_inter=True, ipt_path='_05_01', site_loc='al'):
    if site_loc == 'ak':
        site_path = site_infos.get_data_path(ipt_path) + 's' + site_no + '/'
    else:
        site_path = 'result_08_01/20181101/smap_series/' + 's' + site_no + '/'

    if year_type == 'water':
        file_list, d_list = read_site.get_h5_list('20151001', '20170301', site_no, ob, ipt_path)  #
    elif year_type == 'tibet':
        path_smap_match = 'result_08_01/20181101/smap_series/s%s/SMAP*%s*.h5' % (site_no, ob)
        matched_path = glob.glob(path_smap_match)
        file_list = sorted([f0.split('/')[-1] for f0 in matched_path])
        d_list = [f0.split('.')[0].split('_')[-1] for f0 in file_list]
    else:
        file_list, d_list = read_site.get_h5_list('20160101', '20161225', site_no, ob)
    # initials
    year_2016 = []  # interpolated pixel
    year_2016_all = np.zeros([len(file_list), 4, 20]) - 99999  # all pixels
    att = site_infos.get_attribute(sublayer=attribute_name)
    head = "day, "
    for atti in att[1]:
        head += (atti+',')
    day0 = 0
    for doyi, single_file in enumerate(file_list):
        if single_file == 'SMAP_1233_A_20160531.h5':
            pause = 0
        filename = site_path + single_file
        if is_inter is True:
            v_day, dis_day, locs, stat = data_process.interp_radius(filename, site_no, dat=attribute_name, disref=27)
            year_2016.append(v_day)
        else:
            v_day, v_dis, stat = data_process.read_all_pixel(filename, site_no, dat=attribute_name, disref=27)
            if stat <0:
               year_2016_all[doyi, :, :] = v_day
            else:
                for i2, v2 in enumerate(v_day):
                    year_2016_all[doyi, i2, :] = v2

        # if (day0 < 5) & (stat > -1):
        #     loc_np = np.array(locs).ravel()
        #     id_np = np.zeros(dis_day.size) + int(site_no)
        #     np_dis = np.zeros(dis_day.size) + dis_day
        #     txt_value = np.concatenate((id_np, loc_np, np_dis)).reshape(-1, dis_day.size)
        #     print txt_value
        #     txt_fname = '%stb_multipixels%s%s.txt' % (prefix, ob, site_no)
        #     np.savetxt(txt_fname, txt_value.T,
        #                delimiter=',', fmt='%d, %.5f, %.5f, %.2f', header='id, lon, lat, distance')
        # day0+=1
    doy = data_process.get_doy(d_list)
    if is_inter is True:
        year_2016_np = np.array(year_2016)
        n = att[1].index('cell_tb_v_aft')
        id_valid = year_2016_np[:, n] != -9999
        year_2016_valid = year_2016_np[id_valid, :]
        doy_valid = np.array([doy[id_valid]]).T - 365
        # print doy_valid.shape, year_2016_valid.shape
        saved_np = np.concatenate((doy_valid, year_2016_valid), axis=1)
        print saved_np.shape
        # print saved_np.shape
        np.savetxt(fname, saved_np, delimiter=',',
                   fmt='%d %.4f %.4f %.2f %d %.2f %.2f %d %.2f %.2f %.2f %.2f %d %.2f %.2f %d %.2f %.2f %.2f %d %d',
                   header=head)
    else:
        for i1, dis0 in enumerate(v_dis):
            if dis0 < 0:
                continue
            year_2016_np = year_2016_all[:, i1, :]
            # if any(year_2016_np==-99999):
            #     # dont have this pixel
            #     continue
            n = att[1].index('cell_tb_v_aft')
            id_valid = year_2016_np[:, n] != -9999
            year_2016_valid = year_2016_np[id_valid, :]
            doy_valid = np.array([doy[id_valid]]).T - 365
            # print doy_valid.shape, year_2016_valid.shape
            saved_np = np.concatenate((doy_valid, year_2016_valid), axis=1)
            print saved_np.shape
            # print saved_np.shape
            fname_pixel = fname+'_'+str(v_dis[i1])+'.txt'
            np.savetxt(fname_pixel, saved_np, delimiter=',',
                       fmt='%.5f',
                       header=head)
            status = 0


def read_amsr2(site_list, period, orb='A', th=[5, 5]):
    """
    :param site_list: e.g., ['947', '968']
    :param period: e.g., [date0, date1], fm='%Y%m%d'
    :param orb:
    :return:
    """
    prefix1 = '/media/327A50047A4FC379/amsr2h5/cxy/'
    doy_period = data_process.get_doy(period, fm='%Y.%m.%d')
    doy_end = data_process.get_doy('20160429')
    doys = np.arange(doy_period[0], doy_period[1])  # a array of doys
    for doy0 in doys:
        doy_obj = datetime.datetime(2015, 1, 1) + datetime.timedelta(days=doy0-1)
        month_dir = prefix1 + doy_obj.strftime('%Y.%m') + '/2/'
        if doy0 > doy_end[0]:
            month_dir = '/media/327A50047A4FC379/amsr2h5/cxy/2016.May_Dec'
        print month_dir
        datestr = doy_obj.strftime('%Y%m%d')
        print datestr
        h5_name_search = '*'+datestr+'*%s_*.h5' % orb
        h5_name_search = '%s/*%s*%s_*.h5' % (month_dir, datestr, orb)
        filelist = glob.glob(h5_name_search)
        loc_dicts, hdf5_slice = return_ind(filelist, site_list, 'amsr2', thtb=th)
        for sn in hdf5_slice.keys():
                h5_newfile = 'AMSR2_l2r_%s_%s_%s.h5' % (datestr, sn, orb)
                hf0 = h5py.File('./result_08_01/area/amsr2/'+h5_newfile, 'a')
                for key in hdf5_slice[sn]:
                    hf0[key] = hdf5_slice[sn][key]
                if loc_dicts[1][sn].size>1:
                    hf0['latitude_36GHz'] = loc_dicts[1][sn]
                    hf0['longitude_36GHz'] = loc_dicts[0][sn]
                else:
                    print loc_dicts[1][sn], 'type: ', type(loc_dicts[1][sn])
                hf0.close()
    return 0


def amsr2_series(site_no, attribute_list, orbit='A'):
    h5name = './result_07_01/amsr2_site_h5/*%s_%s.h5' % (site_no, orbit)
    h5_dir_list = glob.glob(h5name)
    # initial output
    save_value = np.zeros([len(h5_dir_list), len(attribute_list)+3]) - 9999
    header0 = 'doy;lon;lat;' + ';'.join(attribute_list)
    for i, h5_dir in enumerate(h5_dir_list):
        hf0 = h5py.File(h5_dir)
        if len(hf0.keys()) > 3:  # check if h5file is empty
            h5name = h5_dir.split('/')[-1]
            h5_date = h5name.split('_')[2]
            h5_doy = data_process.get_doy(h5_date)-365
            lons = hf0['longitude_36GHz'].value
            lats = hf0['latitude_36GHz'].value
            value_list = []
            for att in attribute_list:
                if 'res06' in att:
                    dis0 = 30
                else:
                    dis0 = 14
                values = hf0[att].value
                value_interp, dis, location, status = \
                    data_process.interpolation_spatial(values, lons, lats, site_no, disref=dis0)
                value_list.append(value_interp)
            save_value[i, 0] = h5_doy
            save_value[i, 3:] = np.array(value_list)
    sort_id = np.argsort(save_value[:, 0])
    save_value = save_value[sort_id]
    save_name = 'amsr2_series_%s_%s.txt' % (site_no, orbit)
    np.savetxt(save_name, save_value, fmt='%d', header=header0)
    return 0


def check_ak_ascat(dict):
    lines = 0
    for key in dict.keys():
        lines += dict[key][0].ndim
        print key, ': ', dict[key][0].ndim
        print 'in total are %d lines' % lines


def read_amsr2_l3(site=['947', '968'], prj='EQMA'):
    # get h5 list, file name example: GW1AM2_20160102_01D_PNMD_L3SGSNDLG2210210.h5
    h5_list = []
    l3_path = '/media/Seagate Expansion Drive/AMSR2_SND'
    month = ['01', '02', '03', '04', '05', '06', '07']
    for month0 in month:
        match_name = '%s/%s/*01D_%s*.h5' % (l3_path, month0, prj)
        monthly_file_list = glob.glob(match_name)
        h5_list += monthly_file_list
    # read measurements of interested pixels
    pixels, p_values, p_sec = return_ind(h5_list, site, prj, atts=['Geophysical Data', 'Time Information'], cube=2)
    # dimensions: ('date', 'atts', 'location', 'variables')
    # temp_check:
    p_doy = bxy.time_getlocaltime(p_sec, ref_time=[2000, 1, 1, 0], t_out='utc')[3]
    swe_947 = p_values[:, 0, 1, 0]
    snd_947 = p_values[:, 0, 1, 1]
    i_swe0 = np.where(swe_947 == 0)
    i_snd0 = np.where(swe_947 == 0)
    date_snd0 = p_doy[i_snd0]
    pause=0
    # save in h5 files
    return pixels, p_values, p_sec