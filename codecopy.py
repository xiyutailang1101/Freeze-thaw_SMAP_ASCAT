read_region2(filename, corners, field):
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
            #value.shape = -1,
            #lat.shape = -1,
            #lon.shape = -1,
            # sig0_list, lat_list, lon_list = np.array([]), np.array([]), np.array([])
            # for i in range(0, corners[0].size):
            # lat_ind = np.where(corners[0][0] < lat < corners[1][0])
            print(corners)
            lat_ind = np.where(np.logical_and(lat < corners[1][0], lat > corners[0][0]))
            lat_ind2 = lat_ind[1]
            lon_ind = np.where(np.logical_and(lon < corners[1][1], lon > corners[0][1]))
            print('*******************')
            print(lon_ind)
            lon_ind2 = lon_ind[1]
            print(lon_ind2)
            # lon_ind = np.where(corners[0][1] < lon < corners[1][1])
            region_ind = np.intersect1d(lon_ind2, lat_ind2)
            print('there is %d * 36 pixels were picked' % int(np.size(region_ind)/36))
            sig0_list, lat_list, lon_list = value[region_ind], lat[region_ind], lon[region_ind]
            print(type(sig0_list))
            print('the lat range is %f to %f' % (np.min(lat_list), np.max(lat_list)))
            print('the lon range is %f to %f' % (np.max(lon_list), np.min(lon_list)))
            return sig0_list, lat_list, lon_list
        else:
            return -1, -1, -1
# copy of read_region2, output is 1 dimension
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

            print corners[1][0], corners[0][0], corners[1][1], corners[0][1]
            # locate the range within the swath using true and false
            trf_swath = \
                np.logical_and(np.logical_and(lat < corners[1][0], lat > corners[0][0]),
                               np.logical_and(lon < corners[1][1], lon > corners[0][1]))  # within lat and lon lim
            rc = trf_swath.nonzero()  # return all the rows and cols (rc) for the conditions above
            lim_row = [np.min(rc[0]), np.max(rc[0])]  # for the region, there is lim of rows and cols
            lim_col = [np.min(rc[1]), np.max(rc[1])]
            print([lim_row, lim_col])

            value_regions = value[lim_row[0]: lim_row[1] + 1,    # Pixels in forms of 2-D grid
                           lim_col[0]: lim_col[1] + 1]          # all the possible pixels in the region
            trf_region = trf_swath[lim_row[0]: lim_row[1] + 1,  # true or false of all the possible pixel
                           lim_col[0]: lim_col[1] + 1]          # where the false are pixels out of the corners boundary
            value_regions[np.logical_not(trf_region)] = -9999    # -9999 is assigned to the outside pixel

            lat_regions = lat[lim_row[0]: lim_row[1] + 1,    # Latitude of radar pixels in forms of 2-D grid
                           lim_col[0]: lim_col[1] + 1]
            lon_regions = lon[lim_row[0]: lim_row[1] + 1,    # Longitude of radar pixels in forms of 2-D grid
                           lim_col[0]: lim_col[1] + 1]

            print('*******************')
            print('there is %d * %d pixels were picked' % (value_regions.shape[0], value_regions.shape[1]))
            # sig0_list, lat_list, lon_list = value[region_ind], lat[region_ind], lon[region_ind]
            # print(type(sig0_list))
            # print('the lat range is %f to %f' % (np.min(lat_list), np.max(lat_list)))
            # print('the lon range is %f to %f' % (np.max(lon_list), np.min(lon_list)))
            return value_regions, lat_regions, lon_regions
        else:
            return -1, -1, -1
#  old version for read_region, which may not be used.
                # for i in range(Site_ind.shape[0]):  # loop with the number of nearest "tb" pixels
                #     # *************************
                #     if n == 1:  # at the first loop of date, the picked radar 36*36 images's center is the fixed center
                #         [Center_ind, c2_lat_radar, c2_lon_radar] = return_ind(Sig_name, [site_info[0], c_lat[i], c_lon[i]], 'sigma')
                #         # select the nearest "radar" pixel to the site
                #         dis = (c2_lat_radar - c_lat[i])**2 + (c2_lon_radar - c_lon[i])**2
                #         min_ind = np.argmin(dis)
                #         print(c_lat[i], c_lon[i])
                #         print('in the swath: %s' % Sig_name)
                #         print('the nearest radar pixels are:')
                #         print([c2_lat_radar, c2_lon_radar])
                #         new_centers = [c2_lat_radar[min_ind], c2_lon_radar[min_ind]]
                #         print('the fixed center is %f, %f' % (new_centers[0], new_centers[1]))
                #     # use the fixed center to read region
                #     [new_center_ind, new_lat, new_lon] = return_ind(Sig_name, [site_info[0], new_centers[0], new_centers[1]], 'sigma')
                #     dis = (new_lat - new_centers[0])**2 + (new_lon - new_centers[1])**2
                #     min_ind = np.argmin(dis)
                #     print('the new-defined center is %f, %f with index of %d'
                #           % (new_lat[min_ind], new_lon[min_ind], new_center_ind[min_ind]))
                #     [sig_region_i, lat_region_i, lon_region_i] = read_region(new_center_ind[min_ind], Sig_name, 36)
                #     sig_region[:, i*r:i*r+r] = sig_region_i
                #     lat_region[:, i*r:i*r+r] = lat_region_i
                #     lon_region[:, i*r:i*r+r] = lon_region_i
