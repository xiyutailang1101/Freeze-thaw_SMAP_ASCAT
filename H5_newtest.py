__author__ = 'xiyu'
import re, os
import h5py
import numpy as np


def printarr(arr):
    for name in arr:
        print(name)
    # info of sites

# then only read this orbit, so the TB value reading is optimized.
SiteList = [['947', 65.12422, -146.73390], ['948', 65.25113, -146.15133],
['949', 65.07833, -145.87067], ['950', 64.85033, -146.20945],
['1090', 65.36710, -146.59200]]
rootdir = "/media/Seagate Expansion Drive/Data_Xy/CloudMusic/"  # first catalog: data documents
for siten in SiteList:

    TB_aver = np.array([])
    date_ls = np.array([])
    date0 = 0  # start date, for each loops of files0, date+=1
    for files0 in os.listdir(rootdir):
        date0 += 1
        FileList = []
        ObList = []
        rootdir2 = rootdir + files0  # second catalog: data very day
        print('rootdir2 is %s' % rootdir2)
        for files in os.listdir(rootdir2):
            if files.find('.iso') == -1 and files.find('_A_') != -1:  # pick the A orbit data, eliminate the iso files
                FileList.append(files)
    # Get the orbit number
                p_underline = re.compile('_')
                ObList.append(p_underline.split(files)[3])
    # Search files by each orbit, use the last observation(00n)
        Obset = set(ObList)
        fname_list = []
        for ob_No in Obset:
            file_obn = []
            for files in FileList:
                if re.search(ob_No, files):
                    file_obn.append(files)
            fname_list.append(sorted(file_obn)[-1])  # sort the the same-orbit-list and choose the last one
        # ['SMAP', 'L1C', 'TB', '00869', 'A', '20150331T230335', 'R11850', '002.h5']

        #  ======================hdf5 reading===============================
        print('in the date of xxx there are %d orbits of data' % len(fname_list))
        # use '947' as example
        lat_t = siten[1]
        lon_t = siten[2]  # float
        for oi in np.arange(len(fname_list)):
            hf = h5py.File(rootdir2 + '/' + fname_list[oi], 'r')
            # print('List of dataset in this h5file: \n', hf.keys())
            # [u'Metadata', u'Global_Projection', u'North_Polar_Projection', u'South_Polar_Projection']
            dset_lon = hf['Global_Projection/cell_lon']
            dset_lat = hf['Global_Projection/cell_lat']
            TBs = hf['Global_Projection/cell_tb_v_aft']

            #  select the within station region
            tp2 = TBs[np.abs(dset_lon[...] - lon_t) < 0.36]
            lon_ind = np.where(np.abs(dset_lon[...] - lon_t) < 0.36)
            # print('there is %d pieces of longitude' % len(lon_ind[0]))
            lat_ind = np.where(np.abs(dset_lat[...] - lat_t) < 0.36)
            # print('there is %d pieces of latitude' % len(lat_ind[0]))
            region_ind = np.intersect1d(lon_ind, lat_ind)
            # print('there is %d of pixels around' % len(region_ind))
            if len(region_ind) > 0:
                # TB_region = TBs(region_ind)
                TB_region = []
                for ind in region_ind:
                    TB_region.append(TBs[ind])
                TB_temp = np.mean(TB_region)
                TB_aver = np.append(TB_aver, TB_temp)
                print(fname_list[oi])
                break
                hf.close()
                date_ls = np.append(date_ls, date0)
                continue
    Sitedata = np.stack((date_ls, TB_aver))
    print("Size of Sitedata")
    print(np.shape(Sitedata))
    print(date_ls.dtype)
    print("Size of TB_aver")
    print(np.shape(TB_aver))
    print(TB_aver.dtype)

    # Sitedata2 = Sitedata.reshape(2, -1)
    np.savetxt("TB_" + siten[0] + ".txt", Sitedata)
            # print([dset_lon(region_ind), dset_lat(region_ind)])  # check the selected latitude and longitude