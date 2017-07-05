import numpy as np
import os
import data_process
import peakdetect
import matplotlib.pyplot as plt
import site_infos


def main(n_b):
    path = 'result_03_4/alaska02/ascend/'
    fname_list= sorted(os.listdir(path))
    # background
    # n_b = 53
    x0 = np.load(path+fname_list[n_b])
    x0 = x0.T
    lengz = x0[0].size
    # min(n_b+100, len(fname_list))
    sig, angle, land, fu = [], [], [], []
    for fname in fname_list[n_b+1: len(fname_list)]:
        x1 = np.load(path+fname)
        x1 = x1.T
        for i in range(0, lengz, 1):
            lat0, lon0 = x0[0, i], x0[1, i]
            dis = cal_dis(lat0, lon0, x1[0], x1[1])
            # con_la = (x1[0]-lat0)**2
            # con_lg = (x1[1]-lon0)**2
            ind = np.where(dis<15)
            if ind[0].size > 0:
                #i_near = near_pixel(dis[ind[0]], ind, x1)
                f_l = x1[12, ind[0]]
                #dis_min = dis[ind[0]]
                i_near = ind[0][f_l>0.5]
                if i_near.size < 0.5:
                    i_near = ind[0][0]
                #print lat0, lon0, '\n', x1[0, i_near], x1[1, i_near], '\n'
                sig.append(np.mean(x1[3, i_near])), angle.append(np.mean(x1[9, i_near])), land.append(np.mean(x1[12, i_near])), \
                fu.append(np.mean(x1[6, i_near]))
                print sig[i], angle[i], '\n'
            else:
                sig.append(-99), angle.append(-99), land.append(-99), fu.append(-99)
                print 'no data for this (%.3f, %.3f) at %s' % (lat0, lon0, fname), '\n'
        status = 1
    np.save('sigma_'+str(n_b), np.hstack((x0[3, :], np.array(sig))))
    np.save('angle_'+str(n_b), np.hstack((x0[9, :], np.array(angle))))
    np.save('land_'+str(n_b), np.hstack((x0[12, :], np.array(land))))
    np.save('fu_'+str(n_b), np.hstack((x0[6, :], np.array(land))))


def near_pixel(dis, ind, ak):
    '''

    :param lat:
    :param lon:
    :param ind:
    :param ak: daily alaska data, with coord, sigma and angle
    :return:
    '''
    f_l = ak[12, ind[0]]site_no
    ind0 = np.argsort(dis)
    a = 0
    for i0 in ind0:
        if f_l[i0] > 0.5:
            a +=1
            break
        else:
            continue
    if a == 0:
        i0 = ind0[0]

    return ind[0][i0]


def cal_dis(lat0, lon0, lats, lons):
    lamda0 = (lon0/180.0)*np.pi
    lamdas = (lons/180.0)*np.pi
    phi0 = (lat0/180.0)*np.pi
    phis = (lats/180.0)*np.pi
    x = (lamdas-lamda0) * np.cos((phis+phi0)/2)
    y = phis - phi0
    return 6371*np.sqrt(x**2 + y**2)


def edge_detect(npy_name, npy_date):
    path = 'result_03_4/alaska02/ascend/'
    file_lst = sorted(os.listdir(path))
    bkground = np.load(path+file_lst[npy_date[0]-1])
    site_ind, loc = station_find('2212', bkground)
    ref_num = bkground.shape[0]
    x1 = np.load(npy_name[0]).reshape(ref_num, -1)
    # np.save('./temp_save/sigma52', x1)
    # return 0
    a1 = np.load(npy_name[1]).reshape(ref_num, -1)
    f1 = np.load(npy_name[2]).reshape(ref_num, -1)
    x1[x1 == -99] = np.nan
    a1[a1 == -99] = np.nan
    a1[x1 < -40] = np.nan
    x1[x1 < -40] = np.nan
    f1[f1 == -99] = np.nan
    sig_ref = x1[:, 0:80]
    angle_ref = a1[:, 0:80]
    sigs = x1
    angles = a1
    g_size = 8
    onset_thaw, onset_frz = [], []
    count_noland, count_lackdata = 0, 0
    for i in range(site_ind, ref_num):
        print i, ':', bkground[i, 0], bkground[i, 1]
        # condition 0: land percentage smaller than 0.75
        if i == site_ind:  # station test!!
            print loc[0], loc[1], bkground[i, 0]-loc[0], bkground[i, 1]-loc[1]
        if np.nanmean(f1[i, :]) < 0.75:
            count_noland += 1
            onset_thaw.append(0)
            onset_frz.append(0)
            continue
        # condition 1
        # if np.nanmean(sig_ref[i, 10: 30]) < -20:
        #     onset_thaw.append(-99)
        #     onset_frz.append(-99)
        #     continue
        iga = ~np.isnan(angle_ref[i, :])  #??
        igs = ~np.isnan(sig_ref[i, :])
        # condition 2
        if np.count_nonzero(~np.isnan(angles[i, :]))<100:
            count_lackdata += 1
            onset_thaw.append(0)
            onset_frz.append(0)
            continue
        a, b = np.polyfit(angle_ref[i, :][iga], sig_ref[i, :][igs], 1)
        crr_thaw = sigs[i, :] - (angles[i, :] - 45)*a
        #crr_thaw = sigs[i, :]
        # if np.nanmean(crr_thaw[0]<-20):
        #     onset_thaw.append(-99)
        #     onset_frz.append(-99)
        #     continue
        g_sig, ig2 = data_process.gauss_conv(crr_thaw, sig=3)
        g_sig_valid = (g_sig[g_size: -g_size] - np.nanmin(g_sig[g_size: -g_size]))\
                    /(np.nanmax(g_sig[g_size: -g_size]) - np.nanmin(g_sig[g_size: -g_size]))
        max_gsig_s, min_gsig_s = peakdetect.peakdet(g_sig_valid, 1e-1, npy_date[g_size: -g_size])
        onset = data_process.find_inflect(max_gsig_s, min_gsig_s, typez='annual')
        if i == site_ind:
            print loc[0], loc[1], bkground[i, 0]-loc[0], bkground[i, 1]-loc[1]
            plot_series(npy_date, crr_thaw, onset)
        onset_thaw.append(onset[0])
        onset_frz.append(onset[1])
    txt_th = np.vstack((bkground.T[0], bkground.T[1], np.array(onset_thaw)))
    txt_fr = np.vstack((bkground.T[0], bkground.T[1], np.array(onset_frz)))
    txt_th0, txt_fr0 = txt_th[:, txt_th[2] > 0], txt_fr[:, txt_fr[2] > 0]
    print 'noland pixel: %d, lackdata pixel: %d' % (count_noland, count_lackdata)
    np.savetxt('alaska_thaw'+file_lst[npy_date[0]-1][9:17]+'.txt', txt_th0.T, fmt='%.6f, %.6f, %d', header='Y_lat, X_lon, date')
    np.savetxt('alaska_frze'+file_lst[npy_date[0]-1][9:17]+'.txt', txt_fr0.T, fmt='%.6f, %.6f, %d', header='Y_lat, X_lon, date')
    return 0


def plot_series(x, y, vl):
    fig = plt.figure()
    ax = fig.add_subplot(511)
    ax.plot(x, y)
    symbol = ['--', '-.']

    for i in range(0, 2):
        ax.axvline(x=vl[i], color='k', ls=symbol[i])
    plt.savefig('region_test.png', dpi=120)


def station_find(site, bk):
    info = site_infos.change_site(site)
    dis = (bk[:, 0] - info[1])**2 + (bk[:, 1] - info[2])**2
    return np.argmin(dis), [info[1], info[2]]