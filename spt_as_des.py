import numpy as np
import read_site
import matplotlib.pyplot as plt
import basic_xiyu as bxy
import plot_funcs
# import test_def, peakdetect
# import csv
# import site_infos
# import Read_radar
# import h5py
# import data_process
# import os, re, sys
# import warnings

# site_nos = ['1177']
site_nos = ['1177', '1233', '947', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '2081', '2210', '1089',  '2212', '2211']
# '967' '2065' '2081' '2210' '2213' '1089' '1062' '1233' '2212' '2211'
# '967', '2065', '2081', '2210', '2213', '1089', '1233', '2212', '2211',
# site_nos = ['1062']
site_dic = {'sno_': ['1089', '967', '1062', '947', '949', '950', '960', '962', '968','1090', '1175', '1177'],
                    'scan_': ['2081', '2213', '2210', '2065', '2212', '2211', '1233']}
# ASCAT process

n_pixel = []
inc_st = [[], [], [], [], [], [], []]
for site_no in site_nos:
    txtname = 'ascat_series_'+site_no+'.txt'
    txt_org = np.loadtxt(txtname, delimiter=',')  # daily observation, triplets
    out_select = read_site.read_ascat_txt(txtname)  # fixed by selected inc angle
    inc_st[0].append(np.mean(out_select[2]))
    inc_st[1].append(np.std(out_select[2]))
    angle_cor = read_site.fix_angle_ascat(txtname)  # corrected in 45 degre
    inc_st[2].append(np.std(angle_cor[2]))
    inc_st[3].append(np.nanmean(np.diff(out_select[2])))
    inc_st[4].append(np.nanmean(np.diff(angle_cor[2])))
    inc_st[5].append(np.std(np.diff(out_select[2])))
    inc_st[6].append(np.std(np.diff(angle_cor[2])))
    x_time = out_select[0]
    x_time_np = np.array(x_time)
    sigma0_np = np.array(out_select[1])
    # the orbit difference
    u, i_u, i_r, counts = np.unique(x_time, return_index=True, return_inverse=True, return_counts=True)
    i_rep = np.where(counts>1)
    x_as_des = u[i_rep]
    it = i_u[i_rep]  # th repeated (as + des)
    y_as_des = []  # as_des difference
    for i in it:
        tp = sigma0_np[i] - sigma0_np[i+1]
        y_as_des.append(tp)
    fig = plt.figure()
    ax = fig.add_subplot(511)
    ax.plot(x_as_des, y_as_des)
    ax.set_xlim([0, 350])
    ax1 = fig.add_subplot(512)
    orb_no_selected = np.array(out_select[3])  # orbit indicator of angle-selected data
    ob_as_slt = np.where(orb_no_selected < 0.5)
    ob_des_slt = np.where(orb_no_selected > 0.5)
    orb_no_all = txt_org[:, 12]
    ob_as_all, ob_des_all = np.where(orb_no_all < 0.5), np.where(orb_no_all > 0.5)
    # uncorrected backscatter
    sig_4050 = txt_org[:, 4]
    if any(sig_4050<-1e5):
        sig_4050*=1e-6
    ax1.plot(txt_org[:, 0][ob_as_all], sig_4050[ob_as_all])
    ax1.set_xlim([0, 350])
    ax1.set_ylim([-15, -7])
    # angle_based back scatter
    ax2 = fig.add_subplot(513)
    ax2.plot(x_time_np[ob_as_slt], sigma0_np[ob_as_slt])
    ax2.set_xlim([0, 350])
    ax2.set_ylim([-15, -7])
    # corrected back scatter
    ax3 = fig.add_subplot(514)
    angle_cor_np = angle_cor
    ax3.plot(angle_cor_np[0][ob_as_all], angle_cor_np[2][ob_as_all])
    ax3.set_ylim([-15, -7])
    ax4 = fig.add_subplot(515)
    ax4.plot(angle_cor_np[0][ob_as_all], angle_cor_np[2][ob_as_all]-txt_org[:, 4][ob_as_all])
    # labels
    ax.set_ylabel(r'$\Delta\sigma^0$'+'(dB)')
    ax1.set_ylabel('orginal' + r'$\sigma^0$')
    ax2.set_ylabel('40~50'+r'$^o\sigma^0$')
    ax3.set_ylabel('corrected' + r'$\sigma^0$')
    plt.savefig('ascat_as_des'+site_no+'.png', dpi=120)
    plt.close()
    # comparison between org and cor
    fig1 = plt.figure(figsize=[4, 3])
    ax = fig1.add_subplot(111)
    ax.plot(sig_4050[ob_as_slt], angle_cor_np[2][ob_as_slt], 'bo')
    ax.set_xlabel('original ' + r'$\sigma^0$')
    ax.set_ylabel('corrected ' + r'$\sigma^0$')
    fig1.tight_layout()
    plt.savefig('ascat_inc_compare'+site_no+'.png', dpi=120)
    plt.close()
    # sigma vs theta, 2 part: a. winter time, b. summer time
    theta_4050 = txt_org[:, 11]
    t_date = txt_org[:, 0][ob_as_slt]
    time_window = [t_date<120, bxy.gt_le(t_date, 150, 250)]
    if any(theta_4050>1e2):
        theta_4050 *= 1e-2
    theta_4050 -= 45
    fig5 = plt.figure(figsize=[4, 3])
    ax = fig5.add_subplot(111)
    ax.plot(-0.11 * theta_4050[ob_as_slt][time_window[0]], angle_cor_np[2][ob_as_slt][time_window[0]], 'bo')
    plot_funcs.plt_more(ax, -0.11*theta_4050[ob_as_slt][time_window[1]], angle_cor_np[2][ob_as_slt][time_window[1]], symbol='ro')
    ax.set_ylabel('corrected ' + r'$\sigma^0$')
    ax.set_xlabel('-0.11* '+r'$\theta_i-45^o$')
    fig5.tight_layout()
    plt.savefig('ascat_cor_term0'+site_no+'.png', dpi=120)
    plt.close()

    fig3 = plt.figure(figsize=[4, 3])
    ax = fig3.add_subplot(111)
    ax.plot(-0.11 * theta_4050[ob_as_slt][time_window[0]], sig_4050[ob_as_slt][time_window[0]], 'bo')
    plot_funcs.plt_more(ax, -0.11*theta_4050[ob_as_slt][time_window[1]], sig_4050[ob_as_slt][time_window[1]], symbol='ro')
    ax.set_ylabel('original ' + r'$\sigma^0$')
    ax.set_xlabel('-0.11* '+r'$\theta_i-45^o$')
    fig3.tight_layout()
    plt.savefig('ascat_cor_term2'+site_no+'.png', dpi=120)

fig2 = plt.figure(figsize=[4, 3])
ax = fig2.add_subplot(111)
x = range(1, len(inc_st[0])+1)
ax.errorbar(x, inc_st[0], inc_st[1], linestyle='None', marker='^')
ax.set_ylabel('incidence angle ' + r'$\theta_i $' + '(degree)')
ax.set_xlabel('stations')
fig2.tight_layout()
for p in inc_st:
    print p
ax.set_xlim([0, 10])
plt.savefig('ascat_inc_statistic.png', dpi=120)
plt.close()
