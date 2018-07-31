"""
Plot functions
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
from sys import exit as quit0
import os
import h5py
import basic_xiyu as bxy
import glob
import site_infos
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, colorConverter

def pltyy(t1, s1, fname, label_y1, t2=None, s2=None, label_y2=None, symbol=['bo', 'r+'],
          label_x='Days of year', ylim=None, ylim2=None, clip_on=True, handle=[], nbins2=None,
          label2hide=None, subtick=None):
    """
    Scatter plot of time series t
    :param
        s1, s2: temporal change values
        label_y1, y2: the y label for each value
    """
    if t2 is None:
        t2 = t1
    if handle:
        fig = handle[0]
        ax1 = handle[1]
    else:
        fig, ax1 = plt.subplots()
    plot_lines_list = []  # list saving all handles of curves
    plot_lines = []  # list saving objects of handles in the list above
    line1, = ax1.plot(t1, s1, symbol[0], label=label_y1, clip_on=clip_on, linewidth=2.0, markersize=6)
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y1, color=symbol[0][0])
    plot_lines_list.append(line1)
    if ylim is not None:
        ax1.set_ylim(ylim)
    for tn in ax1.get_yticklabels():
        tn.set_color(symbol[0][0])
    if subtick is not None:
        minorLoc = MultipleLocator(5)
        ax1.xaxis.set_minor_locator(minorLoc)
# plot y2
    if s2 is not None:
        ax2 = ax1.twinx()
        line2, = ax2.plot(t2, s2, symbol[1], linewidth=2.0, markersize=5)
        if nbins2 is not None:  # set numbers of ticks in y2
            ax2.locator_params(axis='y', nbins=nbins2)
        if label_y2 is not None:  # set y2 label
            # set_label = False
            ax2.set_ylabel(label_y2)
            ax2.set_ylabel(label_y2, color=symbol[1][0])
        if label2hide is not None:  # hide y2 label and ticks
            ax2.tick_params(axis='y', which='both', bottom='off', top='off', labelright='off')
            ax2.yaxis.set_ticks_position('none')
        for tn in ax2.get_yticklabels():  # color of y2
            tn.set_color(symbol[1][0])
        plot_lines_list.append(line2)
        if ylim2 is not None:  # lim of y2
            ax2.set_ylim(ylim2)
        # plt.legend(plot_lines, ['a', 'b'], loc=4)
    else:
        ax2 = -1
    # plot_lines.append(plot_lines_list)
    # plt.legend(plot_lines[0], ['a', 'b'], loc=4)

    # bbox_to_anchor=(0., 1.02, 1., 1.02)

    plt.savefig(fname + '.png', dpi=120)
    return ax1, ax2, plot_lines_list

def plt_hist(x, date):
    """
    Plot hist of x
    """
    # mu, sigma = 100, 15
    # x = mu + sigma*np.random.randn(10000)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=True, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    # y = mlab.normpdf(bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=1)
    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    plt.axis([0, 200, 0, 0.3])
    plt.savefig('Th_%_' + date + '.png', dpi=120)
    plt.close()
    return 0


def plt_more(ax, x, y, fname=[], symbol='ro', line_list=[], marksize=3):
    """
    Add new object to the current figure
    :param ax:
    :return: the line object
    """
    line1, = ax.plot(x, y, symbol, markersize=marksize, linewidth=2.0)
    line_list.append(line1)
    # labels_cp5 = []
    # labels_cp5.append(line_list)
    # legend_cp5 = plt.legend(labels_cp5[0], ['tb_n', 'soil moisture', 'tb_gm'], loc=4)
    if len(fname) < 1:
        # no plot to be saved
        return line_list
    else:
        plt.savefig(fname + '.png', dpi=120)
        return line_list


def plt_var(keys):
    if keys == 'swe':
        t1 = 0
    return t1


def plot_all(nc, nr, varz):
    """

    :param nc:
    :param nr:
    :param varz:
        the input case list for plotting, each case includes:
        0. the num of subplot. 1. the x axis. 2. the y value list
    :return:
    """
    fig = plt.figure()
    for n in varz:
        ax1 = fig.add_subplot(nc, nr, n[0])  # plot No.
        x = n[1]
        y1 = n[2][0]
        line1, = ax1.plot(x, y1)


def plot_together():
    return 0


"""
to make plot neat
"""


def tick_doy(ax):
    oldticks = ax.get_xticklabels()
    newtick = change_tick(oldticks, mode='doy')
    ax.set_xticklabels(newtick)


def change_tick(ticks, mode='doy'):
    if mode == 'doy':
        ticklist = []
        for tick in ticks:
            ticknum = np.fromstring(tick.get_text(), dtype=int, sep=' ')
            for int_tick in ticknum:
                if int_tick <= 365:
                    ticklist.append(int_tick)
                else:
                    ticklist.append(int_tick - 365)
            # newtick_obj = datetime.datetime(2015, 1, 1) + datetime.timedelta(ticknum[0] - 1)
            # newtick = newtick_obj.strftime('%d/%m')
            # ticklist.append(newtick)
        return ticklist


def plot_filter(layer='thaw/npr', sig=[1, 2.5, 3, 4, 5, 6, 7, 8]):
    """
    id0: npr, sig or tb
    1. peaks of response to edge
    :return:
    """
    prefix = './result_07_01/methods/'
    doc_name = prefix
    flist = glob.glob(doc_name+'/*.h5')
    # initials:
    peak_nums = np.zeros([len(flist), len(sig)+1])  # row: sites, col: filter width.
    peak_dswe = np.zeros([len(flist), len(sig)+1])
    row0 = -1
    fig_test = []
    for file0 in flist:
        row0 += 1
        col0 = 0
        hf0 = h5py.File(file0)
        # add site_no to the first column
        site_str = bxy.split_strs(file0, '/')
        site_str2 =bxy.split_strs(site_str[3], '_')
        site_no = site_str2[1]
        print site_no
        peak_nums[row0, col0] = int(site_no)
        for width0 in sig:
            col0 += 1
            # data set attributes: date, value, change of sm, mean t and change of swe
            print hf0[layer].keys()
            ds0 = hf0[layer+"/"+"width_"+str(width0)]
            peak0 = np.sum(ds0[1]>0.3)
            peak_nums[row0, col0] = peak0
            if site_no == '960':
                sname = site_no
                threshold0 = ds0[1]>0.3
                x, y_dswe, y_value = ds0[0][threshold0], ds0[-1][threshold0], ds0[1][threshold0]
                fig_test.append([x, y_dswe, y_value])
                # plot
        # end for 194
    # end for 180
    fig0 = plt.figure(figsize=[15, 8])
    i=0
    for fig_data0 in fig_test:
        i+=1
        ax0 = fig0.add_subplot(3, 2, i)
        ax1 = ax0.twinx()
        ax1.plot(fig_data0[0], fig_data0[2], 'g*', markersize=5)
        ax0.bar(fig_data0[0], fig_data0[1], 2, color='r')
        ax0.set_ylabel("width_"+str(sig[i-1]))
        ax0.set_ylim([-60, 30])
        ax0.set_xlim([50, 150])
        ax1.set_ylim([0, 1.2])
    plt.savefig(doc_name+'/figures/'+sname+'_width_swe.png')
    # sorted by site_no, then saved
    peak_nums = peak_nums[peak_nums[:, 0].argsort()]
    print peak_nums[1]
    np.savetxt(doc_name+'/peak_num.txt', peak_nums, delimiter=',', fmt='%d')
    return 0


def plot_filter_series(site_no, indic='npr', sig=[1, 2.5, 5, 8], scale=1.5):
    f0_path = site_infos.get_site_h5(site_no)
    if type(f0_path) is list:
        h0 = h5py.File(f0_path[0])
    else:
        h0 = h5py.File(f0_path)
    axs = []
    # fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(5, 1)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
    axs = [ax1, ax2, ax3, ax4, ax5]
    le = [' ', '$s=1$', '$s=2.5$', '$s=5$', '$s=8$']
    if indic == 'npr':
        le[0] = 'NPR'
    elif indic == 'ascat':
        le[0] = '$\sigma^0$'
    elif indic == 'tb':
        le[0] == 'T$_{BV}$'
    l_ax = []  # legend handle
    layer_series = "all_2016/%s" % indic  # layer of npr series
    npr0 = h0[layer_series]  # time series
    l0, = ax1.plot(npr0[0], npr0[1], 'ko', markersize='3', label=le[0])
    ax1.set_ylabel(le[0])
    l_ax.append(l0)
    i_ax = 0
    for w0 in sig:
        i_ax += 1
        layer_conv = "conv/%s/width_%s" % (indic, str(w0))
        npr_w = h0[layer_conv]
        l0, = axs[i_ax].plot(npr_w[0], npr_w[2], 'g-', label=le[i_ax])
        axs[i_ax].set_ylabel(le[i_ax])
        l_ax.append(l0)
        if i_ax == 1:
            ylim = [scale*np.min(npr_w[2]), scale*np.max(npr_w[2])]
        axs[i_ax].set_ylim(ylim)
    ax1.set_xlim([0, 365])
    for axi in axs:
        axi.locator_params(axis='y', nbins=5)
    fname = './tp/%s_test_filter%s' % (indic, site_no)
    plt.savefig(fname)
    return 0


def simple_plot(tbv0):
    fig = plt.figure(figsize=[8, 5])
    ax = fig.add_subplot(111)
    ax.plot(tbv0[0], tbv0[1], 'ko', markersize=3)
    ax.set_ylabel('$T_B (K)$')
    ax.set_xlabel('Day of year 2016')
    ax.set_xlim([0, 365])
    plt.savefig('test_simple.png', dpi=300)
    return 0


def inc_plot_ascat(ascat_ob, site_no):
    t0 = ascat_ob[:, 0]
    p_win, p_su = (t0>0) & (t0<60), (t0>150)&(t0<260)
    t0, x0, y0 = ascat_ob[p_win, 0], ascat_ob[p_win, 5: 8], ascat_ob[p_win, 2:5]
    t1, x1, y1 = ascat_ob[p_su, 0], ascat_ob[p_su, 5: 8], ascat_ob[p_su, 2:5]
    label_triplets = ['fore-', 'mid-', 'aft-']
    # calculate the linear coefficient: a and b
    # scatter the incidence angles and back scatter
    fig = plt.figure(figsize=[10, 3.5*1.25])
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    ax = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
    for i, label0 in enumerate(label_triplets):
        l_win, = ax[i].plot(x0[:, i], y0[:, i], 'ko', markersize=3, markerfacecolor='none')  # winter
        l_su, = ax[i].plot(x1[:, i], y1[:, i], 'ko', markersize=3)  # summer
    axi_text = ['Fore', 'Mid', 'Aft']
    for i, axi in enumerate(ax):
        axi.set_xlim([25, 65])
        axi.set_ylim([-15, -6])
        axi.yaxis.set_major_locator(MaxNLocator(3))
        axi.xaxis.set_major_locator(MaxNLocator(5))
        axi.text(0.05, 0.1, axi_text[i], transform=axi.transAxes, va='top', ha='left', fontsize=16)
    ax[0].set_ylabel('$\sigma^0$ (dB)')
    ax[1].set_xlabel('$\\theta_i$ ($^o$)')
    leg0 = ax[0].legend([l_win, l_su], ['winter', 'summer'], loc=0, numpoints=1, prop={'size': 14})
    # leg0.get_frame().set_linewidth(0.0)
    # bbox_to_anchor=(-.15, -.3, 1., .102),loc=4, ncol=1,  numpoints=1
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    figname = 'ascat_plot_series_test_%s' % site_no
    plt.savefig(figname, dpi=300)
    fig.clear()
    plt.close()


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_Gassian(sig=7):
    size = 6*sig+1
    x = np.arange(-size/2-3, size/2+4, 0.5)
    filterz = ((-x)/sig**2)*np.exp(-x**2/(2*sig**2))
    fig = plt.figure()
    x2 = np.linspace(-size/2+1, size/2, size)
    filterz2 = ((-x2)/sig**2)*np.exp(-x2**2/(2*sig**2))
    ax = fig.add_subplot(1, 1, 1)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines["left"].set_position('center')
    ax.spines["bottom"].set_position('center')

    # Eliminate upper and right axes
    ax.spines["right"].set_color('none')
    ax.spines["top"].set_color('none')

    # show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, filterz, 'k-')
    plt.fill_between(x2, filterz2, color='grey')
    yticks = ax.yaxis.get_major_ticks()
    for yt in yticks:
        yt.label1.set_visible(False)
    xticks = ax.xaxis.get_major_ticks()
    for xt in xticks:
        xt.label1.set_visible(False)
    ax.text(x2[0], -0.005, '$-3s$', va='center')
    ax.text(x2[-1], 0.005, '$3s$', va='center')
    plt.savefig('plot_gaussian.png')


def make_cm():
    colors = [(0, 0, 1, 1), (0.5, 0, 0, 1), (1, 0, 0, 1)]
    n_bins = [3, 6, 10, 100]
    cm_name = 'my_list'
    # fig, axs = plt.subplots(2, 2, figsize=(6, 9))
    # fig.subplots_adjust()
    cm = LinearSegmentedColormap.from_list(cm_name, colors, N=100)
    return cm

def pastel(color0, weight=2.4):
    rgb = np.asarray(colorConverter.to_rgb(color0))
    #scale
    maxc = max(rgb)
    if maxc < 1.0 and maxc > 0:
        # scaled color
        scale = 1.0/maxc
        rgb = rgb*scale
    total = rgb.sum()
    slack = 0
    for x in rgb:
        slack += 1.0-x
    x = (weight - total) / slack
    rgb = [c + (x * (1.0-c)) for c in rgb]
    return rgb


def get_colors(n):
    base = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    if n <= 3:
        return base[0: n]
    # how many new colours to we need to insert between
    # red and green and between green and blue?
    needed = (((n - 3) + 1) / 2, (n - 3) / 2)

    colours = []
    for start in (0, 1):
        for x in np.linspace(0, 1, needed[start] + 2):
            colours.append((base[start] * (1.0 - x)) +
                           (base[start + 1] * x))

    return [pastel(c) for c in colours[0: n]]


def make_rgba(cm0, orig_array):
    # arr_color = ScalarMappable(cmap=cm0).to_rgba(orig_array, bytes=True)
    lower = orig_array.min()
    upper = orig_array.max()
    arr_color = plt.cm.jet((orig_array-lower)/(upper-lower))
    return arr_color


def get_colors(data_array, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(data_array))


def cust_cmap():
    cdict1 = {'red': ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))}
    cdict2 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 1.0),
                   (1.0, 0.1, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.1),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
    cdict3 = {'red':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.8, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 0.4, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.9, 0.9),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.4),
                   (0.25, 1.0, 1.0),
                   (0.5, 1.0, 0.8),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
    # Make a modified version of cdict3 with some transparency
    # in the middle of the range.
    cdict4 = cdict3.copy()
    cdict4['alpha'] = ((0.0, 1.0, 1.0),
                    #   (0.25,1.0, 1.0),
                       (0.5, 0.3, 0.3),
                    #   (0.75,1.0, 1.0),
                       (1.0, 1.0, 1.0))

    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    # second
    blue_read2 = LinearSegmentedColormap('BlueRed2', cdict2)
    plt.register_cmap(cmap=blue_read2)

    # Third, only for linear map
    plt.register_cmap(name='BlueRed3', data=cdict3)
    plt.register_cmap(name='BlueRedAlpha', data=cdict4)


def make_ticklabels_invisible(ax):
    # ax.text(0.5, 0.5, "ax0", va="center", ha="center")
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_visible(False)


def check_rgba(c, a=1.):
    array0 = colorConverter.to_rgba_array(c, alpha=a)
    print array0
    return array0


def check_rgb(c, a=1.):
    list0 = colorConverter.to_rgb(c)
    print list0
    return list0


def plot_tair_npr_onset(fname):
    xy_onset = np.loadtxt(fname, delimiter=',')
    # ax = fig0.add_subplot(1, 1, 1)
    ax = plt.subplot2grid((1, 5), (0, 0), colspan=3)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    symbs = ['o', '^', '*', 's', 'D', 'h']
    clrs = ['r', 'g', 'b']
    site_no = xy_onset[:, -1]
    for i in range(0, len(symbs)):
        for j in range(0, len(clrs)):
            if i*3+j > 16:
                break
            sitename = site_infos.change_site(str(int(site_no[i*3+j])), names=True)
            ax.plot(xy_onset[i*3+j, 0], xy_onset[i*3+j, 1], clrs[j]+symbs[i], label=sitename, markersize=10)

    # ax.plot(xy_onset[:, 0], xy_onset[:, 1], 'ko')
    # legend(ll, [key0], prop={'size': 10}, numpoints=1)
    # bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0., prop={'size': 12}
    ax.legend(bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0., prop={'size': 17}, numpoints=1)

    ax.set_xlabel('$Thawing\ onset\ (NPR)$ \n Day of year 2016')
    ax.set_ylabel('$Thawing\ onset\ (T_{air})$ \n Day of year 2016')
    ax.text(0.20, 0.85, 'Bias = 1$\pm$3 (days)', transform=ax.transAxes, va='top', fontsize=22)
    ax.plot(np.arange(1, 150), np.arange(1, 150))
    ax.set_xlim([70, 135])
    ax.set_ylim([70, 135])
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    plt.savefig('result_08_01/air_npr_onset')


def plot_comparison(fname, colnum, figname):
    ob_col, pred_col, label_col = colnum[0], colnum[1], colnum[2]
    ax = plt.subplot2grid((1, 10), (0, 0), colspan=7)
    xy_onset = np.loadtxt(fname, delimiter=',')
    with open(fname) as reader0:
        for line0 in reader0:
            heads = line0.split(',')
            break
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    symbs = ['o', '^', '*', 's', 'D', 'h']
    clrs = ['r', 'g', 'b']
    site_no = xy_onset[:, label_col]  # site_no and label
    for i in range(0, len(symbs)):
        for j in range(0, len(clrs)):
            if i*3+j > 16:
                break
            sitename = int(site_no[i*3+j])
            ax.plot(xy_onset[i*3+j, ob_col], xy_onset[i*3+j, pred_col], clrs[j]+symbs[i], label=sitename, markersize=10)


    ax.legend(bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0., prop={'size': 14}, numpoints=1)

    x_label = 'Ob. onset (%s) \n Day of year 2016' % heads[ob_col]
    y_label = 'Pred. onset (%s) \n Day of year 2016' % heads[pred_col]
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.text(0.20, 0.85, 'Bias = 1$\pm$3 (days)', transform=ax.transAxes, va='top', fontsize=22)
    ax_min = np.min(np.array([xy_onset[:, ob_col], xy_onset[:, pred_col]]))
    ax.plot(np.arange(int(ax_min)-20, int(ax_min)+80), np.arange(int(ax_min)-20, int(ax_min)+80))
    ax.set_xlim([int(ax_min)-20, int(ax_min)+80])
    ax.set_ylim([int(ax_min)-20, int(ax_min)+80])
    plt.xticks(rotation='vertical')
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    save_name = 'result_08_01/%s' % figname
    plt.savefig(save_name)
    return 0


def plot_interp_time_series(series_list, label, odd_threshold=0):
    plot_nums = len(series_list)
    if plot_nums > 5:
        print 'the number of subplots is too large:', plot_nums
        return 0
    for i in range(0, plot_nums):
        ax = plt.subplot2grid((plot_nums, 1), (i, 0))
        x_value, y_value, label0 = series_list[i][:, 0], series_list[i][:, 1], label[i]
        valid_id = y_value>odd_threshold
        ax.plot(x_value[valid_id], y_value[valid_id], label=label0)
        ax.legend()
        ax.set_ylabel(label0)
    plt.savefig('tp/temp_timeseries_0730/test.png')
    plt.close()
