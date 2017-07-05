"""
Plot functions
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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
    line1, = ax1.plot(t1, s1, symbol[0], label=label_y1, clip_on=clip_on, linewidth=2.0)
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
        line2, = ax2.plot(t2, s2, symbol[1], linewidth=2.0)
        if nbins2 is not None:  # set numbers of ticks in y2
            ax2.locator_params(axis='y', nbins=nbins2)
        if label_y2 is not None:  # set y2 label
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
    return [ax1, ax2], plot_lines_list

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
    :return:
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