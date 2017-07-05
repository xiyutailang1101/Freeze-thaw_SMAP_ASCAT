import plot_funcs
import numpy as np
import matplotlib.pyplot as plt
import read_site
siteno = ['947', '1175', '950', '2065', '967', '2213', '949', '950', '960', '962', '968','1090',  '1177',  '2081', '2210', '1089', '1233', '2212', '2211']
angle_range = np.arange(25, 65, 0.1)
for site in siteno:
    txtname = 'ascat_series_'+site+'.txt'
    txt_table = np.loadtxt(txtname, delimiter=',')

    out = read_site.read_ascat_txt(txtname)

    # controled by inci
    xt, y1t, y2t = np.array(out[0]), np.array(out[1]), np.array(out[2])
    # 4 periods
    tx = txt_table[:,0]  # time line
    p1, p2, p3, p4 = np.where(tx<80), np.where((tx>90)&(tx<120)), np.where((tx>160)&(tx<250)), np.where(tx>270)
    p_no = 0
    #p_con = (p1[0]+p3[0], p1[1]+p3[1])
    p_con = (np.append(p1[0], p3[0]),)
    for p in [p1, p2, p3, p4, p_con]:
        p_no += 1
        fig = plt.figure(figsize=[4, 3])
        ax = fig.add_subplot(111)
        inci1 = txt_table[p, 9:12].reshape(-1, 3)
        sig1 = txt_table[p, 3:6].reshape(-1, 3)
        if any(inci1[0] > 1e2):
            inci1*=1e-2
        if any(sig1[0] < -1e4):
            sig1*=1e-6
        ax.plot(inci1[:,0], sig1[:,0], 'bo')
        plot_funcs.plt_more(ax, inci1[:,1], sig1[:,1])
        plot_funcs.plt_more(ax, inci1[:,2], sig1[:,2], symbol='go')

        # linear regression
        x = np.concatenate((inci1[:,0].T, inci1[:,1].T, inci1[:,2].T))
        y = np.concatenate((sig1[:,0].T, sig1[:,1].T, sig1[:,2].T))
        a, b = np.polyfit(x, y, 1)
        f = np.poly1d([a, b])
         # r squared
        y_mean = np.sum(y)/y.size
        sstot = np.sum((y-y_mean)**2)
        ssres = np.sum((y-f(x))**2)
        r2 = 1 - ssres/sstot
        fig.text(0.25, 0.25, 'y = %.2f x + %.f\n r2 = %.4f'%(a, b, r2))
        plot_funcs.plt_more(ax, x, f(x), symbol='r--', fname='Incidence_angle_p'+str(p_no))
        fig.clear()
        plt.close()
    fig2 = plt.figure(figsize=[4, 3])
    ax = fig2.add_subplot(111)
    n = 0
    for p in [p1, p3]:
        inci1 = txt_table[p, 9:12].reshape(-1, 3)
        sig1 = txt_table[p, 3:6].reshape(-1, 3)
        if any(inci1[0] > 1e2):
            inci1*=1e-2
        if any(sig1[0] < -1e4):
            sig1*=1e-6
        # linear regression
        # x = np.concatenate((inci1[:,0].T, inci1[:,1].T, inci1[:,2].T))
        x = inci1[:, 1].T
        # y = np.concatenate((sig1[:,0].T, sig1[:,1].T, sig1[:,2].T))
        y = sig1[:, 1].T
        a, b = np.polyfit(x, y, 1)
        f = np.poly1d([a, b])
         # r squared
        y_mean = np.sum(y)/y.size
        sstot = np.sum((y-y_mean)**2)
        ssres = np.sum((y-f(x))**2)
        r2 = 1 - ssres/sstot
        if n < 1:
            ax.plot(inci1[:,0], sig1[:,0], 'ko', markersize=5)
            plot_funcs.plt_more(ax, inci1[:,1], sig1[:,1], symbol='ko', marksize=5)
            plot_funcs.plt_more(ax, inci1[:,2], sig1[:,2], symbol='ko', marksize=5)
            #ax.set_ylim([-16, -6])
            plot_funcs.plt_more(ax, angle_range, f(angle_range), symbol='r--', marksize=5)
            fig2.text(0.15, 0.15, 'y = %.2f x + %.f\n r2 = %.4f' % (a, b, r2))
        else:
            plot_funcs.plt_more(ax, inci1[:,0], sig1[:,0], symbol='k^', marksize=5)
            plot_funcs.plt_more(ax, inci1[:,1], sig1[:,1], symbol='k^', marksize=5)
            plot_funcs.plt_more(ax, inci1[:,2], sig1[:,2], symbol='k^', marksize=5)
            plot_funcs.plt_more(ax, angle_range, f(angle_range), symbol='b--', marksize=5)
            fig2.text(0.6, 0.75, 'y = %.2f x + %.f\n r2 = %.4f' % (a, b, r2))
        n += 1
    ax.set_ylim([-18, -6])
    plt.savefig('Incidence_angle_'+site+'.png', dpi=120)
    plt.close()

    #plt.savefig('Incidence_angle_test.png', dpi=120)
    #p_con = (p1[0]+p3[0], p1[1]+p3[1])
    continue
# """
# Show how to make date plots in matplotlib using date tick locators and
# formatters.  See major_minor_demo1.py for more information on
# controlling major and minor ticks
# # __author__ = 'xiyu'
# """
# from __future__ import print_function
# import datetime
# import matplotlib.pyplot as plt
# from matplotlib.dates import MONDAY
# from matplotlib.finance import quotes_historical_yahoo_ochl
# from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
# import datetime
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import matplotlib.cbook as cbook
#
# def date_demo2():
#     date1 = datetime.date(2002, 1, 5)
#     date2 = datetime.date(2003, 12, 1)
#
#     # every monday
#     mondays = WeekdayLocator(MONDAY)
#
#     # every 3rd month
#     months = MonthLocator(range(1, 13), bymonthday=1, interval=3)
#     monthsFmt = DateFormatter("%b '%y")
#
#
#     quotes = quotes_historical_yahoo_ochl('INTC', date1, date2)
#     if len(quotes) == 0:
#         print('Found no quotes')
#         raise SystemExit
#
#     dates = [q[0] for q in quotes]
#     opens = [q[1] for q in quotes]
#
#     fig, ax = plt.subplots()
#     ax.plot_date(dates, opens, '-')
#     ax.xaxis.set_major_locator(months)
#     ax.xaxis.set_major_formatter(monthsFmt)
#     ax.xaxis.set_minor_locator(mondays)
#     ax.autoscale_view()
#     #ax.xaxis.grid(False, 'major')
#     #ax.xaxis.grid(True, 'minor')
#     ax.grid(True)
#
#     fig.autofmt_xdate()
#
#     plt.show()
#
#
# def date_demo():
#     """
# Show how to make date plots in matplotlib using date tick locators and
# formatters.  See major_minor_demo1.py for more information on
# controlling major and minor ticks
#
# All matplotlib date plotting is done by converting date instances into
# days since the 0001-01-01 UTC.  The conversion, tick locating and
# formatting is done behind the scenes so this is most transparent to
# you.  The dates module provides several converter functions date2num
# and num2date
#
# """
#     years = mdates.YearLocator()   # every year
#     months = mdates.MonthLocator()  # every month
#     yearsFmt = mdates.DateFormatter('%Y')
#
#     # load a numpy record array from yahoo csv data with fields date,
#     # open, close, volume, adj_close from the mpl-data/example directory.
#     # The record array stores python datetime.date as an object array in
#     # the date column
#     datafile = cbook.get_sample_data('goog.npy')
#     try:
#         # Python3 cannot load python2 .npy files with datetime(object) arrays
#         # unless the encoding is set to bytes. Hovever this option was
#         # not added until numpy 1.10 so this example will only work with
#         # python 2 or with numpy 1.10 and later.
#         r = np.load(datafile, encoding='bytes').view(np.recarray)
#     except TypeError:
#         r = np.load(datafile).view(np.recarray)
#
#     fig, ax = plt.subplots()
#     ax.plot(r.date, r.adj_close)
#
#
#     # format the ticks
#     ax.xaxis.set_major_locator(years)
#     ax.xaxis.set_major_formatter(yearsFmt)
#     ax.xaxis.set_minor_locator(months)
#
#     datemin = datetime.date(r.date.min().year, 1, 1)
#     datemax = datetime.date(r.date.max().year + 1, 1, 1)
#     ax.set_xlim(datemin, datemax)
#
#
#     # format the coords message box
#     def price(x):
#         return '$%1.2f' % x
#     ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
#     ax.format_ydata = price
#     ax.grid(True)
#
#     # rotates and right aligns the x labels, and moves the bottom of the
#     # axes up to make room for them
#     fig.autofmt_xdate()
#
#     plt.show()