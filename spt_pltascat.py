__author__ = 'xiyu'
import plot_funcs
import numpy as np
siteno = ['947']
for site in siteno:
    txt_table = np.loadtxt('ascat_series_'+siteno+'.txt', delimiter=',')
    xt, sigma0_f, incf = txt_table[:, 0], txt_table[:, 3], txt_table[:, 9]
    plot_funcs.pltyy(xt, sigma0_f*1e-6, 'ascat_series_test', 'backscatter', t2=xt, s2=incf*1e-2,
                     label_y2='incidence angle fore', symbol=['b-', 'r-'])