import numpy as np

x0 = np.array([10., 8., 0., 10., 2.])
x0[x0<8] = np.nan
print 'isnan test', np.isnan(x0)
