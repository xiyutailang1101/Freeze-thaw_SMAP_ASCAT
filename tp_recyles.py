import basic_xiyu as by1
import numpy as np
import h5py
import os
import read_site
date = [85, 85.875, 95, 95.875, 115.875, 277.875, 279.875,947.000]
date = [a for a in range(95, 105)]
date = [98+a*0.125 for a in range(0, 24)]
date = [50+18/24.0 + a for a in range(0, 10)]
for d0 in date[0: ]:
    opt = read_site.search_snotel('2213', d0)