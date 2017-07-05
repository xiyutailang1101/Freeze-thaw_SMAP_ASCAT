import numpy as np
import os

siteno = 'alaska02/all_orb'
path = 'result_03_4/'+siteno+'/'
txtname = os.listdir(path)
for txt in txtname:
    value = np.loadtxt(path+txt, delimiter=',')
    if value.size > 4:
        value_as = value[value[:, -1] < 1, :]
        np.save('./result_03_4/alaska02/ascend/AS_'+txt[0:-4], value_as)
        value_des = value[value[:, -1] > 0, :]
        np.save('./result_03_4/alaska02/descend/DES_'+txt[0:-4], value_des)
        # np.savetxt('./result_03_4/alaska02/DES_'+txt[0:-4]+'.txt', value_des,
        #            delimiter=',',
        #            header='lat, lon, sigma0, sigma1, sigma2, flag0, flag1, flag2, inc0, inc1, inc2, land0, land2, land3, ob',
        #            fmt='%.6f, %.6f, %.2f, %.2f, %.2f, %d, %d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %d')
