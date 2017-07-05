import test_def
import spt_alaska as ak
import os
import data_process
import numpy as np
# x1 = np.loadtxt('sigma_thaw53.txt').reshape(26575, -1)
# np.save('./temp_save/sigma53', x1)
# del x1
# x1 = np.loadtxt('sigma_thaw54.txt').reshape(23691, -1)
# np.save('./temp_save/sigma54', x1)
# del x1
# a1 = np.loadtxt('angle_thaw53.txt').reshape(26575, -1)
# np.save('./temp_save/angle53', a1)
# del a1
# a1 = np.loadtxt('angle_thaw54.txt').reshape(23691, -1)
# np.save('./temp_save/angle54', a1)
# del a1
# sys.exit()
path = './result_03_4/alaska02/npy/'
path_daily = 'result_03_4/alaska02/ascend/'
fname_list = sorted(os.listdir(path_daily))
doy=[]
for f in fname_list:
    doy.append(data_process.get_doy([f[9:17]])[0]-365)
for i in [0, 1, 2, 3]:
    a = 1
    # ak.edge_detect([path+'sigma_'+str(i)+'.npy', path+'angle_'+str(i)+'.npy', path+'land_'+str(i)+'.npy'],
    #                np.array(doy[i:]))
    # break
    ak.main(i)
    break
    test_def.main([0, 1], ['20160101', '20161225'], sm_wind=7, mode='annual')
