import os
import shutil
import glob
id_list = ['947', '948', '949', '950', '1090']
orbit = ['A', 'D']
for id in id_list:
	os.mkdir('s' + id)
	for h5file in glob.glob('Site_' + id + '*.h5'):
		if h5file:
			print h5file
			shutil.copyfile(h5file, 's' + id + '/' + h5file)
