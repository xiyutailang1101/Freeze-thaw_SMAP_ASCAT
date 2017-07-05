"""
Oct.21, 2016
"""
# new version of hdf5 processed file, include the horizontal polarization, both radar and radiometer.
# but only s950 was replaced.
# the get_attribute function was modified.

"""
Oct.26, 2016
"""
# test the dscending values, only tb was considered, regardless of sigma data

"""
Oct.31, 2016
"""
# Add more radiometer data, to find better winter references.
#	At some certain date, the in situ measurements were missing. Should find
#	these days then eleminate the Remote Sensing observation at these days

"""
Nov.8th
"""
# Using the smooth function, <n_smooth>
# Adding a gaussian 1st derivative function, <gauss_conv>
# both of 'em are in data_process module

"""
Nov. 20th
"""
# 01 Find pixels' close to stations. Based on locs of station, return the pixels nearby, in forms of index in hdf5 files.
#	therefore, the return of lat/lon is unneccessary. Only index matters, cz here we cannot decide what fields to be read.
#   After obtaining the index, we can use it to read data by iterating elements in a list, which contains attributes we want to know.
# 02 site 960, 962 newly added.
# 03 try some d. orbit be cz of wet snow in the A pass (18:00)
# site was found twice in two adjacent swathes, cause an error of creating h5 data set repeatly.
# so first read all site-has-been-found pixels in all possible swathes, then create h5 dataset to store them in forms of list or ndarray
# original code is:
	if status == -1:  # -1 means not found with specified lat and lon
		print tb_name
	else:
		hf3 = h5py.File("Site_"+site_info[0] + orbit + time_str + ".h5", "a")  # new a hdf5 file when the site was found
		new_h5_name = hf3.filename
		for att in tb_attr_name:
			tb_att, status = read_tb(hf, att, Site_ind)
			hf3[proj_g + att] = tb_att
		hf3["Global_Projection/site_loc"] = [site_info[1], site_info[2]]
		hf3[proj_g + "/tb_cell_lat"] = c_lat_g
		hf3[proj_g + "/tb_cell_lon"] = c_lon_g
		hf3.close()
# new coding: assign a dictionary to saving the pixels infos (value and location)
