#ms_pixel_test plotting: out_lier1
#h5_name path: prepare_files/h5/pixel_check/pixel_plot_xxxx.h5
#plot using: 1 quick_process_two_series_detect, 2 validate, and 3 plot
def quick_process_two_series_detect(smap_g=3) 
	smap_melt_initiation(smap_g=3) # get the thaw_onsets_npr
	angular_correct()
	two_series_sigma_process(gk=[7, 7, 7])
return 0

# hystory(2) get the ascat onset map of alaska
## step 1
def combine_detection_v2(gk=[5, 7, 7]):
return 0
# which call
def two_series_detect_v2(angular=True):
	smap_melt_initiation()
	two_series_sigma_process(angular=True)  # multi-process!!!
return 0

"""
prepare outlier series
"""
def quick_process_outlier_prepare(np.array([44610])):

return 0
