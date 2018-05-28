import Read_radar
import spt_quick
import sys

site_nos = ['1177', '1233', '947', '2065', '967', '2213', '949', '950', '960', '962', '968','1090', '1175', '2081', '2210', '1089',  '2212', '2211']
site_nos = ['region_14']
doylist = Read_radar.get_peroid('20151101', '20170401')
for doyz in doylist:
    Read_radar.getascat(site_nos, doyz-365)
    #Read_radar.read_ascat_alaska(doyz-365, orbit=2)



# # ASCAT process
# n_pixel = []
#
# doylist = Read_radar.get_peroid('20160101', '20161201')
# for doyz in doylist:
#     Read_radar.getascat(site_nos, doyz-365, orbit=2)  # orbit 0: des, orbit 1: as, contrary to ascat_formatvim



