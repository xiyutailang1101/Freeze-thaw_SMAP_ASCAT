import datetime


def doy2date(yr, doy):
    t = datetime.datetime.strptime(yr+doy, '%Y%j').timetuple()
    print t.tm_mon, '/', t.tm_mday