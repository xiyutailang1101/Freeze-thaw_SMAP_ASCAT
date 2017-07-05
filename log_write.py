__author__ = 'xiyu'


def initlog(logname):
    import logging
    logger = logging.getLogger()
    LOG_FILE = logname
    hdlr = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.NOTSET)
    return logger
# July 31, 2016
# revised:
def read_series():
	# deleting and adding outputs
	return 0
