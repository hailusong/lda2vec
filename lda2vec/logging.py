import logging

logger = logging.getLogger(__name__)


def logger_init(logfile):
    logger.setLevel(logging.INFO)

    # Log file
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if logfile is not None:
        fh.setFormatter(formatter)
        logger.addHandler(fh)


logger_init(None)
