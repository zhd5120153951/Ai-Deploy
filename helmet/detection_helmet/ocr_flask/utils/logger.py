
import logging
import os
import datetime

Color = {
    'RED': '\033[31m',
    'HEADER': '\033[35m',  # deep purple
    'PURPLE': '\033[95m',  # purple
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m'
}

    
def time_zone(sec, fmt):
    real_time = datetime.datetime.now()
    return real_time.timetuple()


logging.Formatter.converter = time_zone
_logger = logging.getLogger(__name__)


def coloring(message, color="OKGREEN"):
    assert color in Color.keys()
    if os.environ.get('PADDLECLAS_COLORING', False):
        return Color[color] + str(message) + Color["ENDC"]
    else:
        return message


def anti_fleet(log):
    """
    logs will print multi-times when calling Fleet API.
    Only display single log and ignore the others.
    """

    def wrapper(fmt, *args):
        if int(os.getenv("PADDLE_TRAINER_ID", 0)) == 0:
            log(fmt, *args)

    return wrapper


@anti_fleet
def info(fmt, *args):
    _logger.info(fmt, *args)


@anti_fleet
def warning(fmt, *args):
    _logger.warning(coloring(fmt, "RED"), *args)


@anti_fleet
def error(fmt, *args):
    _logger.error(coloring(fmt, "FAIL"), *args)


def advertise():
    info(coloring("\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n".format
           ("""
              _  _           _                      
  | || |  __ _   | |__    __ _    _ _    
   \_, | / _` |  | '_ \  / _` |  | ' \   
  _|__/  \__,_|  |_.__/  \__,_|  |_||_|  """, ), "OKBLUE"))
