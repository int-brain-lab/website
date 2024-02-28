# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import lzstring
from datetime import datetime, date
from uuid import UUID
import io
import json
import locale
import logging
from operator import itemgetter
import os.path as op
import png
import re
import sys

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plots.static_plots import *
from plots.captions import CAPTIONS


# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

mpl.use('Agg')
locale.setlocale(locale.LC_ALL, '')


# -------------------------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------------------------

_logger_fmt = '%(asctime)s.%(msecs)03d [%(levelname)s] %(caller)s %(message)s'
_logger_date_fmt = '%H:%M:%S'


class _Formatter(logging.Formatter):
    def format(self, record):
        # Only keep the first character in the level name.
        record.levelname = record.levelname[0]
        filename = op.splitext(op.basename(record.pathname))[0]
        record.caller = '{:s}:{:d}'.format(filename, record.lineno).ljust(20)
        message = super(_Formatter, self).format(record)
        color_code = {'D': '90', 'I': '0', 'W': '33', 'E': '31'}.get(record.levelname, '7')
        message = '\33[%sm%s\33[0m' % (color_code, message)
        return message


def add_default_handler(logger, level='DEBUG'):
    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = _Formatter(fmt=_logger_fmt, datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


logger = logging.getLogger('ibl_website')
logger.setLevel(logging.DEBUG)
add_default_handler(logger, level='DEBUG')


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

# DATA_DIR contains 1 subfolder per insertion (subfolder name = pid), each subfolder contains
# the data in ONE/npy format.
DATA_DIR = ROOT_DIR / 'static/data'

# CACHE_DIR contains the generated PNG figures, with 1 subfolder per insertion.
CACHE_DIR = ROOT_DIR / 'static/cache'

PORT = 4321
DEFAULT_PID = '1ab86a7f-578b-4a46-9c9c-df3be97abcca'
DEFAULT_DSET = 'bwm'


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

class Bunch(dict):
    def __init__(self, *args, **kwargs):
        self.__dict__ = self
        super().__init__(*args, **kwargs)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


def normalize(x, target='float'):
    m = x.min()
    M = x.max()
    if m == M:
        # logger.warning("degenerate values")
        m = M - 1
    if target == 'float':  # normalize in [-1, +1]
        return -1 + 2 * (x - m) / (M - m)
    elif target == 'uint8':  # normalize in [0, 255]
        return np.round(255 * (x - m) / (M - m)).astype(np.uint8)
    raise ValueError("unknow normalization target")


def to_png(arr):
    p = png.from_array(arr, mode="L")
    b = io.BytesIO()
    p.write(b)
    b.seek(0)
    return b


def is_valid_uuid(uuid_to_test, version=4):
    """
    Check if uuid_to_test is a valid UUID.
    https://stackoverflow.com/a/33245493/1595060

     Parameters
    ----------
    uuid_to_test : str
    version : {1, 2, 3, 4}

     Returns
    -------
    `True` if uuid_to_test is a valid UUID, otherwise `False`.

     Examples
    --------
    >>> is_valid_uuid('c9bf9e57-1685-4c89-bafb-ff5af830be8a')
    True
    >>> is_valid_uuid('c9bf9e58')
    False
    """

    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test


def save_json(path, dct, **kwargs):
    with open(path, 'w') as f:
        json.dump(dct, f, sort_keys=True, cls=DateTimeEncoder, **kwargs)


def load_json(path):
    if not path.exists():
        logger.warning(f"file {path} doesn't exist")
        return {}
    with open(path, 'r') as f:
        return json.load(f)


# -------------------------------------------------------------------------------------------------
# Session iterator
# -------------------------------------------------------------------------------------------------

def get_pids():
    """UUID subfolder names in DATA_DIR."""
    children = list(DATA_DIR.iterdir())
    names = [f.name for f in children if is_valid_uuid(f.name)]
    return names


def iter_session():
    yield from get_pids()


def get_subplot_position(ax1, ax2):
    xmin_ymax = ax1.get_position().corners()[1]
    xmax_ymin = ax2.get_position().corners()[2]

    return np.r_[xmin_ymax, xmax_ymin]


if __name__ == '__main__':

    argv = sys.argv

    # Regenerate all figures.
    if len(argv) == 1:
        # Parallel version
        # Parallel(n_jobs=-4)(delayed(make_all_plots)(pid) for pid in iter_session())

        # Serial version
        for pid in iter_session():
            make_all_plots(pid)

    # Regenerate some figures for all sessions.
    elif len(argv) >= 2 and not is_valid_uuid(argv[1]):
        which = argv[1]

        if which == 'captions':
            make_captions()
        else:
            # which figure numbers to regenerate
            nums = list(map(int, which.split(',')))
            logger.info(f"Regenerating figures {', '.join('#%d' % _ for _ in nums)}")

            # [make_all_plots(pid, nums=nums) for pid in iter_session()]
            Parallel(n_jobs=-3)(delayed(make_all_plots)(pid, nums=nums) for pid in iter_session())

    # Regenerate figures for 1 session.
    elif len(argv) >= 2 and is_valid_uuid(argv[1]):
        nums = tuple(map(int, argv[2].split(','))) if len(argv) >= 3 else ()
        make_all_plots(argv[1], nums=nums)

    # Regenerate static/data.js with all the data.
    make_data_js()
