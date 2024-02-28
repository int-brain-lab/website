# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
from datetime import datetime, date
import io
import json
import locale
import locale
import logging
from operator import itemgetter
import os.path as op
from pathlib import Path
import re
import sys
from uuid import UUID

import numpy as np
import pandas as pd
import png
import matplotlib as mpl
from flask_cors import CORS
from flask import Flask, render_template, send_file, Response, send_from_directory


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

ROOT_DIR = Path(__file__).parent.resolve()

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


def to_png(arr):
    p = png.from_array(arr, mode="L")
    b = io.BytesIO()
    p.write(b)
    b.seek(0)
    return b


def send_image(img):
    return send_file(to_png(img), mimetype='image/png')


def send_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def send_png_bytes(btes):
    buf = io.BytesIO(btes)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def send(path):
    if path.exists():
        return send_file(path)
    else:
        logger.error(f"path {path} does not exist")
        return Response(status=404)


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
# Image paths
# -------------------------------------------------------------------------------------------------

def session_cache_path(pid):
    cp = CACHE_DIR / pid
    cp.mkdir(exist_ok=True, parents=True)
    assert cp.exists(), f"the path `{cp}` does not exist"
    return cp


def figure_details_path():
    return CACHE_DIR / 'figures.json'


def session_details_path(pid):
    return session_cache_path(pid) / 'session.json'


def trial_details_path(pid, trial_idx):
    return session_cache_path(pid) / f'trial-{trial_idx:04d}.json'


def cluster_details_path(pid, cluster_idx):
    return session_cache_path(pid) / f'cluster-{cluster_idx:04d}.json'


def session_overview_path(pid):
    return session_cache_path(pid) / 'overview.png'


def behaviour_overview_path(pid):
    return session_cache_path(pid) / 'behaviour_overview.png'


def trial_event_overview_path(pid):
    return session_cache_path(pid) / 'trial_overview.png'


def trial_overview_path(pid, trial_idx):
    return session_cache_path(pid) / f'trial-{trial_idx:04d}.png'


def cluster_overview_path(pid, cluster_idx):
    return session_cache_path(pid) / f'cluster-{cluster_idx:04d}.png'


def cluster_pixels_path(pid):
    return session_cache_path(pid) / 'cluster_pixels.csv'


def trial_intervals_path(pid):
    return session_cache_path(pid) / f'trial_intervals.csv'


def caption_path(figure):
    return CACHE_DIR.joinpath(f'{figure}_px_locations.csv')


def get_cluster_idx_from_xy(pid, cluster_idx, x, y):
    df = pd.read_csv(cluster_pixels_path(pid))
    norm_dist = (df.x.values - x) ** 2 + (df.y.values - y) ** 2
    min_idx = np.nanargmin(norm_dist)
    if norm_dist[min_idx] < 0.005:  # TODO some limit of distance?
        return int(df.iloc[min_idx].cluster_id), int(min_idx)
    else:
        idx = np.where(df.cluster_id.values == cluster_idx)[0]
        idx = idx.squeeze()
        return int(cluster_idx), int(idx)


# -------------------------------------------------------------------------------------------------
# Data JSON generator
# -------------------------------------------------------------------------------------------------

def load_json_c(path):
    precision = 1000.0
    data = load_json(path)
    for k in data.keys():
        v = data[k]
        if isinstance(v, list) and v and isinstance(v[0], float):
            data[k] = [(int(n * precision) / precision) if n is not None else None for n in v]
    return data


def sessions():
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    pids = sorted([str(p.name) for p in CACHE_DIR.iterdir()])
    pids = [pid for pid in pids if is_valid_uuid(pid)]
    sessions = [load_json_c(session_details_path(pid)) for pid in pids]
    sessions = [_ for _ in sessions if _]
    sessions = sorted(sessions, key=itemgetter('Lab', 'Subject'))
    return sessions


def legends():
    return load_json(figure_details_path())


def generate_data_js():
    FLASK_CTX = {
        "SESSIONS": sessions(),
        "LEGENDS": legends(),
        "DEFAULT_PID": DEFAULT_PID,
        "DEFAULT_DSET": DEFAULT_DSET,
    }
    ctx_json = json.dumps(FLASK_CTX)
    # ctx_compressed = lzstring.LZString().compressToBase64(ctx_json)
    # return ctx_compressed
    return ctx_json


def make_data_js():
    ctx_json = generate_data_js()
    path = 'static/data.js'
    with open(path, 'r') as f:
        contents = f.read()
    # contents = re.sub('const FLASK_CTX_COMPRESSED = .+', f'const FLASK_CTX_COMPRESSED = "{ctx_json}";', contents)
    contents = re.sub('const FLASK_CTX = \{.+', f'const FLASK_CTX = {ctx_json};', contents)
    with open(path, 'w') as f:
        f.write(contents)


# -------------------------------------------------------------------------------------------------
# Server
# -------------------------------------------------------------------------------------------------

def make_app():
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    CORS(app, support_credentials=True)

    # ---------------------------------------------------------------------------------------------
    # Entry points
    # ---------------------------------------------------------------------------------------------

    _render = render_template

    @app.route('/')
    def main():
        return _render('app.html')

    # JSON details
    # ---------------------------------------------------------------------------------------------

    @app.route('/api/session/<pid>/details')
    def session_details(pid):
        return load_json(session_details_path(pid))

    @app.route('/api/session/<pid>/trial_details/<int:trial_idx>')
    def trial_details(pid, trial_idx):
        return load_json(trial_details_path(pid, trial_idx))

    @app.route('/api/session/<pid>/cluster_details/<int:cluster_idx>')
    def cluster_details(pid, cluster_idx):
        return load_json(cluster_details_path(pid, cluster_idx))

    @app.route('/api/session/<pid>/cluster_plot_from_xy/<int:cluster_idx>/<float(signed=True):x>_<float(signed=True):y>/')
    def cluster_from_xy(pid, cluster_idx, x, y):
        cluster_idx, idx = get_cluster_idx_from_xy(pid, cluster_idx, x, y)
        return {
            "idx": int(idx),
            "cluster_idx": int(cluster_idx),
        }

    # Figures
    # ---------------------------------------------------------------------------------------------

    @app.route('/api/session/<pid>/session_plot')
    def session_overview_plot(pid):
        return send(session_overview_path(pid))

    @app.route('/api/session/<pid>/behaviour_plot')
    def behaviour_overview_plot(pid):
        return send(behaviour_overview_path(pid))

    @app.route('/api/session/<pid>/trial_event_plot')
    def trial_event_overview_plot(pid):
        return send(trial_event_overview_path(pid))

    @app.route('/api/session/<pid>/trial_plot/<int:trial_idx>')
    def trial_overview_plot(pid, trial_idx):
        return send(trial_overview_path(pid, trial_idx))

    @app.route('/api/session/<pid>/cluster_plot/<int:cluster_idx>')
    def cluster_overview_plot(pid, cluster_idx):
        return send(cluster_overview_path(pid, cluster_idx))

    return app


if __name__ == '__main__':

    if 'make' in sys.argv:
        # Update static/data.json with the session data, extracted from the files on disk.
        make_data_js()

    else:
        parser = argparse.ArgumentParser(description='Launch the Flask server.')
        parser.add_argument('--port', help='the TCP port')
        args = parser.parse_args()

        port = args.port or PORT
        logger.info(f"Serving the Flask application on port {port}")

        app = make_app()
        # to run with SSL, generate certificate with
        # openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
        app.run(ssl_context=('cert.pem', 'key.pem'), port=port)
