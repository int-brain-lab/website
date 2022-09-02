# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
# import base64
import io
import locale
import logging
# from math import ceil
from pathlib import Path
import png
import time
from uuid import UUID

from flask_cors import CORS  # , cross_origin
from flask_caching import Cache
from flask import Flask, render_template, send_file  # , session, request
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# from psutil import pid_exists

from plots.static_plots import *


# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

logger = logging.getLogger('datoviz')
mpl.use('Agg')
# mpl.style.use('seaborn')
locale.setlocale(locale.LC_ALL, '')


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data'
PORT = 4321
DATACLASS = DataLoader()


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

class Bunch(dict):
    def __init__(self, *args, **kwargs):
        self.__dict__ = self
        super().__init__(*args, **kwargs)


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


def send_image(img):
    return send_file(to_png(img), mimetype='image/png')


def send_figure(fig):
    start = time.time()
    buf = io.BytesIO()
    fig.savefig(buf)  # , dpi=100)
    plt.close(fig)
    buf.seek(0)
    print(time.time() - start)
    return send_file(buf, mimetype='image/png')


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


# -------------------------------------------------------------------------------------------------
# Server
# -------------------------------------------------------------------------------------------------
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['CACHE_TYPE'] = 'FileSystemCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 0
app.config['CACHE_DIR'] = DATA_DIR / 'cache'
CORS(app, support_credentials=True)
# app.config.from_mapping(aconfig)
cache = Cache(app)


# -------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------

def get_pids():
    pids = sorted([str(p.name) for p in DATA_DIR.iterdir()])
    pids = [pid for pid in pids if is_valid_uuid(pid)]
    return pids


def open_file(pid, name):
    return np.load(DATA_DIR / pid / name, mmap_mode='r')


def get_sessions(pids):
    return [{'pid': pid} for pid in pids]


def get_js_context():
    return {}


# -------------------------------------------------------------------------------------------------
# Entry points
# -------------------------------------------------------------------------------------------------

@app.route('/')
def main():
    return render_template(
        'index.html',
        sessions=get_sessions(get_pids()),
        js_context=get_js_context(),
    )


@app.route('/app')
def the_app():
    return render_template(
        'app.html',
        sessions=get_sessions(get_pids()),
        js_context=get_js_context(),
    )


@app.route('/api/session/<pid>/details')
def session_details(pid):
    DATACLASS.session_init(pid)
    return DATACLASS.get_session_details()


@app.route('/api/session/<pid>/trial_details/<int:trial_idx>')
@cache.cached()
def trial_details(pid, trial_idx):
    return DATACLASS.get_trial_details(trial_idx)


@app.route('/api/session/<pid>/cluster_details/<int:cluster_idx>')
@cache.cached()
def cluster_details(pid, cluster_idx):
    return DATACLASS.get_cluster_details(cluster_idx)


@app.route('/api/session/<pid>/raster')
@cache.cached()
def raster(pid):
    fig = DATACLASS.plot_session_raster(DATACLASS.spikes)
    return send_figure(fig)


@app.route('/api/session/<pid>/psychometric')
@cache.cached()
def psychometric_curve(pid):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig = DATACLASS.plot_psychometric_curve(ax=ax)
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/clusters')
@cache.cached()
def cluster_good_bad_plot(pid):
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    fig = DATACLASS.plot_good_bad_clusters(ax=ax)
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/raster/trial/<int:trial_idx>')
@cache.cached()
def raster_with_trial(pid, trial_idx):
    fig, axs = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [10, 1], 'wspace': 0.05})
    DATACLASS.plot_session_raster(trial_idx=trial_idx, ax=axs[0])
    DATACLASS.plot_brain_regions(axs[1])
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/trial_raster/<int:trial_idx>')
@cache.cached()
def trial_raster(pid, trial_idx):
    fig, axs = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [10, 1], 'wspace': 0.05})
    DATACLASS.plot_trial_raster(trial_idx=trial_idx, ax=axs[0])
    DATACLASS.plot_brain_regions(axs[1])
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/cluster/<int:cluster_idx>')
@cache.cached()
def cluster_plot(pid, cluster_idx):
    fig, axs = plt.subplots(1, 3, figsize=(9, 6), gridspec_kw={'width_ratios': [4, 4, 1], 'wspace': 0.05})
    DATACLASS.plot_spikes_amp_vs_depth(cluster_idx, ax=axs[0])
    DATACLASS.plot_spikes_fr_vs_depth(cluster_idx, ax=axs[1], ylabel=None)
    DATACLASS.plot_brain_regions(axs[2])
    axs[1].get_yaxis().set_visible(False)
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/cluster_response/<int:cluster_idx>')
@cache.cached()
def cluster_response_plot(pid, cluster_idx):
    fig, axs = plt.subplots(2, 3, figsize=(9, 6), gridspec_kw={
        'height_ratios': [1, 3], 'hspace': 0, 'wspace': 0.1}, sharex=True)
    axs = axs.ravel()
    set_figure_style(fig)
    DATACLASS.plot_correct_incorrect_single_cluster_raster(cluster_idx, axs=[axs[0], axs[3]])
    DATACLASS.plot_left_right_single_cluster_raster(cluster_idx, axs=[axs[1], axs[4]], ylabel0=None, ylabel1=None)
    DATACLASS.plot_contrast_single_cluster_raster(cluster_idx, axs=[axs[2], axs[5]], ylabel0=None, ylabel1=None)
    axs[1].get_yaxis().set_visible(False)
    axs[4].get_yaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)
    axs[5].get_yaxis().set_visible(False)

    axs[1].sharex(axs[0])
    axs[2].sharex(axs[0])

    return send_figure(fig)


@app.route('/api/session/<pid>/cluster_properties/<int:cluster_idx>')
@cache.cached()
def cluster_properties_plot(pid, cluster_idx):
    axs = []
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2)
    axs.append(fig.add_subplot(gs[0, 0]))
    axs.append(fig.add_subplot(gs[1, 0]))
    axs.append(fig.add_subplot(gs[:, 1]))

    set_figure_style(fig)
    DATACLASS.plot_autocorrelogram(cluster_idx, ax=axs[0], xlabel=None)
    DATACLASS.plot_inter_spike_interval(cluster_idx, ax=axs[1])
    DATACLASS.plot_cluster_waveforms(cluster_idx, ax=axs[2])

    return send_figure(fig)


# -------------------------------------------------------------------------------------------------
# Raw ephys data server
# -------------------------------------------------------------------------------------------------

# @app.route('/<eid>')
# @cross_origin(supports_credentials=True)
# def cluster_plot(eid):
#     fig, ax = plt.subplots(1, 1, figsize=(9, 6))
#     x = np.random.randn(1000)
#     y = np.random.randn(1000)
#     ax.plot(x, y, 'o')
#     out = send_figure(fig)
#     plt.close(fig)
#     return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Launch the Flask server.')
    parser.add_argument('--port', help='the TCP port')
    args = parser.parse_args()

    port = args.port or PORT
    logger.info(f"Serving the Flask application on port {port}")
    app.run('0.0.0.0', port=port)
