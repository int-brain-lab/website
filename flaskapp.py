# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
import io
import functools
import locale
import logging
from pathlib import Path
import png
import sys
import time
from uuid import UUID

from flask_cors import CORS
from flask_caching import Cache
from flask import Flask, render_template, send_file, g
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

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
    buf = io.BytesIO()
    fig.savefig(buf)
    plt.close(fig)
    buf.seek(0)
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


@functools.cache
def get_data_loader(pid):
    # if 'loader' not in g:
    # return g.loader
    loader = DataLoader()
    loader.session_init(pid)
    return loader


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
    loader = get_data_loader(pid)
    return loader.get_session_details()


@app.route('/api/session/<pid>/trial_details/<int:trial_idx>')
@cache.cached()
def trial_details(pid, trial_idx):
    loader = get_data_loader(pid)
    return loader.get_trial_details(trial_idx)


@app.route('/api/session/<pid>/cluster_details/<int:cluster_idx>')
@cache.cached()
def cluster_details(pid, cluster_idx):
    loader = get_data_loader(pid)
    return loader.get_cluster_details(cluster_idx)


@app.route('/api/session/<pid>/raster')
@cache.cached()
def raster(pid):
    loader = get_data_loader(pid)
    fig = loader.plot_session_raster(loader.spikes)
    return send_figure(fig)


@app.route('/api/session/<pid>/psychometric')
@cache.cached()
def psychometric_curve(pid):
    loader = get_data_loader(pid)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig = loader.plot_psychometric_curve(ax=ax)
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/clusters')
@cache.cached()
def cluster_good_bad_plot(pid):
    loader = get_data_loader(pid)
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    fig = loader.plot_good_bad_clusters(ax=ax)
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/raster/trial/<int:trial_idx>')
@cache.cached()
def raster_with_trial(pid, trial_idx):
    loader = get_data_loader(pid)
    fig, axs = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [10, 1], 'wspace': 0.05})
    loader.plot_session_raster(trial_idx=trial_idx, ax=axs[0])
    loader.plot_brain_regions(axs[1])
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/trial_raster/<int:trial_idx>')
@cache.cached()
def trial_raster(pid, trial_idx):
    loader = get_data_loader(pid)
    fig, axs = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [10, 1], 'wspace': 0.05})
    loader.plot_trial_raster(trial_idx=trial_idx, ax=axs[0])
    loader.plot_brain_regions(axs[1])
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/cluster/<int:cluster_idx>')
@cache.cached()
def cluster_plot(pid, cluster_idx):
    loader = get_data_loader(pid)
    fig, axs = plt.subplots(1, 3, figsize=(9, 6), gridspec_kw={'width_ratios': [4, 4, 1], 'wspace': 0.05})
    loader.plot_spikes_amp_vs_depth(cluster_idx, ax=axs[0])
    loader.plot_spikes_fr_vs_depth(cluster_idx, ax=axs[1], ylabel=None)
    loader.plot_brain_regions(axs[2])
    axs[1].get_yaxis().set_visible(False)
    set_figure_style(fig)
    return send_figure(fig)


@app.route('/api/session/<pid>/cluster_response/<int:cluster_idx>')
@cache.cached()
def cluster_response_plot(pid, cluster_idx):
    loader = get_data_loader(pid)
    fig, axs = plt.subplots(2, 3, figsize=(9, 6), gridspec_kw={
        'height_ratios': [1, 3], 'hspace': 0, 'wspace': 0.1}, sharex=True)
    axs = axs.ravel()
    set_figure_style(fig)
    loader.plot_correct_incorrect_single_cluster_raster(cluster_idx, axs=[axs[0], axs[3]])
    loader.plot_left_right_single_cluster_raster(cluster_idx, axs=[axs[1], axs[4]], ylabel0=None, ylabel1=None)
    loader.plot_contrast_single_cluster_raster(cluster_idx, axs=[axs[2], axs[5]], ylabel0=None, ylabel1=None)
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
    loader = get_data_loader(pid)
    axs = []
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2)
    axs.append(fig.add_subplot(gs[0, 0]))
    axs.append(fig.add_subplot(gs[1, 0]))
    axs.append(fig.add_subplot(gs[:, 1]))

    set_figure_style(fig)
    loader.plot_autocorrelogram(cluster_idx, ax=axs[0], xlabel=None)
    loader.plot_inter_spike_interval(cluster_idx, ax=axs[1])
    loader.plot_cluster_waveforms(cluster_idx, ax=axs[2])

    return send_figure(fig)


# -------------------------------------------------------------------------------------------------
# Cache generator
# -------------------------------------------------------------------------------------------------

def _get_uri(rule, values):
    try:
        return rule.build(values)[1]
    except Exception as e:
        print(f"Error: {e}")


def iter_uris(pid=None):
    client = app.test_client()
    values = {}

    for rule in app.url_map.iter_rules():
        args = rule.arguments

        # if a pid is specified as an argument: iterate over all rules depending on the pid
        if pid is not None and 'pid' in args:
            # for pid in pids:
            details = client.get(f"api/session/{pid}/details").json
            cluster_ids = details['_cluster_ids']
            n_trials = int(details['N trials'])
            values['pid'] = pid
            if 'cluster_idx' in args:
                for cluster_idx in (cluster_ids):
                    values['cluster_idx'] = cluster_idx
                    yield _get_uri(rule, values)
            elif 'trial_idx' in args:
                for trial_idx in (range(n_trials)):
                    values['trial_idx'] = trial_idx
                    yield _get_uri(rule, values)
            else:
                yield _get_uri(rule, values)

        # if a pid is not specified as an argument: iterate over all rules NOT depending on the pid
        if pid is None and 'pid' not in args:
            yield _get_uri(rule, values)


def visit_uri(client, uri):
    if uri:
        print(uri)
        client.get(uri)


def generate_cache_pid(pid):
    client = app.test_client()
    for uri in iter_uris(pid):
        visit_uri(client, uri)


def generate_cache():
    client = app.test_client()

    # Single core: all URIs not depending on the pid.
    for uri in iter_uris():
        visit_uri(client, uri)

    # Distributed: URIs depending on the pid.
    parallel = Parallel(n_jobs=-2)
    parallel(delayed(generate_cache_pid)(pid) for pid in get_pids())


if __name__ == '__main__':
    if 'cache' in sys.argv:
        generate_cache()
        exit()

    parser = argparse.ArgumentParser(description='Launch the Flask server.')
    parser.add_argument('--port', help='the TCP port')
    args = parser.parse_args()

    port = args.port or PORT
    logger.info(f"Serving the Flask application on port {port}")
    app.run('0.0.0.0', port=port)
