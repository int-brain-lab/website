# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
import base64
from pathlib import Path
import logging
import io
from math import ceil
import locale

from flask_cors import CORS, cross_origin
from flask import Flask, render_template, send_file, session, request
from psutil import pid_exists
import png
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from plots.static_plots import *

mpl.use('Agg')


# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

logger = logging.getLogger('datoviz')
mpl.style.use('seaborn')
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


# -------------------------------------------------------------------------------------------------
# Server
# -------------------------------------------------------------------------------------------------
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app, support_credentials=True)


# -------------------------------------------------------------------------------------------------
# Logic functions
# -------------------------------------------------------------------------------------------------

def get_pids():
    pids = sorted([str(p.name) for p in DATA_DIR.iterdir()])
    pids = [pid for pid in pids if not pid.startswith('.')]
    if 'README' in pids:
        pids.remove('README')
    return pids


def open_file(pid, name):
    return np.load(DATA_DIR / pid / name, mmap_mode='r')


def get_session_object(pid):
    spike_times = open_file(pid, 'spikes.times.npy')
    n_spikes = len(spike_times)

    metrics = pd.read_parquet(DATA_DIR / pid / 'clusters.table.pqt')
    metrics = metrics.reindex(metrics['cluster_id'])
    n_clusters = len(metrics)

    trials = pd.read_parquet(DATA_DIR / pid / '_ibl_trials.table.pqt')
    n_trials = len(trials)

    duration = spike_times[-1] + 1
    return {
        'pid': pid,
        # 'lab': lab,
        # 'subject_name': subject_name,
        # 'dob': dob,
        # 'probe_count': probe_count,
        'duration': duration,
        'cluster_ids': [int(_) for _ in metrics['cluster_id']],

        # TODO: return integers and not strings, move the string formatting logic to JS
        'n_clusters': f'{n_clusters:n}',
        'n_spikes': f'{n_spikes:n}',
        'n_trials': f'{n_trials:n}',
    }


def get_sessions(pids):
    return [get_session_object(pid) for pid in pids]


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


@app.route('/api/session/<pid>/details')
def session_details(pid):
    return get_session_object(pid)


@app.route('/api/session/<pid>/raster')
def raster(pid):
    spikes = load_spikes(pid)
    fig = plot_session_raster(spikes)
    return send_figure(fig)


@app.route('/api/session/<pid>/trial_raster/<int:trial_idx>')
def trial_raster(pid, trial_idx):
    spikes = load_spikes(pid)
    trials = load_trials(pid)
    # clusters = load_clusters(pid)
    fig = plot_trial_raster(spikes, trials, trial_idx)
    return send_figure(fig)


@app.route('/api/session/<pid>/cluster/<int:cluster_idx>')
def cluster_plot(pid, cluster_idx):
    clusters = load_clusters(pid)
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    plot_spikes_amp_vs_depth(clusters, cluster_idx, fig=fig, ax=axes[0])
    plot_spikes_fr_vs_depth(clusters, cluster_idx, fig=fig, ax=axes[1])
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
