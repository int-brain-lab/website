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
# import time
from uuid import UUID

from flask_cors import CORS
from flask_caching import Cache
from flask import Flask, render_template, send_file, g
# from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    loader = DataLoader()
    loader.session_init(pid)
    return loader


# -------------------------------------------------------------------------------------------------
# Server
# -------------------------------------------------------------------------------------------------

def make_app():
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['CACHE_TYPE'] = 'FileSystemCache'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 0
    app.config['CACHE_THRESHOLD'] = 0
    app.config['CACHE_DIR'] = DATA_DIR / 'cache'
    CORS(app, support_credentials=True)
    # app.config.from_mapping(aconfig)
    cache = Cache(app)

    # ---------------------------------------------------------------------------------------------
    # Entry points
    # ---------------------------------------------------------------------------------------------

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

    @app.route('/api/session/<pid>/session_plot')
    @cache.cached()
    def session_overview_plot(pid):
        loader = get_data_loader(pid)

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[10, 4], wspace=0.2)

        gs0 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs[0], width_ratios=[1, 1, 8, 1, 1, 1], height_ratios=[1, 10],
                                               wspace=0.1)
        ax1 = fig.add_subplot(gs0[0, 0])
        ax2 = fig.add_subplot(gs0[1, 0])
        ax3 = fig.add_subplot(gs0[0, 1])
        ax4 = fig.add_subplot(gs0[1, 1])
        ax5 = fig.add_subplot(gs0[0, 2])
        ax6 = fig.add_subplot(gs0[1, 2])
        ax7 = fig.add_subplot(gs0[0, 3])
        ax8 = fig.add_subplot(gs0[1, 3])
        ax9 = fig.add_subplot(gs0[0, 4])
        ax10 = fig.add_subplot(gs0[1, 4])
        ax11 = fig.add_subplot(gs0[0, 5])
        ax12 = fig.add_subplot(gs0[1, 5])

        loader.plot_good_bad_clusters(ax=ax2, ax_legend=ax1, xlabel='Amp (uV)')
        loader.plot_spikes_amp_vs_depth_vs_firing_rate(ax=ax4, ax_cbar=ax3, xlabel='Amp (uV)')
        loader.plot_session_raster(ax=ax6)
        loader.plot_ap_rms(ax=ax8, ax_cbar=ax7)
        loader.plot_lfp_spectrum(ax=ax10, ax_cbar=ax9)
        loader.plot_brain_regions(ax=ax12)

        ax4.get_yaxis().set_visible(False)
        ax6.get_yaxis().set_visible(False)
        ax8.get_yaxis().set_visible(False)
        ax10.get_yaxis().set_visible(False)
        remove_frame(ax5)
        remove_frame(ax11)

        gs1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], width_ratios=[3, 1, 3, 1, 4], wspace=0.4)
        ax13 = fig.add_subplot(gs1[0, 0])
        ax14 = fig.add_subplot(gs1[0, 1])
        ax15 = fig.add_subplot(gs1[0, 2])
        ax16 = fig.add_subplot(gs1[0, 3])
        ax17 = fig.add_subplot(gs1[0, 4])
        loader.plot_psychometric_curve(ax=ax13, ax_legend=ax14)
        loader.plot_chronometric_curve(ax=ax15, ax_legend=ax16)
        loader.plot_reaction_time(ax=ax17)

        set_figure_style(fig)
        plt.subplots_adjust(top=1.02, bottom=0.05)
        plt.margins(0, 0)

        return send_figure(fig)

    @app.route('/api/session/<pid>/trial_plot/<int:trial_idx>')
    @cache.cached()
    def trial_overview_plot(pid, trial_idx):
        loader = get_data_loader(pid)
        fig, axs = plt.subplots(1, 3, figsize=(8, 4), gridspec_kw={'width_ratios': [10, 10, 1], 'wspace': 0.05})
        loader.plot_session_raster(trial_idx=trial_idx, ax=axs[0])
        loader.plot_trial_raster(trial_idx=trial_idx, ax=axs[1])
        axs[1].get_yaxis().set_visible(False)
        loader.plot_brain_regions(axs[2])
        set_figure_style(fig)
        return send_figure(fig)

    @app.route('/api/session/<pid>/cluster_plot/<int:cluster_idx>')
    @cache.cached()
    def cluster_overview_plot(pid, cluster_idx):

        loader = get_data_loader(pid)

        fig = plt.figure(figsize=(12, 5))

        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 10, 2], wspace=0.2)

        gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
        ax1 = fig.add_subplot(gs0[0, 0])

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1], height_ratios=[1, 3], hspace=0, wspace=0.2)
        ax2 = fig.add_subplot(gs1[0, 0])
        ax3 = fig.add_subplot(gs1[1, 0])
        ax4 = fig.add_subplot(gs1[0, 1])
        ax5 = fig.add_subplot(gs1[1, 1])
        ax6 = fig.add_subplot(gs1[0, 2])
        ax7 = fig.add_subplot(gs1[1, 2])

        gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2], height_ratios=[1, 1, 3], hspace=0.2)
        ax8 = fig.add_subplot(gs2[0, 0])
        ax9 = fig.add_subplot(gs2[1, 0])
        ax10 = fig.add_subplot(gs2[2, 0])

        set_figure_style(fig)

        loader.plot_spikes_amp_vs_depth(cluster_idx, ax=ax1, xlabel='Amp (uV)')

        loader.plot_correct_incorrect_single_cluster_raster(cluster_idx, axs=[ax2, ax3])
        loader.plot_left_right_single_cluster_raster(cluster_idx, axs=[ax4, ax5], ylabel0=None, ylabel1=None)
        loader.plot_contrast_single_cluster_raster(cluster_idx, axs=[ax6, ax7], ylabel0=None, ylabel1=None)
        ax5.get_yaxis().set_visible(False)
        ax7.get_yaxis().set_visible(False)

        loader.plot_autocorrelogram(cluster_idx, ax=ax8)
        loader.plot_inter_spike_interval(cluster_idx, ax=ax9, xlabel=None)
        loader.plot_cluster_waveforms(cluster_idx, ax=ax10)

        ax2.sharex(ax3)
        ax4.sharex(ax5)
        ax6.sharex(ax7)

        return send_figure(fig)

    return app


# -------------------------------------------------------------------------------------------------
# Cache generator
# -------------------------------------------------------------------------------------------------

def _get_uri(rule, values):
    try:
        return rule.build(values)[1]
    except Exception as e:
        logger.debug(f"Error: {e}")


class CacheGenerator:
    def __init__(self):
        self.app = make_app()
        self.client = self.app.test_client()

    def iter_uris(self, pid=None):
        values = {}

        for rule in self.app.url_map.iter_rules():
            args = rule.arguments

            # if a pid is specified as an argument: iterate over all rules depending on the pid
            if pid is not None and 'pid' in args:
                # for pid in pids:
                details = self.client.get(f"api/session/{pid}/details").json
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

    def visit_uri(self, uri):
        if uri:
            self.client.get(uri)

    def iter_all_uris(self):
        # URIs not depending on the pid.
        for uri in self.iter_uris():
            yield uri

        # URIs depending on the pid.
        for pid in get_pids():
            yield from self.iter_uris(pid=pid)

    def generate_cache(self):
        uris = list(self.iter_all_uris())
        for uri in tqdm(uris, desc="Generating cache"):
            self.visit_uri(uri)


if __name__ == '__main__':
    if 'cache' in sys.argv:
        gen = CacheGenerator()
        gen.generate_cache()
        exit()

    parser = argparse.ArgumentParser(description='Launch the Flask server.')
    parser.add_argument('--port', help='the TCP port')
    args = parser.parse_args()

    port = args.port or PORT
    logger.info(f"Serving the Flask application on port {port}")

    app = make_app()
    app.run('0.0.0.0', port=port)
