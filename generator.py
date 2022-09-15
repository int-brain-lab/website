# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
from datetime import datetime, date
import functools
import io
import json
import locale
import logging
from pathlib import Path
import png
import sys
from uuid import UUID

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plots.static_plots import *


# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

logger = logging.getLogger('ibl_website')
mpl.use('Agg')
# mpl.style.use('seaborn')
locale.setlocale(locale.LC_ALL, '')


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data'
CACHE_DIR = ROOT_DIR / 'cache'
PORT = 4321


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


def save_json(path, dct):
    with open(path, 'w') as f:
        json.dump(dct, f, sort_keys=True, cls=DateTimeEncoder)


def load_json(path):
    if not path.exists():
        logger.error(f"file {path} doesn't exist")
        return {}
    with open(path, 'r') as f:
        return json.load(f)


# -------------------------------------------------------------------------------------------------
# Path functions
# -------------------------------------------------------------------------------------------------

def session_data_path(pid):
    return DATA_DIR / pid


def session_cache_path(pid):
    return CACHE_DIR / pid


def session_details_path(pid):
    return session_cache_path(pid) / 'session.json'


def trial_details_path(pid, trial_idx):
    return session_cache_path(pid) / f'trial-{trial_idx:04d}.json'


def cluster_details_path(pid, cluster_idx):
    return session_cache_path(pid) / f'cluster-{cluster_idx:04d}.json'


def session_overview_path(pid):
    return session_cache_path(pid) / 'overview.png'


def trial_overview_path(pid, trial_idx):
    return session_cache_path(pid) / f'trial-{trial_idx:04d}.png'


def cluster_overview_path(pid, cluster_idx):
    return session_cache_path(pid) / f'cluster-{cluster_idx:04d}.png'


# -------------------------------------------------------------------------------------------------
# Session iterator
# -------------------------------------------------------------------------------------------------

def get_pids():
    pids = sorted([str(p.name) for p in DATA_DIR.iterdir()])
    pids = [pid for pid in pids if is_valid_uuid(pid)]
    return pids


def iter_session():
    yield from get_pids()


# -------------------------------------------------------------------------------------------------
# Plot and JSON generator
# -------------------------------------------------------------------------------------------------

DEBUG = 5


class Generator:
    def __init__(self, pid):
        self.dl = DataLoader()
        self.dl.session_init(pid)
        self.pid = pid

        # Ensure the session cache folder exists.
        session_cache_path(pid).mkdir(exist_ok=True, parents=True)

        # Load the session details.
        path = session_details_path(pid)
        self.session_details = self.dl.get_session_details()

        # Save the session details to a JSON file.
        logger.debug(f"saving session details for session {pid}")
        save_json(path, self.session_details)

        self.n_trials = int(self.session_details['N trials'])
        self.cluster_idxs = self.session_details['_cluster_ids']
        self.n_clusters = len(self.cluster_idxs)

    # Iterators
    # -------------------------------------------------------------------------------------------------

    def iter_trial(self):
        yield from range(DEBUG)  # self.n_trials)

    def iter_cluster(self):
        yield from sorted(self.cluster_idxs)[:DEBUG]

    # Saving JSON details
    # -------------------------------------------------------------------------------------------------

    def save_trial_details(self, trial_idx):
        logger.debug(f"saving trial details for session {self.pid}")
        details = self.dl.get_trial_details(trial_idx)
        path = trial_details_path(self.pid)
        save_json(path, details)

    def save_cluster_details(self, cluster_idx):
        logger.debug(f"saving cluster details for session {self.pid}")
        details = self.dl.get_cluster_details(cluster_idx)
        path = cluster_details_path(self.pid)
        save_json(path, details)

    # Plot functions
    # -------------------------------------------------------------------------------------------------

    def make_session_plot(self):
        path = session_overview_path(self.pid)
        if path.exists():
            return
        logger.debug(f"making session overview plot for session {self.pid}")
        loader = self.dl

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
        fig.subplots_adjust(top=1.02, bottom=0.05)

        fig.savefig(path)
        plt.close(fig)

    def make_trial_plot(self, trial_idx):
        path = trial_overview_path(self.pid, trial_idx)
        if path.exists():
            return
        logger.debug(f"making trial overview plot for session {self.pid}, trial #{trial_idx:04d}")
        loader = self.dl

        fig, axs = plt.subplots(1, 3, figsize=(8, 4), gridspec_kw={'width_ratios': [10, 10, 1], 'wspace': 0.05})
        loader.plot_session_raster(trial_idx=trial_idx, ax=axs[0])
        loader.plot_trial_raster(trial_idx=trial_idx, ax=axs[1])
        axs[1].get_yaxis().set_visible(False)
        loader.plot_brain_regions(axs[2])
        set_figure_style(fig)

        fig.savefig(path)
        plt.close(fig)

    def make_cluster_plot(self, cluster_idx):
        path = cluster_overview_path(self.pid, cluster_idx)
        if path.exists():
            return
        logger.debug(f"making cluster overview plot for session {self.pid}, cluster #{cluster_idx:04d}")
        loader = self.dl

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

        fig.savefig(path)
        plt.close(fig)

    # Plot generator functions
    # -------------------------------------------------------------------------------------------------

    def make_all_trial_plots(self):
        desc = "Making all trial plots  "
        for trial_idx in tqdm(self.iter_trial(), total=self.n_trials, desc=desc):
            self.make_trial_plot(trial_idx)

    def make_all_cluster_plots(self):
        desc = "Making all cluster plots"
        for cluster_idx in tqdm(self.iter_cluster(), total=self.n_clusters, desc=desc):
            self.make_cluster_plot(cluster_idx)

    def make_all_session_plots(self):
        logger.info(f"Making all session plots for session {self.pid}")
        self.make_session_plot()
        self.make_all_trial_plots()
        self.make_all_cluster_plots()


def make_session_plots(pid):
    Generator(pid).make_all_session_plots()


def make_all_plots():
    Parallel(n_jobs=-2)(delayed(make_session_plots)(pid) for pid in iter_session())


if __name__ == '__main__':
    if len(sys.argv) == 1:
        make_all_plots()
    elif len(sys.argv) == 2:
        pid = sys.argv[1]
        if not is_valid_uuid(pid):
            raise ValueError(f"{pid} not a valid insertion UUID")
        gen = Generator(pid)
        gen.make_all_session_plots()
