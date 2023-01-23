# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from datetime import datetime, date
# from pathlib import Path
# from pprint import pprint
from uuid import UUID
# import argparse
# import functools
import io
import json
import locale
import logging
import logging
import os.path as op
import png
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
# mpl.style.use('seaborn')
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
# ROOT_DIR and DATA_DIR are loaded from static_plots.py
# ROOT_DIR = Path(__file__).parent.resolve()
# DATA_DIR = ROOT_DIR / 'static/data'
CACHE_DIR = ROOT_DIR / 'static/cache'
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


def save_json(path, dct, **kwargs):
    with open(path, 'w') as f:
        json.dump(dct, f, sort_keys=True, cls=DateTimeEncoder, **kwargs)


def load_json(path):
    if not path.exists():
        logger.error(f"file {path} doesn't exist")
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def get_cluster_idx_from_xy(pid, cluster_idx, x, y, qc):
    if qc == 1:
        df = pd.read_parquet(cluster_qc_pixels_path(pid))
    else:
        df = pd.read_parquet(cluster_pixels_path(pid))
    norm_dist = (df.x.values - x) ** 2 + (df.y.values - y) ** 2
    min_idx = np.nanargmin(norm_dist)
    if norm_dist[min_idx] < 0.005:  # TODO some limit of distance?
        return df.iloc[min_idx].cluster_id, min_idx
    else:
        idx = np.where(df.cluster_id.values == cluster_idx)[0]
        return cluster_idx, idx


# -------------------------------------------------------------------------------------------------
# Path functions
# -------------------------------------------------------------------------------------------------

def session_data_path(pid):
    return DATA_DIR / pid


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


def cluster_qc_overview_path(pid, cluster_idx):
    return session_cache_path(pid) / f'cluster-{cluster_idx:04d}_qc.png'


def cluster_pixels_path(pid):
    return session_cache_path(pid) / 'cluster_pixels.pqt'


def cluster_qc_pixels_path(pid):
    return session_cache_path(pid) / 'cluster_pixels_qc.pqt'


def trial_intervals_path(pid):
    return session_cache_path(pid) / f'trial_intervals.pqt'


def caption_path(figure):
    return CACHE_DIR.joinpath(f'{figure}_px_locations.pqt')


# -------------------------------------------------------------------------------------------------
# Session iterator
# -------------------------------------------------------------------------------------------------

def get_pids():
    df = pd.read_parquet(DATA_DIR.joinpath('session.table.pqt'))
    pids = df.pid.values
    # pids = sorted([str(p.name) for p in DATA_DIR.iterdir()])
    pids = [pid for pid in pids if is_valid_uuid(pid)]
    assert pids
    return pids


def iter_session():
    yield from get_pids()


# def generate_figure_json():
#     # Read the Parquet table and convert to JSON to make it loadable by flaskapp.
#     # TODO: integrate somewhere (in the Generator) the code used to generate this file
#     df = pd.read_parquet(DATA_DIR / 'cluster_fig_subplots.pqt')
#     details = {
#         'cluster': {panel: (xmin, 1 - ymax, xmax, 1 - ymin)
#                     for _, (panel, xmin, ymax, xmax, ymin) in df.iterrows()}}
#     save_json(figure_details_path(), details, indent=2)


def get_subplot_position(ax1, ax2):
    xmin_ymax = ax1.get_position().corners()[1]
    xmax_ymin = ax2.get_position().corners()[2]

    return np.r_[xmin_ymax, xmax_ymin]


# -------------------------------------------------------------------------------------------------
# Plot and JSON generator
# -------------------------------------------------------------------------------------------------

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
        logger.debug(f"Saving session details for session {pid}")
        save_json(path, self.session_details)

        self.trial_idxs = self.session_details['_trial_ids']
        self.n_trials = len(self.trial_idxs)
        self.cluster_idxs = self.session_details['_cluster_ids']
        self.n_clusters = len(self.cluster_idxs)
        self.good_cluster_idxs = np.array(self.session_details['_cluster_ids'])[np.array(self.session_details['_good_ids'])]
        self.n_good_clusters = len(self.good_cluster_idxs)

    # Iterators
    # -------------------------------------------------------------------------------------------------

    def first_trial(self):
        return sorted(self.trial_idxs)[0]

    def iter_trial(self):
        yield from sorted(self.trial_idxs)

    def first_cluster(self):
        return sorted(self.cluster_idxs)[0]

    def first_good_cluster(self):
        return sorted(self.good_cluster_idxs)[0]

    def iter_cluster(self):
        yield from sorted(self.cluster_idxs)

    def iter_good_cluster(self):
        yield from sorted(self.good_cluster_idxs)

    # Saving JSON details
    # -------------------------------------------------------------------------------------------------

    def save_trial_details(self, trial_idx):
        logger.debug(f"saving trial #{trial_idx:04} details for session {self.pid}")
        details = self.dl.get_trial_details(trial_idx)
        path = trial_details_path(self.pid, trial_idx)
        save_json(path, details)

    def save_cluster_details(self, cluster_idx):
        logger.debug(f"saving cluster #{cluster_idx:04} details for session {self.pid}")
        details = self.dl.get_cluster_details(cluster_idx)
        path = cluster_details_path(self.pid, cluster_idx)
        save_json(path, details)

    # -------------------------------------------------------------------------------------------------
    # SESSION OVERVIEW
    # -------------------------------------------------------------------------------------------------

    # FIGURE 1

    def make_session_plot(self, force=False, captions=False):

        path = session_overview_path(self.pid)
        if not force and path.exists():
            return
        logger.debug(f"making session overview plot for session {self.pid}")
        loader = self.dl

        try:
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[9, 4, 4], wspace=0.2)

            # First row
            gs0 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs[0], width_ratios=[8, 1, 1, 1, 1, 1],
                                                   height_ratios=[1, 10], hspace=0.3, wspace=0.1)
            gs0_ax1 = fig.add_subplot(gs0[0, 0])
            gs0_ax2 = fig.add_subplot(gs0[1, 0])
            gs0_ax3 = fig.add_subplot(gs0[0, 1])
            gs0_ax4 = fig.add_subplot(gs0[1, 1])
            gs0_ax5 = fig.add_subplot(gs0[0, 2])
            gs0_ax6 = fig.add_subplot(gs0[1, 2])
            gs0_ax7 = fig.add_subplot(gs0[0, 3])
            gs0_ax8 = fig.add_subplot(gs0[1, 3])
            gs0_ax9 = fig.add_subplot(gs0[0, 4])
            gs0_ax10 = fig.add_subplot(gs0[1, 4])
            gs0_ax11 = fig.add_subplot(gs0[0, 5])
            gs0_ax12 = fig.add_subplot(gs0[1, 5])

            loader.plot_session_raster(ax=gs0_ax2)
            loader.plot_good_bad_clusters(ax=gs0_ax4, ax_legend=gs0_ax3, xlabel='Amp (uV)')
            loader.plot_spikes_amp_vs_depth_vs_firing_rate(ax=gs0_ax6, ax_cbar=gs0_ax5, xlabel='Amp (uV)')
            loader.plot_ap_rms(ax=gs0_ax8, ax_cbar=gs0_ax7)
            loader.plot_lfp_spectrum(ax=gs0_ax10, ax_cbar=gs0_ax9)
            loader.plot_brain_regions(ax=gs0_ax12)

            gs0_ax4.get_yaxis().set_visible(False)
            gs0_ax6.get_yaxis().set_visible(False)
            gs0_ax8.get_yaxis().set_visible(False)
            gs0_ax10.get_yaxis().set_visible(False)
            remove_frame(gs0_ax1)
            remove_frame(gs0_ax11)

            # Second row
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[1], width_ratios=[8, 1, 1, 1, 1, 1],
                                                   wspace=0.1)
            gs11 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs1[0, 0], height_ratios=[2, 3, 1, 3])
            ax_a = fig.add_subplot(gs11[0, 0])
            ax_b = fig.add_subplot(gs11[1, 0])
            ax_c = fig.add_subplot(gs11[2, 0])
            ax_d = fig.add_subplot(gs11[3, 0])

            gs13 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs1[0, 2])
            ax_leg = fig.add_subplot(gs13[0])
            loader.plot_session_reaction_time(ax=ax_a)
            loader.plot_session_contrasts(axs=[ax_b, ax_c, ax_d], ax_legend=ax_leg)

            ax_a.sharex(gs0_ax2)
            ax_b.sharex(gs0_ax2)
            ax_c.sharex(gs0_ax2)
            ax_d.sharex(gs0_ax2)

            plt.setp(ax_a.get_xticklabels(), visible=False)
            plt.setp(ax_b.get_xticklabels(), visible=False)

            gs14 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs1[0, 3:])
            ax_cor = fig.add_subplot(gs14[0])
            loader.plot_coronal_slice(ax_cor)

            # Third row
            gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], width_ratios=[4, 4, 4, 1], wspace=0.1)
            ax13 = fig.add_subplot(gs2[0, 0])
            ax14 = fig.add_subplot(gs2[0, 1])
            ax15 = fig.add_subplot(gs2[0, 2])
            ax16 = fig.add_subplot(gs2[0, 3])

            loader.plot_raw_data(axs=[ax13, ax14, ax15], raster=False)
            loader.plot_brain_regions(ax16)
            ax16.set_ylim(20, 3840)

            set_figure_style_all(fig)

            if captions:
                subplots = []
                fig_pos = get_subplot_position(gs0_ax1, gs0_ax2)
                subplots.append({'panel': 'A', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(gs0_ax3, gs0_ax4)
                subplots.append({'panel': 'B', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(gs0_ax5, gs0_ax6)
                subplots.append({'panel': 'C', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(gs0_ax7, gs0_ax8)
                subplots.append({'panel': 'D', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(gs0_ax9, gs0_ax10)
                subplots.append({'panel': 'E', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(gs0_ax11, gs0_ax12)
                subplots.append({'panel': 'F', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(ax_a, ax_d)
                subplots.append({'panel': 'G', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(ax_cor, ax_cor)
                subplots.append({'panel': 'H', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(ax13, ax15)
                subplots.append({'panel': 'I', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                fig_pos = get_subplot_position(ax16, ax16)
                subplots.append({'panel': 'J', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})

                df = pd.DataFrame.from_dict(subplots)
                df.to_parquet(caption_path('figure1'))
            else:

                fig.savefig(path)

            plt.close(fig)
        except Exception as e:
            print(f"error with session overview plot {self.pid}: {str(e)}")

    # FIGURE 2

    def make_behavior_plot(self, force=False, captions=False):

        path = behaviour_overview_path(self.pid)
        if not force and path.exists():
            return
        logger.debug(f"making behavior plot for session {self.pid}")
        loader = self.dl

        fig = plt.figure(figsize=(15, 10))

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 3], hspace=0.3)

        gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], width_ratios=[4, 1, 4, 4], wspace=0.4)
        ax1 = fig.add_subplot(gs1[0, 0])
        ax2 = fig.add_subplot(gs1[0, 1])
        ax3 = fig.add_subplot(gs1[0, 2])
        ax4 = fig.add_subplot(gs1[0, 3])
        loader.plot_psychometric_curve(ax=ax1, ax_legend=ax2)
        loader.plot_chronometric_curve(ax=ax3)
        loader.plot_reaction_time(ax=ax4)

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs[1], height_ratios=[1, 3], hspace=0, wspace=0.5)
        ax5 = fig.add_subplot(gs1[0, 0])
        ax6 = fig.add_subplot(gs1[1, 0])
        ax7 = fig.add_subplot(gs1[0, 1])
        ax8 = fig.add_subplot(gs1[1, 1])
        ax9 = fig.add_subplot(gs1[0, 2])
        ax10 = fig.add_subplot(gs1[1, 2])
        ax11 = fig.add_subplot(gs1[0, 3])
        ax12 = fig.add_subplot(gs1[1, 3])
        ax13 = fig.add_subplot(gs1[0, 4])
        ax14 = fig.add_subplot(gs1[1, 4])
        ax15 = fig.add_subplot(gs1[0, 5])
        ax16 = fig.add_subplot(gs1[1, 5])

        loader.plot_dlc_feature_raster('left', 'paw_r_speed', axs=[ax5, ax6], ylabel0='Speed (px/s)', title='Left paw')
        loader.plot_dlc_feature_raster('left', 'nose_tip_speed', axs=[ax7, ax8], ylabel0='Speed (px/s)', ylabel1=None,
                                       title='Nose tip')
        loader.plot_dlc_feature_raster('left', 'motion_energy', axs=[ax9, ax10], zscore_flag=True, ylabel0='ME (z-score)',
                                       ylabel1=None, title='Motion energy')
        loader.plot_dlc_feature_raster('left', 'pupilDiameter_smooth', axs=[ax11, ax12], zscore_flag=True, norm=True,
                                       ylabel0='Pupil (z-score)', ylabel1=None, title='Smoothed pupil diameter')
        loader.plot_wheel_raster(axs=[ax13, ax14], ylabel0='Velocity (rad/s)', ylabel1=None, title='Wheel velocity')
        loader.plot_lick_raster(axs=[ax15, ax16], ylabel1=None, title='Licks')

        ax5.get_xaxis().set_visible(False)
        ax7.get_xaxis().set_visible(False)
        ax9.get_xaxis().set_visible(False)
        ax11.get_xaxis().set_visible(False)
        ax13.get_xaxis().set_visible(False)
        ax15.get_xaxis().set_visible(False)

        ax8.get_yaxis().set_visible(False)
        ax10.get_yaxis().set_visible(False)
        ax12.get_yaxis().set_visible(False)
        ax14.get_yaxis().set_visible(False)
        ax16.get_yaxis().set_visible(False)

        ax5.sharex(ax6)
        ax7.sharex(ax8)
        ax9.sharex(ax10)
        ax11.sharex(ax12)
        ax13.sharex(ax14)
        ax15.sharex(ax16)

        set_figure_style_all(fig, top=0.95)

        if captions:
            subplots = []
            fig_pos = get_subplot_position(ax1, ax1)
            subplots.append({'panel': 'A', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax3, ax3)
            subplots.append({'panel': 'B', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax4, ax4)
            subplots.append({'panel': 'C', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax5, ax6)
            subplots.append({'panel': 'D', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax7, ax8)
            subplots.append({'panel': 'E', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax9, ax10)
            subplots.append({'panel': 'F', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax11, ax12)
            subplots.append({'panel': 'G', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax13, ax14)
            subplots.append({'panel': 'H', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax15, ax16)
            subplots.append({'panel': 'I', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})

            df = pd.DataFrame.from_dict(subplots)
            df.to_parquet(caption_path('figure2'))
        else:
            fig.savefig(path)

        plt.close(fig)

    # -------------------------------------------------------------------------------------------------
    # SINGLE TRIAL OVERVIEW
    # -------------------------------------------------------------------------------------------------

    # FIGURE 3

    def make_trial_plot(self, trial_idx, force=False, captions=False):
        path = trial_overview_path(self.pid, trial_idx)
        if not force and path.exists():
            return
        logger.debug(f"making trial overview plot for session {self.pid}, trial #{trial_idx:04d}")
        loader = self.dl

        fig = plt.figure(figsize=(12, 5))

        gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[7.5, 15, 1], height_ratios=[8, 3], hspace=0.05, wspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        loader.plot_session_raster(trial_idx=trial_idx, ax=ax1, xlabel='T in session (s)')
        loader.plot_trial_raster(trial_idx=trial_idx, ax=ax2, xlabel=None)
        ax2.get_yaxis().set_visible(False)
        loader.plot_brain_regions(ax3)

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[4], hspace=0.1)
        ax4 = fig.add_subplot(gs1[0, 0])
        ax5 = fig.add_subplot(gs1[1, 0])

        loader.plot_wheel_trace(trial_idx=trial_idx, ax=ax4, ylabel='Wheel pos (rad)', xlabel=None)
        loader.plot_dlc_feature_trace('left', 'paw_r_speed', trial_idx, ax=ax5, xlabel='T in trial (s)', ylabel='Paw speed (px/s)')

        ax4.sharex(ax2)
        ax5.sharex(ax2)

        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)

        set_figure_style(fig)

        if captions:
            subplots = []
            fig_pos = get_subplot_position(ax1, ax1)
            subplots.append({'panel': 'A', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax2, ax2)
            subplots.append({'panel': 'B', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax3, ax3)
            subplots.append({'panel': 'C', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax4, ax4)
            subplots.append({'panel': 'D', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax5, ax5)
            subplots.append({'panel': 'E', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})

            df = pd.DataFrame.from_dict(subplots)
            df.to_parquet(caption_path('figure3'))
        else:
            fig.savefig(path)

        plt.close(fig)

    # FIGURE 4

    def make_trial_event_plot(self, force=False, captions=False):
        path = trial_event_overview_path(self.pid)
        if not force and path.exists():
            return
        logger.debug(f"making trial event plot for session {self.pid}")
        loader = self.dl

        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 15], width_ratios=[5, 5, 5, 1], wspace=0.1)

        ax1 = fig.add_subplot(gs[0, 0:3])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[1, 3])

        loader.plot_event_aligned_activity(axs=[ax2, ax3, ax4], ax_cbar=ax1)
        loader.plot_brain_regions(ax=ax5)
        set_figure_style(fig)

        if captions:

            subplots = []
            fig_pos = get_subplot_position(ax1, ax1)
            subplots.append({'panel': 'A', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax2, ax2)
            subplots.append({'panel': 'B', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax3, ax3)
            subplots.append({'panel': 'C', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax4, ax4)
            subplots.append({'panel': 'D', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})

            df = pd.DataFrame.from_dict(subplots)
            df.to_parquet(caption_path('figure4'))
        else:
            fig.savefig(path)

            path_interval = trial_intervals_path(self.pid)
            df = pd.DataFrame()
            df['t0'] = loader.trial_intervals[:, 0]
            df['t1'] = loader.trial_intervals[:, 1]
            df.to_parquet(path_interval)


        plt.close(fig)

    # -------------------------------------------------------------------------------------------------
    # SINGLE CLUSTER OVERVIEW
    # -------------------------------------------------------------------------------------------------

    # FIGURE 5

    def make_cluster_plot(self, cluster_idx, force=False, captions=False):
        path = cluster_overview_path(self.pid, cluster_idx)
        if not force and path.exists():
            return
        logger.debug(f"making cluster overview plot for session {self.pid}, cluster #{cluster_idx:04d}")
        loader = self.dl

        fig = plt.figure(figsize=(15, 10))

        gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2, 10, 3], height_ratios=[6, 2], wspace=0.2)

        gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
        ax1 = fig.add_subplot(gs0[0, 0])

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1], height_ratios=[1, 3], hspace=0, wspace=0.2)
        ax2 = fig.add_subplot(gs1[0, 0])
        ax3 = fig.add_subplot(gs1[1, 0])
        ax4 = fig.add_subplot(gs1[0, 1])
        ax5 = fig.add_subplot(gs1[1, 1])
        ax6 = fig.add_subplot(gs1[0, 2])
        ax7 = fig.add_subplot(gs1[1, 2])
        ax8 = fig.add_subplot(gs1[0, 3])
        ax9 = fig.add_subplot(gs1[1, 3])

        gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], width_ratios=[8, 1])
        ax10 = fig.add_subplot(gs2[0, 0])
        ax11 = fig.add_subplot(gs2[0, 1])

        gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3:])
        ax12 = fig.add_subplot(gs3[0, 0])
        ax13 = fig.add_subplot(gs3[0, 1])
        ax14 = fig.add_subplot(gs3[0, 2])

        set_figure_style(fig)

        loader.plot_spikes_amp_vs_depth(cluster_idx, ax=ax1, xlabel='Amp (uV)')

        loader.plot_block_single_cluster_raster(cluster_idx, axs=[ax2, ax3])
        loader.plot_contrast_single_cluster_raster(cluster_idx, axs=[ax4, ax5], ylabel0=None, ylabel1=None)
        loader.plot_left_right_single_cluster_raster(cluster_idx, axs=[ax6, ax7], ylabel0=None, ylabel1=None)
        loader.plot_correct_incorrect_single_cluster_raster(cluster_idx, axs=[ax8, ax9], ylabel0=None, ylabel1=None)
        ax2.get_xaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)
        ax6.get_xaxis().set_visible(False)
        ax8.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)
        ax7.get_yaxis().set_visible(False)
        ax9.get_yaxis().set_visible(False)

        loader.plot_cluster_waveforms(cluster_idx, ax=ax10)
        loader.plot_channel_probe_location(cluster_idx, ax=ax11)

        loader.plot_autocorrelogram(cluster_idx, ax=ax12)
        loader.plot_inter_spike_interval(cluster_idx, ax=ax13)
        loader.plot_cluster_amplitude(cluster_idx, ax=ax14)

        ax2.sharex(ax3)
        ax4.sharex(ax5)
        ax6.sharex(ax7)
        ax8.sharex(ax9)

        yax_to_lim = [ax2, ax4, ax6, ax8]
        max_ax = np.max([ax.get_ylim()[1] for ax in yax_to_lim])
        min_ax = np.min([ax.get_ylim()[0] for ax in yax_to_lim])
        for ax in yax_to_lim:
            ax.set_ylim(min_ax, max_ax)

        fig.savefig(path)

        if captions:

            subplots = []
            fig_pos = get_subplot_position(ax1, ax1)
            subplots.append({'panel': 'A', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax2, ax3)
            subplots.append({'panel': 'B', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax4, ax5)
            subplots.append({'panel': 'C', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax6, ax7)
            subplots.append({'panel': 'D', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax8, ax9)
            subplots.append({'panel': 'E', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax10, ax11)
            subplots.append({'panel': 'F', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax12, ax12)
            subplots.append({'panel': 'G', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax13, ax13)
            subplots.append({'panel': 'H', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax14, ax14)
            subplots.append({'panel': 'I', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            df = pd.DataFrame.from_dict(subplots)

            df.to_parquet(caption_path('figure5'))
        else:
            fig.savefig(path)

            path_scat = cluster_pixels_path(self.pid)
            if not path_scat.exists():
                idx = np.argsort(loader.clusters_good.depths)[::-1]
                pixels = ax1.transData.transform(np.vstack([loader.clusters_good.amps[idx].astype(np.float64) * 1e6,
                                                            loader.clusters_good.depths[idx].astype(np.float64)]).T)
                width, height = fig.canvas.get_width_height()
                pixels[:, 0] /= width
                pixels[:, 1] /= height
                df = pd.DataFrame()
                # Sort by depth so they are in the same order as the cluster selector drop down
                df['cluster_id'] = loader.clusters_good.cluster_id[idx].astype(np.int32)
                df['x'] = pixels[:, 0]
                df['y'] = pixels[:, 1]
                df.to_parquet(path_scat)

        plt.close(fig)

    def make_cluster_qc_plot(self, cluster_idx, force=False, captions=False):

        path = cluster_qc_overview_path(self.pid, cluster_idx)
        if not force and path.exists():
            return
        logger.debug(f"making cluster qc overview plot for session {self.pid}, cluster #{cluster_idx:04d}")

        loader = self.dl
        fig = plt.figure(figsize=(15.24, 10))

        gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[2, 10, 3, 1], height_ratios=[6, 2], wspace=0.2, hspace=0.3)

        gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 0])
        ax1 = fig.add_subplot(gs0[0, 0])

        loader.plot_spikes_amp_vs_depth(cluster_idx, ax=ax1, xlabel='Amp (uV)', type='all')

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0, 1], height_ratios=[1, 3], hspace=0, wspace=0.2)
        ax2 = fig.add_subplot(gs1[0, 0])
        ax3 = fig.add_subplot(gs1[1, 0])
        ax4 = fig.add_subplot(gs1[0, 1])
        ax5 = fig.add_subplot(gs1[1, 1])
        ax6 = fig.add_subplot(gs1[0, 2])
        ax7 = fig.add_subplot(gs1[1, 2])
        ax8 = fig.add_subplot(gs1[0, 3])
        ax9 = fig.add_subplot(gs1[1, 3])

        loader.plot_block_single_cluster_raster(cluster_idx, axs=[ax2, ax3])
        loader.plot_contrast_single_cluster_raster(cluster_idx, axs=[ax4, ax5], ylabel0=None, ylabel1=None)
        loader.plot_left_right_single_cluster_raster(cluster_idx, axs=[ax6, ax7], ylabel0=None, ylabel1=None)
        loader.plot_correct_incorrect_single_cluster_raster(cluster_idx, axs=[ax8, ax9], ylabel0=None, ylabel1=None)
        ax2.get_xaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)
        ax6.get_xaxis().set_visible(False)
        ax8.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)
        ax7.get_yaxis().set_visible(False)
        ax9.get_yaxis().set_visible(False)


        gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 2])
        ax10 = fig.add_subplot(gs2[0])
        loader.plot_cluster_waveforms(cluster_idx, ax=ax10)

        gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 3])
        ax11 = fig.add_subplot(gs3[0])
        loader.plot_channel_probe_location(cluster_idx, ax=ax11)

        gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :])
        gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs4[0], height_ratios=[1, 10],
                                                width_ratios=[3, 1], wspace=0.05)

        gs00_ax0 = fig.add_subplot(gs00[0, 0])
        gs00_ax1 = fig.add_subplot(gs00[0, 1])
        gs00_ax2 = fig.add_subplot(gs00[1, 0], projection='scatter_density')
        gs00_ax3 = fig.add_subplot(gs00[1, 1])

        loader.plot_noise_cutoff(cluster_idx, axs=[gs00_ax2, gs00_ax3, gs00_ax0, gs00_ax1])

        gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs4[1], height_ratios=[1, 10])
        gs01_ax0 = fig.add_subplot(gs01[0, :])
        gs01_ax1 = fig.add_subplot(gs01[1, 0])
        gs01_ax2 = fig.add_subplot(gs01[1, 1])

        loader.plot_sliding_rp(cluster_idx, axs=[gs01_ax1, gs01_ax2, gs01_ax0])

        set_figure_style_all(fig)

        if captions:

            subplots = []
            fig_pos = get_subplot_position(ax1, ax1)
            subplots.append({'panel': 'A', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax2, ax3)
            subplots.append({'panel': 'B', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax4, ax5)
            subplots.append({'panel': 'C', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax6, ax7)
            subplots.append({'panel': 'D', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax8, ax9)
            subplots.append({'panel': 'E', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax10, ax11)
            subplots.append({'panel': 'F', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(gs00_ax0, gs00_ax3)
            subplots.append({'panel': 'G', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(gs01_ax0, gs01_ax2)
            subplots.append({'panel': 'H', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})

            df = pd.DataFrame.from_dict(subplots)
            df.to_parquet(caption_path('figure5_qc'))
        else:
            fig.savefig(path)

            path_scat = cluster_qc_pixels_path(self.pid)
            if not path_scat.exists() or force:
                idx = np.argsort(loader.clusters.depths)[::-1]
                pixels = ax1.transData.transform(np.vstack([loader.clusters.amps[idx].astype(np.float64) * 1e6,
                                                            loader.clusters.depths[idx].astype(np.float64)]).T)
                width, height = fig.canvas.get_width_height()
                pixels[:, 0] /= width
                pixels[:, 1] /= height
                df = pd.DataFrame()
                # Sort by depth so they are in the same order as the cluster selector drop down
                df['cluster_id'] = loader.clusters.cluster_id[idx].astype(np.int32)
                df['x'] = pixels[:, 0]
                df['y'] = pixels[:, 1]
                df.to_parquet(path_scat)

        plt.close(fig)

    # Plot generator functions
    # -------------------------------------------------------------------------------------------------

    def make_all_trial_plots(self, force=False):

        path = trial_overview_path(self.pid, self.first_trial())
        if not force and path.exists():
            logger.debug("Skipping trial plot generation as they seem to already exist")
            return

        desc = "Making all trial plots  "
        for trial_idx in tqdm(self.iter_trial(), total=self.n_trials, desc=desc):
            self.save_trial_details(trial_idx)
            try:
                self.make_trial_plot(trial_idx, force=force)
            except Exception as e:
                print(f"error with session {self.pid} trial #{trial_idx}: {str(e)}")

    def make_all_cluster_plots(self, force=False):

        path = cluster_overview_path(self.pid, self.first_good_cluster())
        if not force and path.exists():
            logger.debug("Skipping cluster plot generation as they seem to already exist")
            return

        desc = "Making all cluster plots"
        for cluster_idx in tqdm(self.iter_good_cluster(), total=self.n_good_clusters, desc=desc):
            self.save_cluster_details(cluster_idx)
            try:
                self.make_cluster_plot(cluster_idx, force=force)
            except Exception as e:
                print(f"error with session {self.pid} cluster #{cluster_idx}: {str(e)}")

    def make_all_cluster_qc_plots(self, force=False):

        path = cluster_qc_overview_path(self.pid, self.first_cluster())
        if not force and path.exists():
            logger.debug("Skipping cluster plot generation as they seem to already exist")
            return

        desc = "Making all cluster qc plots"
        for cluster_idx in tqdm(self.iter_cluster(), total=self.n_clusters, desc=desc):
            self.save_cluster_details(cluster_idx)
            try:
                self.make_cluster_qc_plot(cluster_idx, force=force)
            except Exception as e:
                print(f"error with session {self.pid} cluster #{cluster_idx}: {str(e)}")

    def make_all_plots(self, nums=()):
        if 0 in nums:  # used to regenerate the session.json only
            return
        # nums is a list of numbers 1-5 (figure numbers)

        logger.info(f"Making all session plots for session {self.pid} {nums}")

        # Figure 1
        self.make_session_plot(force=1 in nums)

        # Figure 2
        try:
            self.make_behavior_plot(force=2 in nums)
        except Exception as e:
            print(f"error with session {self.pid} behavior plot: {str(e)}")

        # Figure 3 (one plot per trial)
        self.make_all_trial_plots(force=3 in nums)

        # Figure 4
        self.make_trial_event_plot(force=4 in nums)

        # Figure 5 (one plot per cluster)
        self.make_all_cluster_plots(force=5 in nums)

        # Figure 5 QC (one plot per cluster)
        self.make_all_cluster_qc_plots(force=6 in nums)

    def make_captions(self):
        self.make_session_plot(force=True, captions=True)
        self.make_behavior_plot(force=True, captions=True)
        self.make_trial_plot(self.trial_idxs[0], force=True, captions=True)
        self.make_trial_event_plot(force=True, captions=True)
        self.make_cluster_plot(self.good_cluster_idxs[0], force=True, captions=True)
        self.make_cluster_qc_plot(self.good_cluster_idxs[0], force=True, captions=True)

        caption_json = {}
        for fig in CAPTIONS.keys():
            df = pd.read_parquet(caption_path(fig))
            fig_panels = {}
            for _, row in df.iterrows():
                fig_panels[row['panel']] = {'coords': (row['xmin'], 1 - row['ymax'], row['xmax'], 1 - row['ymin']),
                                            'legend': CAPTIONS[fig][row['panel']]}

            caption_json[fig] = fig_panels

        save_json(figure_details_path(), caption_json, indent=2)


def make_captions():
    pid = get_pids()[0]
    Generator(pid).make_captions()


def make_all_plots(pid, nums=()):
    logger.info(f"Generating all plots for session {pid}")
    Generator(pid).make_all_plots(nums=nums)


if __name__ == '__main__':

    # Regenerate all figures.
    if len(sys.argv) == 1:
        Parallel(n_jobs=-3)(delayed(make_all_plots)(pid) for pid in iter_session())

    # Regenerate some figures for all sessions.
    elif len(sys.argv) >= 2 and not is_valid_uuid(sys.argv[1]):
        which = sys.argv[1]

        # which figure numbers to regenerate
        nums = list(map(int, which.split(',')))
        logger.info(f"Regenerating figures {', '.join('#%d' % _ for _ in nums)}")

        Parallel(n_jobs=-3)(delayed(make_all_plots)(pid, nums=nums) for pid in iter_session())

    # Regenerate figures for 1 session.
    elif len(sys.argv) >= 2 and is_valid_uuid(sys.argv[1]):
        nums = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) >= 3 else ()
        make_all_plots(sys.argv[1], nums=nums)
