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
import logging
from operator import itemgetter
import os.path as op
import png
import re
import sys
import gc

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plots.static_plots_photometry import *
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

CACHE_DIR = ROOT_DIR / 'static/cache'
PORT = 4321
DEFAULT_eid = 'decc8d40-cf74-4263-ae9d-a0cc68b47e86'
DEFAULT_DSET = 'bwm'  # 'bwm' (brain wide map) or 'rs' (repeated sites)


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


# -------------------------------------------------------------------------------------------------
# Path functions
# -------------------------------------------------------------------------------------------------

# def session_data_path(eid):
#     return DATA_DIR / eid


def session_cache_path(eid, cache_path=None):
    cache_path = cache_path or CACHE_DIR
    cp = cache_path / eid
    cp.mkdir(exist_ok=True, parents=True)
    assert cp.exists(), f"the path `{cp}` does not exist"
    return cp


def figure_details_path(cache_path=None):
    cache_path = cache_path or CACHE_DIR
    return cache_path / 'figures.json'


def session_details_path(eid, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / 'session.json'


def trial_details_path(eid, trial_idx, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / f'trial-{trial_idx:04d}.json'


def session_overview_path(eid, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / 'session_overview.png'


def behaviour_overview_path(eid, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / 'behaviour_overview.png'


def trial_event_overview_path(eid, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / 'trial_overview.png'


def trial_overview_path(eid, trial_idx, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / f'trial-{trial_idx:04d}.png'


def trial_intervals_path(eid, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / f'trial_intervals.pqt'


def caption_path(figure, cache_path=None):
    cache_path = cache_path or CACHE_DIR
    return cache_path.joinpath(f'{figure}_px_locations.pqt')


# -------------------------------------------------------------------------------------------------
# Session iterator
# -------------------------------------------------------------------------------------------------

def get_eids(data_path=None):
    data_path = data_path or DATA_DIR
    df = pd.read_parquet(data_path.joinpath('session.table.pqt'))
    eids = df.eid.values
    eids = [eid for eid in eids if is_valid_uuid(eid)]
    assert eids
    return eids


def iter_session(data_path=None):
    yield from get_eids(data_path=data_path)


def get_subplot_position(ax1, ax2):
    xmin_ymax = ax1.get_position().corners()[1]
    xmax_ymin = ax2.get_position().corners()[2]

    return np.r_[xmin_ymax, xmax_ymin]


# -------------------------------------------------------------------------------------------------
# Plot and JSON generator
# -------------------------------------------------------------------------------------------------

class Generator:
    def __init__(self, eid, cache_path=None, data_path=None):
        self.cache_path = cache_path or CACHE_DIR
        self.dl = DataLoader(data_path=data_path)
        self.dl.session_init(eid)
        self.eid = eid

        # Ensure the session cache folder exists.
        session_cache_path(eid, cache_path=self.cache_path).mkdir(exist_ok=True, parents=True)

        # Load the session details.
        self.session_details = self.dl.get_session_details()

        # Save the session details to a JSON file.
        path = session_details_path(eid, cache_path=self.cache_path)
        logger.debug(f"Saving session details for session {eid}")
        save_json(path, self.session_details)

        self.trial_idxs = self.session_details['_trial_ids']
        self.n_trials = len(self.trial_idxs)

        # Get the number of rois - need to figure this out
        self.n_rois = 1


    # Iterators
    # -------------------------------------------------------------------------------------------------

    def first_trial(self):
        return sorted(self.trial_idxs)[0]

    def iter_trial(self):
        yield from sorted(self.trial_idxs)

    # Saving JSON details
    # -------------------------------------------------------------------------------------------------

    def save_trial_details(self, trial_idx):
        logger.debug(f"saving trial #{trial_idx:04} details for session {self.eid}")
        details = self.dl.get_trial_details(trial_idx)
        path = trial_details_path(self.eid, trial_idx, cache_path=self.cache_path)
        save_json(path, details)

    # -------------------------------------------------------------------------------------------------
    # SESSION OVERVIEW
    # -------------------------------------------------------------------------------------------------

    # FIGURE 1

    def make_session_plot(self, force=False, captions=False):

        path = session_overview_path(self.eid, cache_path=self.cache_path)
        if not force and path.exists():
            return
        logger.debug(f"making session overview plot for session {self.eid}")
        loader = self.dl

        try:
            fig = plt.figure(figsize=(15, 7))
            gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[9, 4], wspace=0.3, hspace=0.3)

            # First row
            gs0 = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[0], width_ratios=[6, 2, 1, 1],
                                                   wspace=0.1, hspace=0.3)
            gs0_ax1 = fig.add_subplot(gs0[0, 0])
            gs0_ax2 = fig.add_subplot(gs0[1, 0])
            gs0_ax3 = fig.add_subplot(gs0[2, 0])
            gs0_ax4 = fig.add_subplot(gs0[0, 1])
            gs0_ax5 = fig.add_subplot(gs0[1, 1])
            gs0_ax6 = fig.add_subplot(gs0[2, 1])
            gs0_ax7 = fig.add_subplot(gs0[:2, 3])

            # Full session photometry signal
            loader.plot_raw_photometry_signal(ax=gs0_ax1, xlabel=None, ylabel='Raw isobestic')
            loader.plot_photometry_signal('calcium', ax=gs0_ax2, xlabel=None, ylabel='Calcium')
            loader.plot_photometry_signal('calcium_moving_avg', ax=gs0_ax3, xlabel='Time',
                                          ylabel='Moving average calcium')

            # Zoomed in photometry signal
            loader.plot_raw_photometry_signal(ax=gs0_ax4, xlim=[1000, 1010], xlabel=None, ylabel2='Raw calcium')
            loader.plot_photometry_signal('calcium', xlim=[1000, 1010], ax=gs0_ax5)
            loader.plot_photometry_signal('calcium_moving_avg', xlim=[1000, 1010], ax=gs0_ax6)

            # Zoomed in slice (placeholder for now)
            loader.plot_brain_slice(ax=gs0_ax7)

            gs0_ax1.get_xaxis().set_visible(False)
            gs0_ax2.get_xaxis().set_visible(False)
            gs0_ax4.get_xaxis().set_visible(False)
            gs0_ax5.get_xaxis().set_visible(False)
            remove_frame(gs0_ax7)

            # Second row
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], width_ratios=[6, 2, 1, 1],
                                                   wspace=0.1)
            gs11 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs1[0, 0], height_ratios=[2, 3, 1, 3])
            ax_a = fig.add_subplot(gs11[0, 0])
            ax_b = fig.add_subplot(gs11[1, 0])
            ax_c = fig.add_subplot(gs11[2, 0])
            ax_d = fig.add_subplot(gs11[3, 0])

            gs13 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs1[0, 1])
            ax_leg = fig.add_subplot(gs13[0])
            loader.plot_session_reaction_time(ax=ax_a)
            loader.plot_session_contrasts(axs=[ax_b, ax_c, ax_d], ax_legend=ax_leg)

            ax_a.sharex(gs0_ax2)
            ax_b.sharex(gs0_ax2)
            ax_c.sharex(gs0_ax2)
            ax_d.sharex(gs0_ax2)

            plt.setp(ax_a.get_xticklabels(), visible=False)
            plt.setp(ax_b.get_xticklabels(), visible=False)

            gs14 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs1[0, 2:])
            ax_cor = fig.add_subplot(gs14[0])
            loader.plot_coronal_slice(ax_cor)

            set_figure_style_all(fig, margin_inches=0.8)

            if captions:
                subplots = []
                # fig_pos = get_subplot_position(gs0_ax1, gs0_ax2)
                # subplots.append({'panel': 'A', 'xmin': fig_pos[0], 'ymax': fig_pos[1] - 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(gs0_ax3, gs0_ax4)
                # subplots.append({'panel': 'B', 'xmin': fig_pos[0], 'ymax': fig_pos[1] - 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(gs0_ax5, gs0_ax6)
                # subplots.append({'panel': 'C', 'xmin': fig_pos[0], 'ymax': fig_pos[1] - 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(gs0_ax7, gs0_ax8)
                # subplots.append({'panel': 'D', 'xmin': fig_pos[0], 'ymax': fig_pos[1] - 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(gs0_ax9, gs0_ax10)
                # subplots.append({'panel': 'E', 'xmin': fig_pos[0], 'ymax': fig_pos[1] - 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(gs0_ax11, gs0_ax12)
                # subplots.append({'panel': 'F', 'xmin': fig_pos[0], 'ymax': fig_pos[1] - 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(gs0_ax13, gs0_ax14)
                # subplots.append({'panel': 'G', 'xmin': fig_pos[0], 'ymax': fig_pos[1] - 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(ax_a, ax_d)
                # subplots.append({'panel': 'H', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(ax_cor, ax_cor)
                # subplots.append({'panel': 'I', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(ax13, ax15)
                # subplots.append({'panel': 'J', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
                # fig_pos = get_subplot_position(ax16, ax16)
                # subplots.append({'panel': 'K', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})

                df = pd.DataFrame.from_dict(subplots)
                df.to_parquet(caption_path('figure1', cache_path=self.cache_path))
            else:

                fig.savefig(path)

            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.error(f"error with session overview plot {self.eid}: {str(e)}")

    # FIGURE 2

    def make_behavior_plot(self, force=False, captions=False):

        path = behaviour_overview_path(self.eid, cache_path=self.cache_path)
        if not force and path.exists():
            return
        logger.debug(f"making behavior plot for session {self.eid}")
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
                                       ylabel0='Pupil (z-score)', ylabel1=None, title='Pupil diameter')
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
            df.to_parquet(caption_path('figure2', cache_path=self.cache_path))
        else:
            fig.savefig(path)

        plt.close(fig)
        gc.collect()

    # -------------------------------------------------------------------------------------------------
    # SINGLE TRIAL OVERVIEW
    # -------------------------------------------------------------------------------------------------

    # FIGURE 3

    def make_trial_plot(self, trial_idx, force=False, captions=False):
        path = trial_overview_path(self.eid, trial_idx, cache_path=self.cache_path)
        if not force and path.exists():
            return
        logger.debug(f"making trial overview plot for session {self.eid}, trial #{trial_idx:04d}")
        loader = self.dl

        fig = plt.figure(figsize=(12, 5))

        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[7.5, 15], height_ratios=[5, 3], hspace=0.12,
                               wspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        loader.plot_photometry_signal('calcium', trial_idx=trial_idx, ax=ax1, xlabel='Time in session (s)',
                                      ylabel='Calcium')
        t0 = loader.trial_intervals[trial_idx, 0]
        t1 = loader.trial_intervals[trial_idx, 1]
        loader.plot_photometry_signal('calcium', trial_idx=trial_idx, ax=ax2, xlim=[t0, t1], xlabel=None)
        ax2.get_yaxis().set_visible(False)

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[4], hspace=0.4)
        ax4 = fig.add_subplot(gs1[0, 0])
        ax5 = fig.add_subplot(gs1[1, 0])

        loader.plot_wheel_trace(trial_idx=trial_idx, ax=ax4, ylabel='Wheel pos (rad)', xlabel=None)
        loader.plot_dlc_feature_trace('left', 'paw_r_speed', trial_idx, ax=ax5, xlabel='T in trial (s)',
                                      ylabel='Paw speed (px/s)')

        ax4.sharex(ax2)
        ax5.sharex(ax2)

        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)

        set_figure_style(fig)

        if captions:
            subplots = []
            fig_pos = get_subplot_position(ax1, ax1)
            subplots.append({'panel': 'A', 'xmin': fig_pos[0], 'ymax': fig_pos[1] + 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3] + 0.03})
            fig_pos = get_subplot_position(ax2, ax2)
            subplots.append({'panel': 'B', 'xmin': fig_pos[0], 'ymax': fig_pos[1] + 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3] + 0.03})
            fig_pos = get_subplot_position(ax3, ax3)
            subplots.append({'panel': 'C', 'xmin': fig_pos[0], 'ymax': fig_pos[1] + 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3] + 0.03})
            fig_pos = get_subplot_position(ax4, ax4)
            subplots.append({'panel': 'D', 'xmin': fig_pos[0], 'ymax': fig_pos[1] + 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3] + 0.03})
            fig_pos = get_subplot_position(ax5, ax5)
            subplots.append({'panel': 'E', 'xmin': fig_pos[0], 'ymax': fig_pos[1] + 0.03, 'xmax': fig_pos[2], 'ymin': fig_pos[3] + 0.03})

            df = pd.DataFrame.from_dict(subplots)
            df.to_parquet(caption_path('figure3', cache_path=self.cache_path))
            fig.savefig(path)
        else:
            fig.savefig(path)

        plt.close(fig)
        gc.collect()

    # FIGURE 4

    def make_trial_event_plot(self, force=False, captions=False):

        path = trial_event_overview_path(self.eid, cache_path=self.cache_path)
        if not force and path.exists():
            return
        logger.debug(f"making trial event plot for session {self.p\eid}")
        loader = self.dl

        fig = plt.figure(figsize=(15, 7))

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1])

        gs0 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0, 0], height_ratios=[1, 3], hspace=0, wspace=0.5)
        ax1 = fig.add_subplot(gs0[0, 0])
        ax2 = fig.add_subplot(gs0[1, 0])
        ax3 = fig.add_subplot(gs0[0, 1])
        ax4 = fig.add_subplot(gs0[1, 1])
        ax5 = fig.add_subplot(gs0[0, 2])
        ax6 = fig.add_subplot(gs0[1, 2])
        ax7 = fig.add_subplot(gs0[0, 3])
        ax8 = fig.add_subplot(gs0[1, 3])

        event = 'feedback_times'
        xlabel = 'T from Feedback (s)'
        loader.plot_block_raster(event, axs=[ax1, ax2], xlabel=xlabel)
        loader.plot_contrast_raster(event, axs=[ax3, ax4], xlabel=xlabel, ylabel0=None, ylabel1=None)
        loader.plot_left_right_raster(event, axs=[ax5, ax6], xlabel=xlabel, ylabel0=None, ylabel1=None)
        loader.plot_correct_incorrect_raster(event, axs=[ax7, ax8], xlabel=xlabel, ylabel0=None, ylabel1=None)

        ax1.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax5.get_xaxis().set_visible(False)
        ax7.get_xaxis().set_visible(False)

        ax1.sharex(ax2)
        ax3.sharex(ax4)
        ax5.sharex(ax6)
        ax7.sharex(ax8)

        yax_to_lim = [ax1, ax3, ax5, ax7]
        max_ax = np.max([ax.get_ylim()[1] for ax in yax_to_lim])
        min_ax = np.min([ax.get_ylim()[0] for ax in yax_to_lim])
        for ax in yax_to_lim:
            ax.set_ylim(min_ax, max_ax)

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1, 0], height_ratios=[1, 3], hspace=0, wspace=0.5)
        ax9 = fig.add_subplot(gs1[0, 0])
        ax10 = fig.add_subplot(gs1[1, 0])
        ax11 = fig.add_subplot(gs1[0, 1])
        ax12 = fig.add_subplot(gs1[1, 1])
        ax13 = fig.add_subplot(gs1[0, 2])
        ax14 = fig.add_subplot(gs1[1, 2])
        ax15 = fig.add_subplot(gs1[0, 3])
        ax16 = fig.add_subplot(gs1[1, 3])

        event = 'stimOnTrigger_times'
        xlabel = 'T from Stim on (s)'
        loader.plot_block_raster(event, axs=[ax9, ax10], xlabel=xlabel)
        loader.plot_contrast_raster(event, axs=[ax11, ax12], xlabel=xlabel, ylabel0=None, ylabel1=None)
        loader.plot_left_right_raster(event, axs=[ax13, ax14], xlabel=xlabel, ylabel0=None, ylabel1=None)
        loader.plot_correct_incorrect_raster(event, axs=[ax15, ax16], xlabel=xlabel, ylabel0=None, ylabel1=None)

        ax9.get_xaxis().set_visible(False)
        ax11.get_xaxis().set_visible(False)
        ax13.get_xaxis().set_visible(False)
        ax15.get_xaxis().set_visible(False)

        ax9.sharex(ax10)
        ax11.sharex(ax12)
        ax13.sharex(ax14)
        ax15.sharex(ax16)

        yax_to_lim = [ax9, ax11, ax13, ax15]
        max_ax = np.max([ax.get_ylim()[1] for ax in yax_to_lim])
        min_ax = np.min([ax.get_ylim()[0] for ax in yax_to_lim])
        for ax in yax_to_lim:
            ax.set_ylim(min_ax, max_ax)

        set_figure_style_all(fig, margin_inches=0.8, top=0.95)

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
            fig_pos = get_subplot_position(ax15, ax17)
            subplots.append({'panel': 'G', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax12, ax12)
            subplots.append({'panel': 'H', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax13, ax13)
            subplots.append({'panel': 'I', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            fig_pos = get_subplot_position(ax14, ax14)
            subplots.append({'panel': 'J', 'xmin': fig_pos[0], 'ymax': fig_pos[1], 'xmax': fig_pos[2], 'ymin': fig_pos[3]})
            df = pd.DataFrame.from_dict(subplots)

            df.to_parquet(caption_path('figure5', cache_path=self.cache_path))
            fig.savefig(path)
        else:
            fig.savefig(path)

        plt.close(fig)
        gc.collect()

    # Plot generator functions
    # -------------------------------------------------------------------------------------------------

    def make_all_trial_plots(self, force=False):

        path = trial_overview_path(self.eid, self.first_trial(), cache_path=self.cache_path)
        if not force and path.exists():
            logger.debug("Skipping trial plot generation as they seem to already exist")
            return

        desc = "Making all trial plots  "
        for trial_idx in tqdm(self.iter_trial(), total=self.n_trials, desc=desc):
            self.save_trial_details(trial_idx)
            try:
                self.make_trial_plot(trial_idx, force=force)
            except Exception as e:
                logger.error(f"error with session {self.eid} trial #{trial_idx}: {str(e)}")

    def make_all_plots(self, nums=()):
        if 0 in nums:  # used to regenerate the session.json only
            return
        # nums is a list of numbers 1-5 (figure numbers)

        logger.info(f"Making all session plots for session {self.eid} {nums}")

        # Figure 1
        self.make_session_plot(force=1 in nums)

        # Figure 2
        try:
            self.make_behavior_plot(force=2 in nums)
        except Exception as e:
            logger.error(f"error with session {self.eid} behavior plot: {str(e)}")

        # Figure 3 (one plot per trial)
        self.make_all_trial_plots(force=3 in nums)

        # Figure 4
        self.make_trial_event_plot(force=4 in nums)


    def make_captions(self):
        self.make_session_plot(force=True, captions=True)
        self.make_behavior_plot(force=True, captions=True)
        self.make_trial_plot(self.trial_idxs[0], force=True, captions=True)
        self.make_trial_event_plot(force=True, captions=True)

        caption_json = {}
        for fig in CAPTIONS.keys():
            df = pd.read_parquet(caption_path(fig, cache_path=self.cache_path))
            fig_panels = {}
            for _, row in df.iterrows():
                fig_panels[row['panel']] = {'coords': (row['xmin'], 1 - row['ymax'], row['xmax'], 1 - row['ymin']),
                                            'legend': CAPTIONS[fig][row['panel']]}

            caption_json[fig] = fig_panels

        save_json(figure_details_path(cache_path=self.cache_path), caption_json, indent=2)


def make_captions(cache_path=None, data_path=None):
    eid = get_eids(data_path=data_path)[0]
    Generator(eid, cache_path=cache_path, data_path=data_path).make_captions()


def make_all_plots(eid, nums=(), cache_path=None, data_path=None):
    logger.info(f"Generating all plots for session {eid}")
    Generator(eid, cache_path=cache_path, data_path=data_path).make_all_plots(nums=nums)


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
    eids = sorted([str(p.name) for p in CACHE_DIR.iterdir()])
    eids = [eid for eid in eids if is_valid_uuid(eid)]
    sessions = [load_json_c(session_details_path(eid)) for eid in eids]
    sessions = [_ for _ in sessions if _]
    sessions = sorted(sessions, key=itemgetter('Lab', 'Subject'))
    return sessions


def legends():
    return load_json(figure_details_path())


def generate_data_js():
    FLASK_CTX = {
        "SESSIONS": sessions(),
        "LEGENDS": legends(),
        "DEFAULT_eid": DEFAULT_eid,
        "DEFAULT_DSET": DEFAULT_DSET,
    }
    ctx_json = json.dumps(FLASK_CTX)
    ctx_compressed = lzstring.LZString().compressToBase64(ctx_json)
    return ctx_compressed


def make_data_js():
    ctx_json = generate_data_js()
    path = 'static/data.js'
    with open(path, 'r') as f:
        contents = f.read()
    contents = re.sub('const FLASK_CTX_COMPRESSED = .+', f'const FLASK_CTX_COMPRESSED = "{ctx_json}";', contents)
    with open(path, 'w') as f:
        f.write(contents)


if __name__ == '__main__':

    # Regenerate static/data.js with all the data.
    make_data_js()
    exit()

    # Regenerate all figures.
    if len(sys.argv) == 1:
        Parallel(n_jobs=-4)(delayed(make_all_plots)(eid) for eid in iter_session())

    # Regenerate some figures for all sessions.
    elif len(sys.argv) >= 2 and not is_valid_uuid(sys.argv[1]):
        which = sys.argv[1]

        if which == 'captions':
            make_captions()
        else:
            # which figure numbers to regenerate
            nums = list(map(int, which.split(',')))
            logger.info(f"Regenerating figures {', '.join('#%d' % _ for _ in nums)}")

            # [make_all_plots(eid, nums=nums) for eid in iter_session()]
            Parallel(n_jobs=-3)(delayed(make_all_plots)(eid, nums=nums) for eid in iter_session())

    # Regenerate figures for 1 session.
    elif len(sys.argv) >= 2 and is_valid_uuid(sys.argv[1]):
        nums = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) >= 3 else ()
        make_all_plots(sys.argv[1], nums=nums)
