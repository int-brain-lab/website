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


def roi_details_path(eid, roi_idx, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / f'roi-{roi_idx}.json'


def session_overview_path(eid, roi, preprocess, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / f'session_overview_roi{roi}_pre_{preprocess}.png'


def behaviour_overview_path(eid, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / 'behaviour_overview.png'


def trial_raster_overview_path(eid, roi, preprocess, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / f'trial_overview_raster_roi{roi}_pre_{preprocess}.png'


def trial_psth_overview_path(eid, roi, preprocess, cache_path=None):
    return session_cache_path(eid, cache_path=cache_path) / f'trial_overview_psth_roi{roi}_pre_{preprocess}.png'


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
        self.dl.load_photometry_data(DEFAULT_ROI)
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
        self.n_rois = self.dl.n_rois


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

    def save_roi_details(self, roi_idx):
        logger.debug(f"saving roi #{roi_idx} details for session {self.eid}")
        details = self.dl.get_roi_details(roi_idx)
        path = roi_details_path(self.eid, roi_idx, cache_path=self.cache_path)
        save_json(path, details)

    # -------------------------------------------------------------------------------------------------
    # SESSION OVERVIEW
    # -------------------------------------------------------------------------------------------------

    # FIGURE 1
    def make_session_plot(self, roi, preprocess, force=False, captions=False):

        path = session_overview_path(self.eid, roi, preprocess, cache_path=self.cache_path)
        if not force and path.exists():
            return
        logger.debug(f"making session overview plot for session: {self.eid}, roi: {roi}, "
                     f"preprocessing: {preprocess}")

        try:
            fig = plt.figure(figsize=(15, 8))

            gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[6, 4], wspace=0.1, hspace=0.3)

            # First column
            gs0 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0], hspace=0.8, height_ratios=[1, 1, 1, 2])

            gs01 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0:3], hspace=0.3)
            gs0_ax1 = fig.add_subplot(gs01[0, 0])
            gs0_ax2 = fig.add_subplot(gs01[1, 0])
            gs0_ax3 = fig.add_subplot(gs01[2, 0])

            # Full session photometry signal
            _, gs0_ax1_r = self.dl.plot_raw_photometry_signal(ax=gs0_ax1, xlabel=None, ylabel='Raw isobestic')
            self.dl.plot_photometry_signal(preprocess, ax=gs0_ax2, xlabel=None, ylabel=preprocess.capitalize())
            self.dl.plot_photometry_signal(preprocess, mvg_avg=True, ax=gs0_ax3, xlabel='Time in session (s)',
                                           ylabel=f'Mov avg {preprocess}')

            gs0_ax1_r.yaxis.set_ticklabels([])

            # Session behavior plots
            gs11 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[3, 0], height_ratios=[2, 3, 1, 3])
            ax_a = fig.add_subplot(gs11[0, 0])
            ax_b = fig.add_subplot(gs11[1, 0])
            ax_c = fig.add_subplot(gs11[2, 0])
            ax_d = fig.add_subplot(gs11[3, 0])

            # Second column
            gs1 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs[1], hspace=0.3, wspace=0.7,
                                                   height_ratios=[1, 1, 1, 2])
            ax_leg = fig.add_subplot(gs1[3, 0])
            ax_cor = fig.add_subplot(gs1[3, 1])

            self.dl.plot_session_reaction_time(ax=ax_a)
            self.dl.plot_session_contrasts(axs=[ax_b, ax_c, ax_d], ax_legend=ax_leg)

            remove_frame(ax_cor)
            # self.dl.plot_coronal_slice(ax_cor)

            gs0_ax4 = fig.add_subplot(gs1[0, 0])
            gs0_ax5 = fig.add_subplot(gs1[1, 0])
            gs0_ax6 = fig.add_subplot(gs1[2, 0])

            self.dl.plot_raw_photometry_signal(ax=gs0_ax4, xlim=self.dl.photometry_lim, xlabel=None,
                                               ylabel2='Raw calcium')
            self.dl.plot_photometry_signal(preprocess, xlim=self.dl.photometry_lim,  xlabel=None, ax=gs0_ax5)
            self.dl.plot_photometry_signal(preprocess, mvg_avg=True, xlim=self.dl.photometry_lim,
                                           xlabel='Time in session (s)', ax=gs0_ax6)

            gs0_ax4.yaxis.set_ticklabels([])
            gs0_ax5.yaxis.set_ticklabels([])
            gs0_ax6.yaxis.set_ticklabels([])

            gs0_ax7 = fig.add_subplot(gs1[1:3, 1])
            gs0_ax8 = fig.add_subplot(gs1[0, 1])

            self.dl.plot_photometry_correlation(ax=gs0_ax7, ax_cbar=gs0_ax8)
            remove_frame(gs0_ax8)

            fig.subplots_adjust(left=0.08, right=1-0.08, bottom=0.08, top=1-0.08)

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

        fig = plt.figure(figsize=(15, 10))

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 3], hspace=0.3)

        gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], width_ratios=[4, 1, 4, 4], wspace=0.4)
        ax1 = fig.add_subplot(gs1[0, 0])
        ax2 = fig.add_subplot(gs1[0, 1])
        ax3 = fig.add_subplot(gs1[0, 2])
        ax4 = fig.add_subplot(gs1[0, 3])
        self.dl.plot_psychometric_curve(ax=ax1, ax_legend=ax2)
        self.dl.plot_chronometric_curve(ax=ax3)
        self.dl.plot_reaction_time(ax=ax4)

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

        # Attempt to plot DLC
        if self.dl.camera_flag:
            self.dl.plot_dlc_feature_raster('left', 'paw_r_speed', axs=[ax5, ax6], ylabel0='Speed (px/s)', title='Left paw')
            self.dl.plot_dlc_feature_raster('left', 'nose_tip_speed', axs=[ax7, ax8], ylabel0='Speed (px/s)', ylabel1=None,
                                            title='Nose tip')
            self.dl.plot_dlc_feature_raster('left', 'motion_energy', axs=[ax9, ax10], zscore_flag=True, ylabel0='ME (z-score)',
                                            ylabel1=None, title='Motion energy')
            self.dl.plot_dlc_feature_raster('left', 'pupilDiameter_smooth', axs=[ax11, ax12], zscore_flag=True, norm=True,
                                            ylabel0='Pupil (z-score)', ylabel1=None, title='Pupil diameter')
        else:
            for ax in [ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
                remove_frame(ax)

        # Attempt to plot wheel
        if self.dl.wheel_flag:
            self.dl.plot_wheel_raster(axs=[ax13, ax14], ylabel0='Velocity (rad/s)', ylabel1=None, title='Wheel velocity')
        else:
            for ax in [ax13, ax14]:
                remove_frame(ax)

        # Attempt to plot licks
        if self.dl.lick_flag:
            self.dl.plot_lick_raster(axs=[ax15, ax16], ylabel1=None, title='Licks')
        else:
            for ax in [ax15, ax16]:
                remove_frame(ax)

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

        fig.savefig(path)

        plt.close(fig)
        gc.collect()

    # FIGURE 3
    def make_trial_raster_plot(self, roi, preprocess, force=False, captions=False):

        path = trial_raster_overview_path(self.eid, roi, preprocess, cache_path=self.cache_path)
        if not force and path.exists():
            return
        logger.debug(f"making trial raster plot for session: {self.eid}, roi: {roi}, "
                     f"preprocessing: {preprocess}")

        fig = plt.figure(figsize=(15, len(PSTH_EVENTS.keys()) * 5))

        gs = gridspec.GridSpec(len(PSTH_EVENTS.keys()), 1, figure=fig)

        for i, event in enumerate(PSTH_EVENTS.keys()):

            gs0 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[i, 0], height_ratios=[1, 3], hspace=0,
                                                   wspace=0.5)
            ax1 = fig.add_subplot(gs0[0, 0])
            ax2 = fig.add_subplot(gs0[1, 0])
            ax3 = fig.add_subplot(gs0[0, 1])
            ax4 = fig.add_subplot(gs0[1, 1])
            ax5 = fig.add_subplot(gs0[0, 2])
            ax6 = fig.add_subplot(gs0[1, 2])
            ax7 = fig.add_subplot(gs0[0, 3])
            ax8 = fig.add_subplot(gs0[1, 3])

            xlabel = PSTH_EVENTS[event]
            self.dl.plot_block_raster(preprocess, event, axs=[ax1, ax2], xlabel=xlabel)
            self.dl.plot_contrast_raster(preprocess, event, axs=[ax3, ax4], xlabel=xlabel, ylabel0=None, ylabel1=None)
            self.dl.plot_left_right_raster(preprocess, event, axs=[ax5, ax6], xlabel=xlabel, ylabel0=None, ylabel1=None)
            self.dl.plot_correct_incorrect_raster(preprocess, event, axs=[ax7, ax8], xlabel=xlabel, ylabel0=None,
                                                  ylabel1=None)

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
                ax.vlines(0, *ax.get_ylim(), color='k', ls='--', zorder=ax.get_zorder() + 1)


        fig.savefig(path)

        plt.close(fig)
        gc.collect()

    # FIGURE 4
    def make_trial_psth_plot(self, roi, preprocess, force=False, captions=False):

        path = trial_psth_overview_path(self.eid, roi, preprocess, cache_path=self.cache_path)
        if not force and path.exists():
            return
        logger.debug(f"making trial psth plot for session: {self.eid}, roi: {roi}, "
                     f"preprocessing: {preprocess}")

        fig = plt.figure(figsize=(15, len(PSTH_EVENTS.keys()) * 5))

        gs = gridspec.GridSpec(len(PSTH_EVENTS.keys()), 1, figure=fig)

        for i, event in enumerate(PSTH_EVENTS.keys()):

            gs0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[i, 0], wspace=0.5)
            ax1 = fig.add_subplot(gs0[0, 0])
            ax2 = fig.add_subplot(gs0[0, 1])
            ax3 = fig.add_subplot(gs0[0, 2])
            ax4 = fig.add_subplot(gs0[0, 3])

            xlabel = PSTH_EVENTS[event]
            self.dl.plot_block_raster(preprocess, event, axs=[ax1], xlabel=xlabel, ylabel0='Signal')
            self.dl.plot_contrast_raster(preprocess, event, axs=[ax2], xlabel=xlabel, ylabel0=None, ylabel1=None)
            self.dl.plot_left_right_raster(preprocess, event, axs=[ax3], xlabel=xlabel, ylabel0=None, ylabel1=None)
            self.dl.plot_correct_incorrect_raster(preprocess, event, axs=[ax4], xlabel=xlabel, ylabel0=None, ylabel1=None)

            yax_to_lim = [ax1, ax2, ax3, ax4]
            max_ax = np.max([ax.get_ylim()[1] for ax in yax_to_lim])
            min_ax = np.min([ax.get_ylim()[0] for ax in yax_to_lim])
            for ax in yax_to_lim:
                ax.set_ylim(min_ax, max_ax)
                ax.vlines(0, *ax.get_ylim(), color='k', ls='--', zorder=ax.get_zorder() + 1)


        fig.savefig(path)

        plt.close(fig)
        gc.collect()
    # -------------------------------------------------------------------------------------------------
    # SINGLE TRIAL OVERVIEW
    # -------------------------------------------------------------------------------------------------

    # FIGURE 5

    def make_trial_plot(self, trial_idx, force=False, captions=False):
        path = trial_overview_path(self.eid, trial_idx, cache_path=self.cache_path)
        if not force and path.exists():
            return
        logger.debug(f"making trial overview plot for session {self.eid}, trial #{trial_idx:04d}")

        fig = plt.figure(figsize=(12, 5))

        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[7.5, 15], height_ratios=[5, 3], hspace=0.12,
                               wspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        self.dl.plot_photometry_signal('calcium', trial_idx=trial_idx, ax=ax1, xlabel='Time in session (s)',
                                       ylabel='Calcium')
        t0 = self.dl.trial_intervals[trial_idx, 0]
        t1 = self.dl.trial_intervals[trial_idx, 1]
        self.dl.plot_photometry_signal('calcium', trial_idx=trial_idx, ax=ax2, xlim=[t0, t1], xlabel=None)
        ax2.get_yaxis().set_visible(False)

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.4)
        ax3 = fig.add_subplot(gs1[0, 0])
        ax4 = fig.add_subplot(gs1[1, 0])

        if self.dl.camera_flag and self.dl.wheel_flag:
            self.dl.plot_wheel_trace(trial_idx=trial_idx, ax=ax3, ylabel='Wheel pos (rad)', xlabel=None)
            self.dl.plot_dlc_feature_trace('left', 'paw_r_speed', trial_idx, ax=ax4, xlabel='T in trial (s)',
                                           ylabel='Paw speed (px/s)')
        elif self.dl.wheel_flag:
            self.dl.plot_wheel_trace(trial_idx=trial_idx, ax=ax3, ylabel='Wheel pos (rad)', xlabel='T in trial (s)')
            remove_frame(ax4)
        else:
            remove_frame(ax3)
            remove_frame(ax4)

        ax3.sharex(ax2)
        ax4.sharex(ax2)

        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)

        set_figure_style(fig)

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

        # Figure 2
        try:
            self.make_behavior_plot(force=2 in nums)
        except Exception as e:
            logger.error(f"error with session {self.eid} behavior plot: {str(e)}")

        for roi in range(self.n_rois):
            self.dl.load_photometry_data(roi)
            self.save_roi_details(roi)
            for preprocess in PREPROCESS:
                # Figure 1
                self.make_session_plot(roi, preprocess, force=1 in nums)
                # Figure 3
                self.make_trial_raster_plot(roi, preprocess, force=3 in nums)
                # Figure 4
                self.make_trial_psth_plot(roi, preprocess, force=4 in nums)


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
            pass
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
