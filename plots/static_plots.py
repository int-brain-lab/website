# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy as np
import pandas as pd
import copy
from collections import OrderedDict
import seaborn as sns
import yaml
from scipy.stats import zscore
import scipy.signal

from brainbox.task.trials import find_trial_ids
from brainbox.behavior.wheel import velocity_filtered, interpolate_position
from brainbox.behavior.training import (plot_psychometric, plot_reaction_time, plot_reaction_time_over_trials, get_signed_contrast,
compute_reaction_time)
from iblatlas.atlas import AllenAtlas, Insertion, Trajectory
from iblutil.util import Bunch
import iblphotometry.preprocessing as iblphot

import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound

from one.api import ONE


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / 'static/data'
CACHE_DIR = ROOT_DIR / 'static/cache'

# -------------------------------------------------------------------------------------------------
# Colourmaps
# -------------------------------------------------------------------------------------------------

CMAP = sns.diverging_palette(20, 220, n=3, center="dark")

LINE_COLOURS = {
    'raw_isosbestic': '#9d4edd',
    'raw_calcium': '#43aa8b',
    'calcium': '#0081a7',
    'isosbestic_control': '#0081a7',
    'photobleach': '#0081a7',
    'moving_avg': '#f4a261'
}

# -------------------------------------------------------------------------------------------------
# Photometry variables
# -------------------------------------------------------------------------------------------------
DEFAULT_ROI = 0
DEFAULT_PREPROCESS = 'calcium'
PREPROCESS = [
    'calcium',
    'isosbestic_control',
    'photobleach'
]

PSTH_EVENTS = {
    'feedback_times': 'T from Feedback (s)',
    'stimOnTrigger_times': 'T from Stim on (s)',
    'firstMovement_times': 'T from First move (s)'
}

SUBJECT_ROI = {
    "ZFM-04022": 'VTA, SNc', # Ask Kcenia about these ones
    "ZFM-04026": 'VTA',
    "ZFM-03447": 'VTA',
    "ZFM-04019": 'SNc',
    "ZFM-03448": 'SNc, SI',
    "ZFM-05245": 'DR',
    "ZFM-05248": 'DR',
    "ZFM-05235": 'DR',
    "ZFM-05236": 'DR',
    "ZFM-04392": 'DR',
    "ZFM-03059": 'DR',
    "ZFM-03061": 'DR',
    "ZFM-03065": 'DR',
    "ZFM-06271": 'LC',
    "ZFM-06272": 'LC',
    "ZFM-04533": 'LC',
    "ZFM-04534": 'LC',
    "ZFM-06275": 'LC',
    "ZFM-06268": 'LC',
    "ZFM-06171": 'LC',
    "ZFM-06305": 'SI',
    "ZFM-06948": 'SI',
}

# -------------------------------------------------------------------------------------------------
# Loading functions
# -------------------------------------------------------------------------------------------------

def load_trials(eid, one, data_path=None):
    data_path = data_path or DATA_DIR
    trials = one.load_object(eid, 'trials')
    return trials


def load_photometry(eid, roi=None, data_path=None):
    data_path = data_path or DATA_DIR
    photometry = pd.read_parquet(data_path.joinpath('alf', roi, 'raw_photometry.pqt'))
    return photometry


def load_camera(eid, one, camera, data_path=None):
    data_path = data_path or DATA_DIR
    try:
        camera = one.load_object(eid, f'{camera}Camera', collection='alf')
    except ALFObjectNotFound:
        camera = None
    return camera


def load_licks(eid, one, data_path=None):
    data_path = data_path or DATA_DIR
    try:
        licks = one.load_object(eid, 'licks')
    except ALFObjectNotFound:
        licks = None
    return licks


def load_wheel(eid, one, data_path=None):
    data_path = data_path or DATA_DIR
    try:
        wheel = one.load_object(eid, 'wheel')
    except ALFObjectNotFound:
        wheel = None
    return wheel


# -------------------------------------------------------------------------------------------------
# Filtering
# -------------------------------------------------------------------------------------------------
def _filter(obj, idx):
    obj = Bunch(copy.deepcopy(obj))
    for key in obj.keys():
        obj[key] = obj[key][idx]

    return obj


def filter_trials_by_trial_idx(trials, trial_idx):
    return _filter(trials, trial_idx)


# -------------------------------------------------------------------------------------------------
# Processing functions
# -------------------------------------------------------------------------------------------------

def bin_spikes(spike_times, align_times, pre_time, post_time, bin_size, weights=None):

    align_times = align_times.values if isinstance(align_times, pd.Series) else align_times
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], n_bins))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (
            np.floor((spike_times[ep[0]:ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0]:ep[1]] if weights is not None else None
        r = np.bincount(xind, minlength=tscale.shape[0], weights=w)
        if w is not None:
            r_norm = np.bincount(xind, minlength=tscale.shape[0])
            r_norm[r_norm == 0] = 1
            r = r / r_norm
        bins[i, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale


def interpolate_along_track(xyz_track, depths):

    # this is the cumulative distance from the lowest picked point (first)
    distance = np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz_track, axis=0) ** 2, axis=1))])
    xyz_channels = np.zeros((depths.shape[0], 3))
    for m in np.arange(3):
        xyz_channels[:, m] = np.interp(depths, distance, xyz_track[:, m])
    return xyz_channels


def compute_psth(signal, times, events, fs, peri_event_window=None):

    peri_event_window = [-1, 2] if peri_event_window is None else peri_event_window
    # compute a vector of indices corresponding to the perievent window at the given sampling rate
    sample_window = np.arange(peri_event_window[0] * fs, peri_event_window[1] * fs + 1)
    # we inflate this vector to a 2d array where each column corresponds to an event
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, events.size))
    # we add the index of each event too their respective column
    idx_event = np.searchsorted(times, events)
    idx_psth += idx_event
    
    nan_idx = np.where(idx_psth >= signal.size)
    if nan_idx[1].size > 0:
        nan_idx = np.unique(nan_idx[1])
        nan_append = idx_psth[:, nan_idx[0]:]
        idx_psth = idx_psth[:, 0:nan_idx[0]]
        psth = signal[idx_psth]  # psth is a 2d array (ntimes, nevents)
        psth = np.c_[psth, nan_append]
    else:
        psth = signal[idx_psth]

    psth[idx_psth > (signal.size - 1)] = np.nan
    # remove events that are out of bounds
    return psth


# -------------------------------------------------------------------------------------------------
# Styling functions
# -------------------------------------------------------------------------------------------------

def set_figure_style(fig, margin_inches=0.8):
    x_inches, y_inches = fig.figure.get_size_inches()
    fig.subplots_adjust(margin_inches / x_inches, margin_inches / y_inches)
    return fig


def set_figure_style_all(fig, margin_inches=0.8, top=0.99):
    x_inches, y_inches = fig.figure.get_size_inches()
    fig.subplots_adjust(left=margin_inches / x_inches, bottom=margin_inches / y_inches, right=1 - margin_inches / x_inches,
                        top=top)
    return fig


def set_axis_style(ax, fontsize=12, **kwargs):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(kwargs.get('xlabel', None), fontsize=fontsize)
    ax.set_ylabel(kwargs.get('ylabel', None), fontsize=fontsize)
    ax.set_title(kwargs.get('title', None), fontsize=fontsize)

    return ax


def remove_spines(ax, spines=('left', 'right', 'top', 'bottom')):
    for sp in spines:
        ax.spines[sp].set_visible(False)

    return ax


def remove_frame(ax):

    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    return ax


# -------------------------------------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------------------------------------

class DataLoader:

    # Loading functions
    # ---------------------------------------------------------------------------------------------

    def __init__(self, one=None, data_path=None):
        self.data_path = data_path or DATA_DIR
        self.one = one or ONE()

        # self.BRAIN_ATLAS = AllenAtlas()
        # self.BRAIN_ATLAS.compute_surface()
        # self.BRAIN_REGIONS = self.BRAIN_ATLAS.regions

    def session_init(self, eid):

        self.eid = eid
        self.load_session_data(eid)

    def load_session_data(self, eid):
        """
        Load in the data associated with selected pid
        :param pid:
        :return:
        """

        session_path = self.one.eid2path(eid)
        session_info = self.one.path2record(session_path).to_dict()
        subject = self.one.alyx.rest('subjects', 'list', nickname=session_info['subject'])[0]

        self.regions = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]

        self.n_rois = len(self.regions)

        self.session_info = {
                'eid': eid,
                'lab': session_info['lab'],
                'subject': session_info['subject'],
                'date': str(session_info['date']),
                'dob': subject['birth_date'],
                'roi': self.n_rois,
                'protocol': session_info['task_protocol']
            }

        # Load photometry data for default roi and preprocessing
        self.photometry = load_photometry(eid, roi=self.regions[0], data_path=self.data_path)

        # Load trials data
        self.trials = load_trials(eid, self.one, data_path=self.data_path)
        self.trial_intervals, self.trial_idx = self.compute_trial_intervals()

        # Load wheel data
        self.wheel = load_wheel(eid, self.one, data_path=self.data_path)
        if self.wheel is None:
            self.wheel_flag = False
        else:
            self.wheel_flag = True

        # Load camera data
        self.camera = load_camera(eid, self.one, 'left', data_path=self.data_path)
        if self.camera is None or 'computedFeatures' not in self.camera.keys():
            self.camera_flag = False
        else:
            self.camera = True

        # Load lick data
        self.licks = load_licks(eid, self.one, data_path=self.data_path)
        if self.licks is None:
            self.lick_flag = False
        else:
            self.lick_flag = True

    def load_photometry_data(self, roi):

        self.photometry = load_photometry(self.eid, roi=self.regions[roi], data_path=self.data_path)
        self.preprocess_photometry_data()
        self.psth = self.compute_photometry_psth()

        self.photometry_lim = [self.photometry.times[int(self.photometry.times.size/2)],
                               self.photometry.times[int(self.photometry.times.size/2) + 250]]

    def preprocess_photometry_data(self):

        self.photometry = iblphot.isosbestic_correction_dataframe(self.photometry)
        self.photometry['photobleach'] = iblphot.photobleaching_lowpass(self.photometry['raw_calcium'].values)

    def compute_photometry_psth(self):
        psth_preprocess = Bunch()
        event_window = [-1, 2]
        fs = 30
        for preprocess in PREPROCESS:
            psth = Bunch()
            for event in PSTH_EVENTS.keys():
                psth[event] = compute_psth(self.photometry[preprocess].values, self.photometry['times'].values,
                                           self.trials[event], fs, peri_event_window=event_window).T
            psth_preprocess[preprocess] = psth

        psth_preprocess['times'] = np.arange(event_window[0] * fs, event_window[1] * fs + 1)

        return psth_preprocess

    def get_session_details(self):
        """
        Get dict of metadata for session
        :return:
        """

        details = OrderedDict()
        details['ID'] = self.eid
        details['Subject'] = self.session_info['subject']
        details['Lab'] = self.session_info['lab']
        details['DOB'] = self.session_info['dob']
        details['Recording date'] = self.session_info['date']
        details['Recording length'] = f'{int(np.max(self.photometry["times"]) / 60)} minutes'
        details['N trials'] = f'{self.trials.stimOn_times.size}'
        details['N rois'] = self.n_rois
        details['Protocol'] = self.session_info['protocol']
        details['eid'] = self.eid

        # Internal fields used by the frontend.
        details['_trial_ids'] = [int(_) for _ in self.trial_idx]
        details['_roi_ids'] = [int(_) for _ in range(self.n_rois)]
        # details['_preprocess_ids'] = [int(_) for _ in range(4)]

        # Trial intervals.
        details['_trial_onsets'] = [float(_) if not np.isnan(_) else None for _ in self.trial_intervals[:, 0]]
        details['_trial_offsets'] = [float(_) if not np.isnan(_) else None for _ in self.trial_intervals[:, 1]]

        # Session duration
        details['_duration'] = np.max(self.photometry["times"])

        return details

    def get_roi_details(self, roi_idx):
        """
        Get dict of metadata for rois
        :return:
        """

        details = OrderedDict()
        details['name'] = self.regions[roi_idx]
        details['acronym'] = SUBJECT_ROI[self.session_info['subject']]

        return details

    def get_trial_details(self, trial_idx):
        trials = filter_trials_by_trial_idx(self.trials, trial_idx)

        def _get_block_probability(pLeft):
            if pLeft == 0.5:
                return 'neutral'
            elif pLeft == 0.8:
                return 'left'
            elif pLeft == 0.2:
                return 'right'

        details = OrderedDict()
        details['Trial #'] = trial_idx
        details['Contrast'] = np.nanmean([trials.contrastLeft, trials.contrastRight]) * 100
        details['Stim side'] = 'left' if np.isnan(trials.contrastRight) else 'right'
        details['Block proba'] = _get_block_probability(trials.probabilityLeft)
        details['Resp type'] = 'correct' if trials.feedbackType == 1 else 'incorrect'
        details['Resp time'] = f'{np.round((trials.feedback_times - trials.stimOn_times) * 1e3, 0)} ms'
        details['Mov time'] = f'{np.round((trials.firstMovement_times - trials.stimOn_times) * 1e3, 0)} ms'

        return details

    def compute_trial_intervals(self):
        # Find the nan trials and remove these
        nan_idx = np.bitwise_or(np.isnan(self.trials['stimOn_times']), np.isnan(self.trials['feedback_times']))
        trial_no = np.arange(len(self.trials['stimOn_times']))

        t0 = np.nanmax(np.c_[self.trials['stimOn_times'] - 1, self.trials['intervals'][:, 0]], axis=1)
        t1 = np.nanmin(np.c_[self.trials['feedback_times'] + 1.5, self.trials['intervals'][:, 1]], axis=1)

        # For the first trial limit t0 to be 0.18s before stimOn, to be consistent with the videos
        t0[0] = self.trials['stimOn_times'][0] - 0.18

        t0[nan_idx] = None
        t1[nan_idx] = None

        return np.c_[t0, t1], trial_no[~nan_idx]

    # Plotting functions
    # ---------------------------------------------------------------------------------------------

    def plot_raw_photometry_signal(self, ax=None, xlim=None, ylim=None, ylim2=None, xlabel='Time', ylabel=None,
                                   ylabel2=None, title=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        ax_r = ax.twinx()

        linewidth = 0.1 if xlim is None else 1

        ax.plot(self.photometry['times'], self.photometry['raw_isosbestic'], linewidth=linewidth,
                c=LINE_COLOURS['raw_isosbestic'])
        ax_r.plot(self.photometry['times'], self.photometry['raw_calcium'], linewidth=linewidth,
                  c=LINE_COLOURS['raw_calcium'])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax_r.set_ylim(ylim2)

        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
        set_axis_style(ax_r, xlabel=xlabel, ylabel=ylabel2, title=title)
        ax_r.spines['right'].set_visible(True)

        ax.tick_params(axis='both', which='major')
        ax_r.tick_params(axis='both', which='major')

        return fig, ax_r

    def plot_photometry_signal(self, signal, mvg_avg=False, trial_idx=None, ax=None, xlim=None, ylim=None,
                               xlabel='Time', ylabel=None, title=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        linewidth = 0.1 if xlim is None else 1

        if mvg_avg:
            window_size = 250
            phot_signal = self.photometry[signal].rolling(window=window_size).mean()
            col = LINE_COLOURS['moving_avg']
        else:
            phot_signal = self.photometry[signal]
            col = LINE_COLOURS[signal]

        ax.plot(self.photometry['times'], phot_signal, linewidth=linewidth, c=col)

        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.tick_params(axis='both', which='major')

        if trial_idx is not None:
            if xlim is None:
                trials = filter_trials_by_trial_idx(self.trials, trial_idx)
                ax.vlines(trials['intervals'][:, 0], *ax.get_ylim(), color='k', ls='--', zorder=ax.get_zorder() + 1)
                ax.text(trials['intervals'][:, 0], 1.01, f'Trial {trial_idx}', c='k', rotation=45,
                        rotation_mode='anchor', ha='left', transform=ax.get_xaxis_transform())
            else:
                trials = filter_trials_by_trial_idx(self.trials, trial_idx)
                self.add_trial_events_to_raster(ax, trials)

        return fig

    def plot_photometry_correlation(self, ax=None, ax_cbar=None, title=None):

        if ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [1, 10]})
            ax = axs[1]
            ax_cbar = axs[0]
        else:
            fig = ax.get_figure()

        sos = scipy.signal.butter(**{'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}, output='sos')
        calcium_lp = scipy.signal.sosfiltfilt(sos, self.photometry['raw_calcium'])
        isosbestic_lp = scipy.signal.sosfiltfilt(sos, self.photometry['raw_isosbestic'])

        scat = ax.scatter(isosbestic_lp, calcium_lp, s=1, c=self.photometry['times'],
                          cmap='magma', alpha=.8)
        set_axis_style(ax, xlabel='raw isobestic', ylabel='raw calcium', title=title)
        fig.colorbar(scat, ax=ax_cbar, orientation='horizontal', label='Time in session (s)')

        return fig

    def plot_session_contrasts(self, axs=None, ax_legend=None):

        if axs is None:
            fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 3], 'hspace': 0.1})
        else:
            fig = axs[0].get_figure()

        contrasts = np.nansum(np.c_[-1 * self.trials['contrastLeft'], self.trials['contrastRight']], axis=1)

        right_correct_trials = np.bitwise_and(contrasts > 0, self.trials.feedbackType == 1)
        right_incorrect_trials = np.bitwise_and(contrasts > 0, self.trials.feedbackType == -1)
        left_correct_trials = np.bitwise_and(contrasts < 0, self.trials.feedbackType == 1)
        left_incorrect_trials = np.bitwise_and(contrasts < 0, self.trials.feedbackType == -1)
        zero_correct_trials = np.bitwise_and(contrasts == 0, self.trials.feedbackType == 1)
        zero_incorrect_trials = np.bitwise_and(contrasts == 0, self.trials.feedbackType == -1)

        axs[0].scatter(self.trials['stimOn_times'][right_correct_trials], contrasts[right_correct_trials] * 100,
                       facecolors='none', edgecolors='b', s=9)
        axs[0].scatter(self.trials['stimOn_times'][right_incorrect_trials], contrasts[right_incorrect_trials] * 100,
                       marker='x', c='r', s=9)

        axs[1].scatter(self.trials['stimOn_times'][zero_correct_trials], contrasts[zero_correct_trials] * 100,
                       facecolors='none', edgecolors='b', s=9)
        axs[1].scatter(self.trials['stimOn_times'][zero_incorrect_trials], contrasts[zero_incorrect_trials] * 100,
                       marker='x', c='r', s=9)

        axs[2].scatter(self.trials['stimOn_times'][left_correct_trials], np.abs(contrasts[left_correct_trials]) * 100,
                       facecolors='none', edgecolors='b', s=9)
        axs[2].scatter(self.trials['stimOn_times'][left_incorrect_trials], np.abs(contrasts[left_incorrect_trials]) * 100,
                       marker='x', c='r', s=9)

        dividers = np.where(np.diff(self.trials['probabilityLeft']) != 0)[0]
        blocks = self.trials['probabilityLeft'][np.r_[0, dividers + 1]]

        colours = np.full((blocks.shape[0], 3), np.array([*CMAP[0]]))
        colours[np.where(blocks == 0.5)] = np.array([*CMAP[1]])
        colours[np.where(blocks == 0.8)] = np.array([*CMAP[2]])

        dividers = [0] + list(dividers) + [np.where(~np.isnan(self.trials.stimOn_times))[0][-1]]  # sometimes last trial is nan
        for ax in axs:
            for iD in range(len(dividers) - 1):
                ax.fill_betweenx([-100, 100],
                                 [self.trials.stimOn_times[dividers[iD + 1]], self.trials.stimOn_times[dividers[iD + 1]]],
                                 [self.trials.stimOn_times[dividers[iD]], self.trials.stimOn_times[dividers[iD]]], color=colours[iD],
                                 alpha=0.2)

        axs[0].set_yscale('log')
        axs[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axs[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[0].set_yticks([25, 100])
        axs[0].set_ylim([5, 101])
        axs[0].minorticks_off()

        axs[1].set_yticks([0])
        axs[1].set_ylim(-1, 1)

        axs[2].set_yscale('log')
        axs[2].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axs[2].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[2].set_yticks([25, 100])
        axs[0].set_ylim([5, 101])
        axs[2].invert_yaxis()
        axs[2].minorticks_off()

        axs[0].xaxis.tick_top()
        remove_spines(axs[0], spines=['bottom', 'right'])
        axs[0].set_xticklabels([])
        axs[0].set_ylabel('Right', fontdict={'size': 10}, labelpad=-2)
        axs[0].spines['top'].set_visible(True)

        remove_spines(axs[1], spines=['bottom', 'right', 'top'])
        axs[1].axes.get_xaxis().set_visible(False)
        axs[1].set_ylabel('Contrasts (%)', fontdict={'size': 12}, labelpad=25)

        axs[2].set_ylabel('Left', fontdict={'size': 10}, labelpad=-2)
        axs[2].axes.get_xaxis().set_visible(False)
        remove_spines(axs[2], spines=['bottom', 'right', 'top'])

        legend_elements = [Line2D([0], [0], color=CMAP[0], lw=4, alpha=0.5, label='20 % of trials on left side'),
                           Line2D([0], [0], color=CMAP[1], lw=4, alpha=0.5, label='equal % of trials on both sides'),
                           Line2D([0], [0], color=CMAP[2], lw=4, alpha=0.5, label='80 % of trials on left side'),
                           Line2D([0], [0], color='w', marker='o', markeredgecolor='b', label='correct', markersize=10),
                           Line2D([0], [0], color='w', marker='x', markeredgecolor='r', label='incorrect', markersize=10)]

        ax_legend.legend(handles=legend_elements, loc=3, frameon=False)
        remove_frame(ax_legend)

        return fig

    def plot_session_reaction_time(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        reaction_time = self.trials['response_times'] - self.trials['stimOn_times']
        correct_idx = np.where(self.trials.feedbackType == 1)[0]
        incorrect_idx = np.where(self.trials.feedbackType == -1)[0]

        ax.scatter(self.trials['stimOn_times'][correct_idx], reaction_time[correct_idx], facecolors='none', edgecolors='b',
                   s=7)
        ax.scatter(self.trials['stimOn_times'][incorrect_idx], reaction_time[incorrect_idx], facecolors='none', marker='x',
                   c='r', s=7)
        ax.set_yscale('log')
        ax.set_yticks([1, 10])
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.minorticks_off()

        dividers = np.where(np.diff(self.trials['probabilityLeft']) != 0)[0]
        blocks = self.trials['probabilityLeft'][np.r_[0, dividers + 1]]

        colours = np.full((blocks.shape[0], 3), np.array([*CMAP[0]]))
        colours[np.where(blocks == 0.5)] = np.array([*CMAP[1]])
        colours[np.where(blocks == 0.8)] = np.array([*CMAP[2]])

        dividers = [0] + list(dividers) + [np.where(~np.isnan(self.trials.stimOn_times))[0][-1]]  # sometimes last trial is nan
        for iD in range(len(dividers) - 1):
            ax.fill_betweenx([-100, 100],
                             [self.trials.stimOn_times[dividers[iD + 1]], self.trials.stimOn_times[dividers[iD + 1]]],
                             [self.trials.stimOn_times[dividers[iD]], self.trials.stimOn_times[dividers[iD]]], color=colours[iD],
                             alpha=0.2)

        ax.xaxis.tick_top()
        set_axis_style(ax, ylabel='RT (s)', fontsize=11)
        ax.spines['top'].set_visible(True)
        remove_spines(ax, spines=['bottom'])

        return fig

    def plot_coronal_slice(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        ins = Insertion.from_dict(self.session, brain_atlas=self.BRAIN_ATLAS)

        ax, sec_ax = self.BRAIN_ATLAS.plot_tilted_slice(ins.xyz, 1, volume='annotation', ax=ax, return_sec=True)
        ax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, 'k', linewidth=2)
        # ax.plot(top_bottom[:, 0] * 1e6, top_bottom[:, 2] * 1e6, 'grey', linewidth=2)
        remove_frame(ax)
        sec_ax.get_yaxis().set_visible(False)

        return fig

    def plot_brain_slice(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        cmin, cmax = np.quantile(self.BRAIN_ATLAS.image, [0.1, 0.98])

        ins = Insertion.from_dict(self.session, brain_atlas=self.BRAIN_ATLAS)

        depths = np.arange(0, 4000, 20)
        xyz_samples = interpolate_along_track(ins.xyz, depths / 1e6) * 1e6
        xyz_samples = xyz_samples[np.argsort(xyz_samples[:, 2])]

        traj = Trajectory.fit(xyz_samples / 1e6)

        vector_perp = np.array([1, -1 * traj.vector[0] / traj.vector[2]])
        extent = 2000
        steps = np.ceil(extent * 2 / self.BRAIN_ATLAS.res_um).astype(int)
        image = np.zeros((xyz_samples.shape[0], steps))

        for i, xyz in enumerate(xyz_samples):
            origin = np.array([xyz[0], xyz[2]])
            anchor = np.r_[[origin + extent * vector_perp], [origin - extent * vector_perp]]
            xargmin = np.argmin(anchor[:, 0])
            xargmax = np.argmax(anchor[:, 0])
            perp_ml = np.linspace(anchor[xargmin, 0], anchor[xargmax, 0], steps)
            perp_ap = np.ones_like(perp_ml) * xyz[1]
            perp_dv = np.linspace(anchor[xargmin, 1], anchor[xargmax, 1], steps)

            idx = self.BRAIN_ATLAS.bc.xyz2i(np.c_[perp_ml, perp_ap, perp_dv] / 1e6, mode='clip')
            idx[idx[:, 2] >= self.BRAIN_ATLAS.image.shape[2], 2] = self.BRAIN_ATLAS.image.shape[2] - 1
            image[i, :] = self.BRAIN_ATLAS.image[idx[:, 1], idx[:, 0], idx[:, 2]]

        image = np.flipud(image)

        y_extent = [np.min(depths), np.max(depths)]
        ax.imshow(image, aspect='auto', extent=np.r_[[0, 4000], y_extent], cmap='bone', alpha=1, vmin=cmin, vmax=cmax)
        ax.scatter(np.ones_like(depths) * 20 + extent, depths, s=2, c='k')

        remove_frame(ax)

        return fig

    def plot_psychometric_curve(self, ax=None, ax_legend=None):

        if ax is None:
            fig, axs = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw={'width_ratios': [3, 1]})
            ax = axs[0]
            ax_legend = axs[1]
        else:
            fig = ax.get_figure()

        plot_psychometric(self.trials, ax=ax)
        set_axis_style(ax, xlabel='Contrasts', ylabel='Probability Choosing Right')

        ax.get_legend().remove()

        legend_elements = [Line2D([0], [0], color='w', lw=0, label='20 % of trials on left side'),
                           Line2D([0], [0], color='w', lw=0, label='equal % of trials on both sides'),
                           Line2D([0], [0], color='w', lw=0, label='80 % of trials on left side'),
                           Line2D([0], [0], color='w', marker='o', markerfacecolor='k', label='data', markersize=10),
                           Line2D([0], [0], color='k', lw=2, label='model fit')]

        leg = ax_legend.legend(handles=legend_elements, loc=4, frameon=False)
        remove_frame(ax_legend)
        cmap = sns.diverging_palette(20, 220, n=3, center="dark")

        for text, hand in zip(leg.get_texts(), leg.legend_handles):
            if '80 %' in text.get_text():
                text.set_color(cmap[2])
                text.set_position((-40, 0))
                hand.set_visible(False)
            elif '20 %' in text.get_text():
                text.set_color(cmap[0])
                text.set_position((-40, 0))
            elif 'equal %' in text.get_text():
                text.set_color(cmap[1])
                text.set_position((-40, 0))

        return fig

    def plot_chronometric_curve(self, ax=None, ax_legend=None):

        if ax is None:
            fig, axs = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw={'width_rations': [3, 1]})
            ax = axs[0]
            ax_legend = axs[1]
        else:
            fig = ax.get_figure()

        if np.all(self.trials['probabilityLeft'] == 0.5):
            plot_reaction_time_50(self.trials, ax=ax)
        else:
            plot_reaction_time(self.trials, ax=ax)
        set_axis_style(ax, xlabel='Contrasts', ylabel='Reaction time (s)')
        leg = ax.get_legend()
        h = leg.legend_handles
        l = [str(x._text) for x in leg.texts]
        ax.get_legend().remove()
        if ax_legend is not None:
            ax_legend.legend(handles=h, labels=l, frameon=False, loc=7)
            remove_frame(ax_legend)

        return fig

    def plot_reaction_time(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        else:
            fig = ax.get_figure()

        plot_reaction_time_over_trials(self.trials, ax=ax)
        set_axis_style(ax, xlabel='Trial number', ylabel='Reaction time (s)')

        return fig

    @staticmethod
    def plot_reaction_time_50(trials, ax=None, title=None, plot_ci=False, ci_alpha=0.32, **kwargs):
        """
        Function to plot reaction time against contrast a la datajoint webpage.

        The reaction times are plotted individually for the following three blocks: {0.5, 0.2, 0.8}.

        Parameters
        ----------
        trials : one.alf.io.AlfBunch
            An ALF trials object containing the keys {'probabilityLeft', 'contrastLeft',
            'contrastRight', 'feedbackType', 'choice', 'response_times', 'stimOn_times'}.
        ax : matplotlib.pyplot.Axes
            An axis object to plot onto.
        title : str
            An optional plot title.
        plot_ci : bool
            If true, computes and plots the confidence intervals for response at each contrast.
        ci_alpha : float, default=0.32
            Significance level for confidence interval. Must be in (0, 1). If `plot_ci` is false,
            this value is ignored.
        **kwargs
            If `ax` is None, these arguments are passed to matplotlib.pyplot.subplots.

        Returns
        -------
        matplotlib.pyplot.Figure
            The figure handle containing the plot.
        matplotlib.pyplot.Axes
            The plotted axes.

        See Also
        --------
        scipy.stats.bootstrap - the function used to compute the confidence interval.
        """

        signed_contrast = get_signed_contrast(trials)
        out_50 = compute_reaction_time(trials, signed_contrast=signed_contrast, block=0.5, compute_ci=plot_ci,
                                       alpha=ci_alpha)

        cmap = sns.diverging_palette(20, 220, n=3, center='dark')

        if not ax:
            fig, ax = plt.subplots(**kwargs)
        else:
            fig = plt.gcf()

        data_50 = ax.plot(out_50[1], out_50[0], '-o', color=cmap[1])

        if plot_ci:
            errbar_50 = np.c_[out_50[0] - out_50[3][:, 0], out_50[3][:, 1] - out_50[0]].T

            ax.errorbar(out_50[1], out_50[0], yerr=errbar_50, ecolor=cmap[1], fmt='none', capsize=5, alpha=0.4)

        ax.legend([data_50[0]],
                  ['p_left=0.5 data'],
                  loc='upper left')
        ax.set_ylabel('Reaction time (s)')
        ax.set_xlabel('Contrasts')

        if title:
            ax.set_title(title)

        return fig, ax

    def add_trial_events_to_raster(self, ax, trials, text=True):

        events = ['stimOn_times', 'firstMovement_times', 'feedback_times']
        colors = ['b', 'g', 'r']
        labels = ['Stim On', 'First Move', 'Feedback']
        trans = ax.get_xaxis_transform()

        for e, c, l in zip(events, colors, labels):
            if text:
                ax.text(trials[e], 1.01, l, c=c, rotation=45,
                        rotation_mode='anchor', ha='left', transform=trans)
            ax.vlines(trials[e], *ax.get_ylim(), color=c, zorder=ax.get_zorder() + 1)

        return ax

    def plot_dlc_feature_raster(self, camera, feature, axs=None, xlabel='T from Stim On (s)', ylabel0='Speed (px/s)',
                                ylabel1='Sorted Trial Number', title=None, zscore_flag=False, norm=False):

        camera = load_camera(self.eid, camera, data_path=self.data_path)
        feature = camera.computedFeatures[feature]

        if zscore_flag:
            feature = zscore(feature, nan_policy='omit')

        trial_idx, dividers = find_trial_ids(self.trials, sort='choice')
        fig, axs = self.single_cluster_raster(camera.times, self.trials['stimOn_times'], trial_idx, dividers, ['b', 'r'],
                                              ['correct', 'incorrect'], weights=feature, axs=axs, fr=False, norm=norm)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0, title=title)

        return fig

    def plot_dlc_feature_trace(self, camera, feature, trial_idx, ax=None, xlabel='T in session (s)', ylabel=None,
                               title=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        else:
            fig = ax.get_figure()

        trials = filter_trials_by_trial_idx(self.trials, trial_idx)

        camera = load_camera(self.eid, camera, data_path=self.data_path)
        idx = np.searchsorted(camera.times, self.trial_intervals[trial_idx])
        ax.plot(camera.times[idx[0]:idx[1]], camera.computedFeatures[feature][idx[0]:idx[1]], c='k')
        self.add_trial_events_to_raster(ax, trials, text=False)
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
        ax.spines['right'].set_visible(True)
        ax.set_ylabel(ylabel, rotation=0, fontsize=10, labelpad=40)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        return fig

    def plot_lick_raster(self, axs=None, xlabel='T from Feedback (s)', ylabel0='Licks (count)',
                         ylabel1='Sorted Trial Number', title=None):

        trial_idx, dividers = find_trial_ids(self.trials, sort='choice')
        fig, axs = self.single_cluster_raster(self.licks, self.trials['stimOn_times'], trial_idx, dividers, ['b', 'r'],
                                              ['correct', 'incorrect'], axs=axs, fr=False)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0, title=title)

        return fig

    def plot_wheel_raster(self, axs=None, xlabel='T from First Move (s)', ylabel0='Wheel velocity (rad/s)',
                          ylabel1='Sorted Trial Number', title=None):

        speed, _ = velocity_filtered(self.wheel.position, 1000)

        trial_idx, dividers = find_trial_ids(self.trials, sort='side')
        fig, axs = self.single_cluster_raster(
            self.wheel.timestamps, self.trials['firstMovement_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'],
            weights=speed, fr=False, axs=axs)

        set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
        set_axis_style(axs[0], ylabel=ylabel0, title=title)

        return fig

    def plot_wheel_trace(self, trial_idx, ax=None, xlabel='Time in trial (s)', ylabel='Wheel pos (rad)'):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        else:
            fig = ax.get_figure()

        trials = filter_trials_by_trial_idx(self.trials, trial_idx)

        wheel_pos, wheel_time = interpolate_position(self.wheel.timestamps, self.wheel.position)

        idx = np.searchsorted(wheel_time, self.trial_intervals[trial_idx])
        ax.plot(wheel_time[idx[0]:idx[1]], wheel_pos[idx[0]:idx[1]] - wheel_pos[idx[0]], c='k')

        max_rad = np.max(np.abs(wheel_pos[idx[0]:idx[1]] - wheel_pos[idx[0]]))
        ax.set_ylim([-1.1 * max_rad, 1.1 * max_rad])
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel)
        ax.spines['right'].set_visible(True)
        ax.set_ylabel(ylabel, rotation=0, fontsize=10, labelpad=40)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        self.add_trial_events_to_raster(ax, trials, text=False)

        return fig

    def plot_left_right_raster(self, signal, event, axs=None, xlabel='T from Feedback (s)',
                               ylabel0='Signal', ylabel1='Sorted Trial Number',
                               order='trial num'):

        trial_idx, dividers = find_trial_ids(self.trials, sort='side', order=order)
        colours = ['g', 'y']
        labels = ['left', 'right']

        if len(axs) == 1:
            fig, ax = self.processed_psth(self.psth[signal][event], trial_idx, dividers, colours, labels,
                                           ax=axs[0])
            set_axis_style(ax,  xlabel=xlabel, ylabel=ylabel0)
        else:
            fig, axs = self.processed_raster(self.psth[signal][event], trial_idx, dividers, colours, labels,
                                             axs=axs)

            set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
            set_axis_style(axs[0], ylabel=ylabel0)

        return fig

    def plot_correct_incorrect_raster(self, signal, event, axs=None, xlabel='T from Feedback (s)',
                                      ylabel0='Signal', ylabel1='Sorted Trial Number',
                                      order='trial num'):

        trial_idx, dividers = find_trial_ids(self.trials, sort='choice', order=order)
        colours = ['b', 'r']
        labels = ['correct', 'incorrect']

        if len(axs) == 1:
            fig, ax = self.processed_psth(self.psth[signal][event], trial_idx, dividers, colours, labels,
                                           ax=axs[0])
            set_axis_style(ax,  xlabel=xlabel, ylabel=ylabel0)
        else:
            fig, axs = self.processed_raster(self.psth[signal][event], trial_idx, dividers, colours, labels,
                                             axs=axs)

            set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
            set_axis_style(axs[0], ylabel=ylabel0)

        return fig

    def plot_block_raster(self, signal, event, axs=None, xlabel='T from Feedback (s)',
                                         ylabel0='Signal', ylabel1='Sorted Trial Number'):

        trial_idx = np.arange(len(self.trials['probabilityLeft']))
        dividers = np.where(np.diff(self.trials['probabilityLeft']) != 0)[0]

        blocks = self.trials['probabilityLeft'][np.r_[0, dividers + 1]]
        cmap = sns.diverging_palette(20, 220, n=3, center="dark")
        colours = np.full((blocks.shape[0], 3), np.array([*cmap[0]]))
        colours[np.where(blocks == 0.5)] = np.array([*cmap[1]])
        colours[np.where(blocks == 0.8)] = np.array([*cmap[2]])
        dividers = list(dividers)
        labels = blocks

        if len(axs) == 1:
            fig, ax = self.processed_psth(self.psth[signal][event], trial_idx, dividers, colours, labels,
                                           ax=axs[0])
            set_axis_style(ax,  xlabel=xlabel, ylabel=ylabel0)
        else:
            fig, axs = self.processed_raster(self.psth[signal][event], trial_idx, dividers, colours, labels,
                                             axs=axs)

            set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
            set_axis_style(axs[0], ylabel=ylabel0)

        return fig

    def plot_contrast_raster(self, signal, event, axs=None, xlabel='T from Feedback (s)',
                                            ylabel0='Signal', ylabel1='Sorted Trial Number'):

        contrasts = np.nanmean(np.c_[self.trials.contrastLeft, self.trials.contrastRight], axis=1)
        trial_idx = np.argsort(contrasts)
        dividers = list(np.where(np.diff(np.sort(contrasts)) != 0)[0])
        labels = [str(_ * 100) for _ in np.unique(contrasts)]
        colours = ['0.9', '0.7', '0.5', '0.3', '0.0']

        if len(axs) == 1:
            fig, ax = self.processed_psth(self.psth[signal][event], trial_idx, dividers, colours, labels,
                                           ax=axs[0])
            set_axis_style(ax,  xlabel=xlabel, ylabel=ylabel0)
        else:
            fig, axs = self.processed_raster(self.psth[signal][event], trial_idx, dividers, colours, labels,
                                             axs=axs)

            set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
            set_axis_style(axs[0], ylabel=ylabel0)

        return fig

    def processed_psth(self, psth, trial_idx, dividers, colors, labels, ax=None):

        t_psth = self.psth['times']

        dividers = [0] + dividers + [len(trial_idx)]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        else:
            fig = ax.get_figure()

        label, lidx = np.unique(labels, return_index=True)
        for lab, lid in zip(label, lidx):
            idx = np.where(np.array(labels) == lab)[0]
            for iD in range(len(idx)):
                if iD == 0:
                    t_ids = trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]
                    t_ints = dividers[idx[iD] + 1] - dividers[idx[iD]]
                else:
                    t_ids = np.r_[t_ids, trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]]
                    t_ints = np.r_[t_ints, dividers[idx[iD] + 1] - dividers[idx[iD]]]

            psth_div = np.nanmean(psth[t_ids], axis=0)

            ax.plot(t_psth, psth[t_ids].T, alpha=0.01, color=colors[lid])
            ax.plot(t_psth, psth_div, alpha=1, color=colors[lid], zorder=t_ids.size + 10)

        remove_spines(ax, spines=['right', 'top'])

        return fig, ax

    def processed_raster(self, raster, trial_idx, dividers, colors, labels, axs=None):

        t_raster = self.psth['times']
        dividers = [0] + dividers + [len(trial_idx)]
        # t_raster = np.arange(raster.shape[1]) - 30.5
        post_time = np.max(t_raster)
        pre_time = np.abs(np.min(t_raster))
        raster_bin = 0.5

        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [1, 3], 'hspace': 0},
                                    sharex=True)
        else:
            fig = axs[0].get_figure()

        label, lidx = np.unique(labels, return_index=True)
        label_pos = []
        for lab, lid in zip(label, lidx):
            idx = np.where(np.array(labels) == lab)[0]
            for iD in range(len(idx)):
                if iD == 0:
                    t_ids = trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]
                    t_ints = dividers[idx[iD] + 1] - dividers[idx[iD]]
                else:
                    t_ids = np.r_[t_ids, trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]]
                    t_ints = np.r_[t_ints, dividers[idx[iD] + 1] - dividers[idx[iD]]]

            psth_div = np.nanmean(raster[t_ids], axis=0)

            axs[0].plot(t_raster, raster[t_ids].T, alpha=0.01, color=colors[lid])
            axs[0].plot(t_raster, psth_div, alpha=1, color=colors[lid], zorder=t_ids.size + 10)

            lab_max = idx[np.argmax(t_ints)]
            label_pos.append((dividers[lab_max + 1] - dividers[lab_max]) / 2 + dividers[lab_max])

        axs[1].imshow(raster[trial_idx], cmap='binary', origin='lower',
                      extent=[np.min(t_raster), np.max(t_raster), 0, len(trial_idx)], aspect='auto')

        width = raster_bin * 4
        for iD in range(len(dividers) - 1):
            axs[1].fill_between([post_time + raster_bin / 2, post_time + raster_bin / 2 + width],
                                [dividers[iD + 1], dividers[iD + 1]], [dividers[iD], dividers[iD]], color=colors[iD])

        axs[1].set_xlim([-1 * pre_time, post_time + raster_bin / 2 + width])
        secax = axs[1].secondary_yaxis('right')

        secax.set_yticks(label_pos)
        secax.set_yticklabels(label, rotation=90,
                              rotation_mode='anchor', ha='center')
        for ic, c in enumerate(np.array(colors)[lidx]):
            secax.get_yticklabels()[ic].set_color(c)

        remove_spines(axs[1], spines=['right', 'top'])
        remove_spines(axs[0], spines=['right', 'top'])

        # This is set in the generator when adjusting the axis
        # axs[0].vlines(0, *axs[0].get_ylim(), color='k', ls='--', zorder=axs[0].get_zorder() + 1)
        axs[1].vlines(0, *axs[1].get_ylim(), color='k', ls='--', zorder=axs[1].get_zorder() + 1)

        return fig, axs

    def single_cluster_raster(self, spike_times, events, trial_idx, dividers, colors, labels, weights=None, fr=True, norm=False,
                              axs=None):

        pre_time = 0.4
        post_time = 1
        raster_bin = 0.01
        psth_bin = 0.05
        raster, t_raster = bin_spikes(
            spike_times, events, pre_time=pre_time, post_time=post_time, bin_size=raster_bin, weights=weights)
        psth, t_psth = bin_spikes(
            spike_times, events, pre_time=pre_time, post_time=post_time, bin_size=psth_bin, weights=weights)

        if fr:
            psth = psth / psth_bin

        if norm:
            psth = psth - np.repeat(psth[:, 0][:, np.newaxis], psth.shape[1], axis=1)
            raster = raster - np.repeat(raster[:, 0][:, np.newaxis], raster.shape[1], axis=1)

        dividers = [0] + dividers + [len(trial_idx)]
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [1, 3], 'hspace': 0},
                                    sharex=True)
        else:
            fig = axs[0].get_figure()

        label, lidx = np.unique(labels, return_index=True)
        label_pos = []
        for lab, lid in zip(label, lidx):
            idx = np.where(np.array(labels) == lab)[0]
            for iD in range(len(idx)):
                if iD == 0:
                    t_ids = trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]
                    t_ints = dividers[idx[iD] + 1] - dividers[idx[iD]]
                else:
                    t_ids = np.r_[t_ids, trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]]
                    t_ints = np.r_[t_ints, dividers[idx[iD] + 1] - dividers[idx[iD]]]

            psth_div = np.nanmean(psth[t_ids], axis=0)
            std_div = np.nanstd(psth[t_ids], axis=0) / np.sqrt(len(t_ids))

            axs[0].fill_between(t_psth, psth_div - std_div,
                                psth_div + std_div, alpha=0.4, color=colors[lid])
            axs[0].plot(t_psth, psth_div, alpha=1, color=colors[lid])

            lab_max = idx[np.argmax(t_ints)]
            label_pos.append((dividers[lab_max + 1] - dividers[lab_max]) / 2 + dividers[lab_max])

        axs[1].imshow(raster[trial_idx], cmap='binary', origin='lower',
                      extent=[np.min(t_raster), np.max(t_raster), 0, len(trial_idx)], aspect='auto')

        width = raster_bin * 4
        for iD in range(len(dividers) - 1):
            axs[1].fill_between([post_time + raster_bin / 2, post_time + raster_bin / 2 + width],
                                [dividers[iD + 1], dividers[iD + 1]], [dividers[iD], dividers[iD]], color=colors[iD])

        axs[1].set_xlim([-1 * pre_time, post_time + raster_bin / 2 + width])
        secax = axs[1].secondary_yaxis('right')

        secax.set_yticks(label_pos)
        secax.set_yticklabels(label, rotation=90,
                              rotation_mode='anchor', ha='center')
        for ic, c in enumerate(np.array(colors)[lidx]):
            secax.get_yticklabels()[ic].set_color(c)

        remove_spines(axs[1], spines=['right', 'top'])
        remove_spines(axs[0], spines=['right', 'top'])

        axs[0].vlines(0, *axs[0].get_ylim(), color='k', ls='--', zorder=axs[0].get_zorder() + 1)
        axs[1].vlines(0, *axs[1].get_ylim(), color='k', ls='--', zorder=axs[1].get_zorder() + 1)

        return fig, axs








    # dl = DataLoader()
    # dl.session_init('decc8d40-cf74-4263-ae9d-a0cc68b47e86')
    # dl.get_session_details()

    # pid = list(DATA_DIR.iterdir())[0]

    # clusters = load_clusters(pid)
    # channels = load_channels(pid)
    # spikes = load_spikes(pid)
    # trials = load_trials(pid)
    # # cluster_waveforms = load_cluster_waveforms(pid)

    # cluster_idx = clusters.cluster_id[10]
    # trial_idx = len(trials) // 2

    # # fig = plot_session_raster(spikes, trials, cluster_idx, trial_idx)

    # plot_spikes_fr_vs_depth(clusters, cluster_idx)

    # plt.show()
