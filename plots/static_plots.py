# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import copy

from brainbox.task.trials import find_trial_ids
from brainbox.population.decode import xcorr
from iblutil.util import Bunch

import one.alf.io as alfio


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'data'


# -------------------------------------------------------------------------------------------------
# Loading functions
# -------------------------------------------------------------------------------------------------

def load_clusters(pid):
    clusters = alfio.load_object(DATA_PATH.joinpath(pid), object='clusters')
    return clusters


def load_channels(pid):
    channels = alfio.load_object(DATA_PATH.joinpath(pid), object='channels')
    return channels


def load_spikes(pid):
    spikes = alfio.load_object(DATA_PATH.joinpath(pid), object='spikes')
    return spikes


def load_trials(pid):
    trials = alfio.load_object(DATA_PATH.joinpath(pid), object='trials')
    return trials


def load_cluster_waveforms(pid):
    wfs = np.load(DATA_PATH.joinpath(pid, 'clusters.waveforms.npy'))
    wf_chns = np.load(DATA_PATH.joinpath(
        pid, 'clusters.waveformsChannels.npy'))

    return wfs, wf_chns


# -------------------------------------------------------------------------------------------------
# Filtering
# -------------------------------------------------------------------------------------------------

def _filter(obj, idx):
    obj = Bunch(copy.deepcopy(obj))
    for key in obj.keys():
        obj[key] = obj[key][idx]

    return obj


def filter_spikes_by_good_clusters(spikes):
    return _filter(spikes, spikes.good)


def filter_clusters_by_good_clusters(clusters):
    idx = np.where(clusters.label == 1)[0]
    return _filter(clusters, idx)


def filter_spikes_by_cluster_idx(spikes, cluster_idx):
    idx = np.where(spikes.clusters == cluster_idx)[0]

    return _filter(spikes, idx)


def filter_spikes_by_trial(spikes, tstart, tend):
    idx = np.where(np.bitwise_and(
        spikes.times >= tstart, spikes.times <= tend))[0]
    return _filter(spikes, idx)


def filter_spikes_by_sample(spikes, sample):
    for key in spikes.keys():
        spikes[key] = spikes[key][::sample]

    return spikes


def filter_trials_by_trial_idx(trials, trial_idx):
    return _filter(trials, trial_idx)


def filter_clusters_by_cluster_idx(clusters, cluster_idx):
    idx = np.where(clusters.cluster_id == cluster_idx)[0][0]
    return _filter(clusters, idx)


def filter_wfs_by_cluster_idx(waveforms, waveform_channels, clusters, cluster_idx):
    idx = np.where(clusters.cluster_id == cluster_idx)[0][0]
    return waveforms[idx], waveform_channels[idx]


# -------------------------------------------------------------------------------------------------
# Processing functions
# -------------------------------------------------------------------------------------------------

def bin_spikes(spike_times, align_times, pre_time, post_time, bin_size, weights=None):

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
        bins[i, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale


# -------------------------------------------------------------------------------------------------
# Styling functions
# -------------------------------------------------------------------------------------------------

def set_figure_style(fig, margin_inches=0.8):
    x_inches, y_inches = fig.figure.get_size_inches()
    fig.subplots_adjust(margin_inches / x_inches, margin_inches / y_inches)
    return fig


def set_axis_style(ax, **kwargs):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(kwargs.get('xlabel', None), fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', None), fontsize=12)
    ax.set_title(kwargs.get('title', None), fontsize=12)

    return ax


# -------------------------------------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------------------------------------

def plot_session_raster(spikes, trials, cluster_idx, trial_idx, subsample=100):

    alpha = spikes.sizes / np.max(spikes.sizes)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    set_figure_style(fig)
    ax.set_xlim(0, np.max(spikes.times))
    ax.set_ylim(0, 3840)
    ax.scatter(spikes.times[::subsample], spikes.depths[::subsample], s=spikes.sizes[::subsample], alpha=alpha[::subsample],
               c='grey')

    # TODO Allen colours
    spikes = filter_spikes_by_cluster_idx(spikes, cluster_idx)
    ax.scatter(spikes.times, spikes.depths, s=spikes.sizes,
               facecolors='none', edgecolors='r')
    ax = set_axis_style(ax, xlabel='Time (s)', ylabel='Depth (um)')

    trials = filter_trials_by_trial_idx(trials, trial_idx)
    ax.axvline(trials['goCue_times'], *ax.get_ylim(), c='k', ls='--')

    return fig


def plot_trial_raster(spikes, trials, cluster_idx, trial_idx):

    trials = filter_trials_by_trial_idx(trials, trial_idx)
    spikes = filter_spikes_by_trial(
        spikes, trials['stimOn_times'] - 2, trials['feedback_times'] + 2)
    alpha = spikes.sizes / np.max(spikes.sizes)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    set_figure_style(fig)
    ax.scatter(spikes.times, spikes.depths,
               s=spikes.sizes, alpha=alpha, c='grey')
    ax.set_ylim(0, 3840)
    ax.set_xlim(np.min(spikes.times), np.max(spikes.times))

    spikes = filter_spikes_by_cluster_idx(spikes, cluster_idx)
    ax.scatter(spikes.times, spikes.depths, s=spikes.sizes, c='r')
    ax = add_trial_events_to_raster(ax, trials)
    ax = set_axis_style(ax, xlabel='Time (s)', ylabel='Depth (um)')

    return ax


def add_trial_events_to_raster(ax, trials):

    events = ['goCue_times', 'firstMovement_times', 'feedback_times']
    colors = ['b', 'g', 'r']
    labels = ['Go Cue', 'First Move', 'Feedback']
    trans = ax.get_xaxis_transform()

    for e, c, l in zip(events, colors, labels):
        ax.axvline(trials[e], *ax.get_ylim(), c=c)
        ax.text(trials[e], 1.01, l, c=c, rotation=45,
                rotation_mode='anchor', ha='left', transform=trans)

    return ax


def plot_spikes_amp_vs_depth(clusters, cluster_idx):

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    set_figure_style(fig)
    ax.set_ylim(0, 3840)
    ax.set_xlim(-5, 500)
    ax.scatter(clusters.amps * 1e6, clusters.depths,
               facecolors='none', edgecolors='grey')
    clusters = filter_clusters_by_cluster_idx(clusters, cluster_idx)
    ax.scatter(clusters.amps * 1e6, clusters.depths, c='r')
    set_axis_style(ax, xlabel='Amplitude (uV)', ylabel='Depth (um)')
    return fig


def plot_spikes_fr_vs_depth(clusters, cluster_idx):

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    set_figure_style(fig)
    ax.set_ylim(0, 3840)
    ax.set_xlim(-2, 30)
    ax.scatter(clusters.firing_rate, clusters.depths,
               facecolors='none', edgecolors='grey')
    clusters = filter_clusters_by_cluster_idx(clusters, cluster_idx)
    ax.scatter(clusters.firing_rate, clusters.depths, c='r')
    set_axis_style(ax, xlabel='Firing Rate (Hz)', ylabel='Depth (um)')
    return fig


def plot_left_right_single_cluster_raster(spikes, trials, cluster_idx):

    spikes = filter_spikes_by_cluster_idx(spikes, cluster_idx)
    trial_idx, dividers = find_trial_ids(trials, sort='side')
    fig, ax = single_cluster_raster(
        spikes.times, trials['stimOn_times'], trial_idx, dividers, ['g', 'y'], ['left', 'right'])
    set_figure_style(fig)
    ax[0].set_ylim(0, 20)  # TODO this shouldn't be harcoded here
    ax[1].set_yticklabels([])
    set_axis_style(ax[1], xlabel='Time from Stim On (s)',
                   ylabel='Sorted Trial Number')
    set_axis_style(ax[0], ylabel='Firing Rate (Hz)')

    return fig


def plot_correct_incorrect_single_cluster_raster(spikes, trials, cluster_idx):

    spikes = filter_spikes_by_cluster_idx(spikes, cluster_idx)
    trial_idx, dividers = find_trial_ids(trials, sort='choice')
    fig, ax = single_cluster_raster(spikes.times, trials['feedback_times'], trial_idx, dividers, ['b', 'r'],
                                    ['correct', 'incorrect'])
    set_figure_style(fig)
    ax[0].set_ylim(0, 20)  # TODO this shouldn't be harcoded here
    ax[1].set_yticklabels([])
    set_axis_style(ax[1], xlabel='Time from Feedback (s)',
                   ylabel='Sorted Trial Number')
    set_axis_style(ax[0], ylabel='Firing Rate (Hz)')


def single_cluster_raster(spike_times, events, trial_idx, dividers, colors, labels):

    pre_time = 0.4
    post_time = 1
    raster_bin = 0.01
    psth_bin = 0.05
    raster, t_raster = bin_spikes(
        spike_times, events, pre_time=pre_time, post_time=post_time, bin_size=raster_bin)
    psth, t_psth = bin_spikes(
        spike_times, events, pre_time=pre_time, post_time=post_time, bin_size=psth_bin)

    dividers = [0] + dividers + [len(trial_idx)]
    fig, ax = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={
                           'height_ratios': [1, 3], 'hspace': 0}, sharex=True)
    for iD in range(len(dividers) - 1):
        psth_div = np.nanmean(
            psth[trial_idx[dividers[iD]:dividers[iD + 1]]], axis=0) / psth_bin
        std_div = (np.nanstd(psth[trial_idx[dividers[iD]:dividers[iD + 1]]], axis=0) / psth_bin) \
            / np.sqrt(dividers[iD + 1] - dividers[iD])

        ax[0].fill_between(t_psth, psth_div - std_div,
                           psth_div + std_div, alpha=0.4, color=colors[iD])
        ax[0].plot(t_psth, psth_div, alpha=1, color=colors[iD])

    ax[1].imshow(raster[trial_idx], cmap='binary', origin='lower',
                 extent=[np.min(t_raster), np.max(t_raster), 0, len(trial_idx)], aspect='auto')

    width = raster_bin * 4
    label_pos = []
    for iD in range(len(dividers) - 1):
        ax[1].fill_between([post_time + raster_bin / 2, post_time + raster_bin / 2 + width],
                           [dividers[iD + 1], dividers[iD + 1]], [dividers[iD], dividers[iD]], color=colors[iD])
        label_pos.append((dividers[iD + 1] - dividers[iD]) / 2 + dividers[iD])

    secax = ax[1].secondary_yaxis('right')

    secax.set_yticks(label_pos)
    secax.set_yticklabels(labels, rotation=90,
                          rotation_mode='anchor', ha='center')
    ax[1].set_xlim([-1 * pre_time, post_time + raster_bin / 2 + width])
    for ic, c in enumerate(colors):
        secax.get_yticklabels()[ic].set_color(c)

    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[0].axvline(0, *ax[0].get_ylim(), c='k', ls='--')
    ax[1].axvline(0, *ax[1].get_ylim(), c='k', ls='--')

    return fig, ax


def plot_cluster_waveforms(waveforms, waveform_channels, channels, clusters, cluster_idx):

    wfs, wf_chns = filter_wfs_by_cluster_idx(
        waveforms, waveform_channels, clusters, cluster_idx)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    set_figure_style(fig)
    wf_chn_pos = np.c_[channels['localCoordinates'][:, 0]
                       [wf_chns], channels['localCoordinates'][:, 1][wf_chns]]

    wf_chn_pos[:, 0] = wf_chn_pos[:, 0] - np.min(wf_chn_pos[:, 0])
    wf_chn_pos[:, 1] = wf_chn_pos[:, 1] - np.min(wf_chn_pos[:, 1])

    # TODO some kind of scale bar

    for ichn, chn in enumerate(wf_chn_pos):
        x = (np.arange(wfs.shape[0]) / 5 + chn[0])
        y = (wfs[:, ichn])
        y = (y * 1e6 + chn[1] * 10)
        ax.plot(x, y, c='grey')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    set_axis_style(ax)

    return fig


def plot_autocorrelogram(spikes, cluster_idx):

    spikes = filter_spikes_by_cluster_idx(spikes, cluster_idx)

    x_corr = xcorr(spikes.times, spikes.clusters, 1 / 1e3, 50 / 1e3)
    corr = x_corr[0, 0, :]
    corr = corr / np.max(corr)  # normalise

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    set_figure_style(fig)
    ax.bar(np.arange(corr.size), height=corr, width=0.8, color='grey')
    set_axis_style(ax, xlabel='T (ms)', ylabel='Autocorr')

    return fig


# -------------------------------------------------------------------------------------------------
# Interactivity functions
# -------------------------------------------------------------------------------------------------

def on_pid_change(pid, cluster_idx, trial_idx):
    spikes = load_spikes(pid)
    clusters = load_clusters(pid)
    spikes = filter_spikes_by_good_clusters(spikes)
    clusters = filter_clusters_by_good_clusters(clusters)
    trials = load_trials(pid)
    waveforms, waveform_channels = load_cluster_waveforms(pid)
    channels = load_channels(pid)

    figs = []
    figs.append(plot_session_raster(spikes, trials, cluster_idx, trial_idx))
    figs.append(plot_trial_raster(spikes, trials, cluster_idx, trial_idx))
    # fig = plot_brain_areas() # TODO
    # fig = plot_lfp_spectrum() # TODO
    figs.append(plot_spikes_amp_vs_depth(clusters, cluster_idx))
    figs.append(plot_spikes_fr_vs_depth(clusters, cluster_idx))
    figs.append(plot_left_right_single_cluster_raster(
        spikes, trials, cluster_idx))
    figs.append(plot_correct_incorrect_single_cluster_raster(
        spikes, trials, cluster_idx))
    # fig = plot_contrast_single_cluster_raster() # TODO
    figs.append(plot_cluster_waveforms(
        waveforms, waveform_channels, channels, clusters, cluster_idx))
    figs.append(plot_autocorrelogram(spikes, cluster_idx))
    # fig = plot_interspike_intreval() # TODO

    return figs


def on_trial_change(pid, cluster_idx, trial_idx):
    spikes = load_spikes(pid)
    spikes = filter_spikes_by_good_clusters(spikes)
    trials = load_trials(pid)

    figs = []
    figs.append(plot_session_raster(spikes, trials, cluster_idx, trial_idx))
    figs.append(plot_trial_raster(spikes, trials, cluster_idx, trial_idx))


def on_cluster_change(pid, cluster_idx, trial_idx):
    spikes = load_spikes(pid)
    clusters = load_clusters(pid)
    spikes = filter_spikes_by_good_clusters(spikes)
    clusters = filter_clusters_by_good_clusters(clusters)
    trials = load_trials(pid)
    waveforms, waveform_channels = load_cluster_waveforms(pid)
    channels = load_channels(pid)

    figs = []
    figs.append(plot_session_raster(spikes, trials, cluster_idx, trial_idx))
    figs.append(plot_trial_raster(spikes, trials, cluster_idx, trial_idx))
    figs.append(plot_spikes_amp_vs_depth(clusters, cluster_idx))
    figs.append(plot_spikes_fr_vs_depth(clusters, cluster_idx))
    figs.append(plot_left_right_single_cluster_raster(
        spikes, trials, cluster_idx))
    figs.append(plot_correct_incorrect_single_cluster_raster(
        spikes, trials, cluster_idx))
    # fig = plot_contrast_single_cluster_raster() # TODO
    figs.append(plot_cluster_waveforms(
        waveforms, waveform_channels, channels, clusters, cluster_idx))
    figs.append(plot_autocorrelogram(spikes, cluster_idx))
    # fig = plot_interspike_intreval() # TODO

    return figs


def get_cluster_choices(pid):
    """
    Helper function to figure out which clusters we can choose
    :param pid:
    :return:
    """
    clusters = load_clusters(pid)
    clusters = filter_clusters_by_good_clusters(clusters)
    return clusters.cluster_id


if __name__ == '__main__':

    pid = list(DATA_PATH.iterdir())[0]

    clusters = load_clusters(pid)
    channels = load_channels(pid)
    spikes = load_spikes(pid)
    trials = load_trials(pid)
    # cluster_waveforms = load_cluster_waveforms(pid)

    cluster_idx = clusters.cluster_id[10]
    trial_idx = len(trials) // 2

    # fig = plot_session_raster(spikes, trials, cluster_idx, trial_idx)

    plot_spikes_fr_vs_depth(clusters, cluster_idx)

    plt.show()
