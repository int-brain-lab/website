from one.api import ONE
from ibllib.io.video import get_video_meta, get_video_frame
from brainbox.behavior.dlc import plot_trace_on_frame

import numpy as np
import pandas as pd
from pathlib import Path
import copy
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

one = ONE()
data_path = Path(r'C:\proj\int-brain-lab\ibl-website\TrialViewer\Assets\AddressableAssets\0802ced5-33a3-405e-8336-b65ebc5cb07c')

eid = '0802ced5-33a3-405e-8336-b65ebc5cb07c'

video_params = {
    'left': {
        'fps': 60,
        'width': 1280,
        'height': 1024
    },
    'right': {
        'fps': 150,
        'width': 640,
        'height': 512
    },
    'body': {
        'fps': 30,
        'width': 640,
        'height': 512
    }
}

new_fs = 30
new_width = 64
new_height = 50


for label in ['left', 'right', 'body']:

    video_path = data_path.joinpath(f'{eid}_{label}_scaled.mp4')
    video_meta = get_video_meta(video_path)


    fs_subsamp_factor = int(video_params[label]['fps'] / new_fs)
    w_subsamp_factor = video_params[label]['width'] / new_width
    h_subsamp_factor = video_params[label]['height'] / new_height

    video_data = one.load_object(eid, f'{label}Camera', collection='alf')
    subsamp_times = video_data.times[::fs_subsamp_factor]
    # Make sure the times and video are same size. (This doesn't hold for right video! Mismatch by one frame)
    # assert subsamp_times.size == video_meta['length']

    dlc = pd.DataFrame()
    dlc_positions = copy.copy(video_data.dlc)

    for col in dlc_positions.keys():
        if 'likelihood' in col:
            dlc[col] = dlc_positions[col].values[::fs_subsamp_factor]
        elif '_x' in col:
            dlc[col] = dlc_positions[col].values[::fs_subsamp_factor] / w_subsamp_factor
        elif '_y' in col:
            dlc[col] = dlc_positions[col].values[::fs_subsamp_factor] / h_subsamp_factor

    # check that the positions make sense
    plot_trace_on_frame(get_video_frame(video_path, 200), dlc, label)

    # interpolate the wheel data at subsampled timestamps
    wheel_data = one.load_object(eid, 'wheel', collection='alf')
    wheel_pos = interpolate.interp1d(wheel_data.timestamps, wheel_data.position, kind='linear')(subsamp_times)

    # check the interpolation makes sense
    fig, ax = plt.subplots()
    ax.plot(wheel_data.timestamps, wheel_data.position, 'x')
    ax.plot(subsamp_times, wheel_pos, 'o')

    # Save the files
    np.save(data_path.joinpath(f'{eid}_{label}_times_scaled.npy'), subsamp_times)
    dlc.to_csv(data_path.joinpath(f'{eid}_{label}_dlc_scaled.csv'))
    # dlc.to_parquet(data_path.joinpath(f'{eid}_{label}_dlc_scaled.pqt'))
    np.save(data_path.joinpath(f'{eid}_{label}_wheel_scaled.npy'), wheel_pos)


# Save the trial data
trials_data = one.load_object(eid, 'trials')
trials = pd.DataFrame()
trials['start_timestamp'] = trials_data.intervals[:, 0]
trials['stim_on_timestamp'] = trials_data.stimOn_times
trials['feedback_timestamp'] = trials_data.feedback_times

right_idx = np.where(~np.isnan(trials_data.contrastRight))[0]
stim_side = np.full(trials_data.contrastLeft.shape, 'L')
stim_side[right_idx] = 'R'
trials['stimulus_side'] = stim_side
trials['contrast'] = np.nansum(np.c_[trials_data.contrastLeft, trials_data.contrastRight], axis=1)
trials['feedback'] = trials_data.feedbackType  # correct = 1, incorrect = -1

trials.to_csv(data_path.joinpath(f'{eid}_trials.csv'))


