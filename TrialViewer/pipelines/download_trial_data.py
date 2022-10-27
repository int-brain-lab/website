from ibllib.io.video import get_video_meta, get_video_frame
from brainbox.behavior.dlc import plot_trace_on_frame
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')

from ibllib.qc.camera import CameraQC

import numpy as np
import pandas as pd
from pathlib import Path
import copy
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

import os
from os.path import exists

data_path = Path(r'D:\ibl-website-videos\proc')
out_path = Path(r'./data')
PAD_S = 0.5

new_fs = 24
new_width = 160
new_height = 128

import pickle
with open("selectable.pids", "rb") as fp:   # Unpickling
  selectable_pids = pickle.load(fp)

for pid in selectable_pids:
  # skip sessions that were already run
  if exists(f'{out_path}/{pid}/{pid}_trials.csv'):
    continue

  eid, probe = one.pid2eid(pid) #'0802ced5-33a3-405e-8336-b65ebc5cb07c'

  print((pid,eid))

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

  # remove any rows with NaNs
  nan_idx = np.bitwise_or(np.isnan(trials_data['stimOn_times']), np.isnan(trials_data['feedback_times']))
  trials = trials.loc[~nan_idx]

  if not exists(f'{out_path}/{pid}'):
    os.makedirs(f'{out_path}/{pid}')

  trials.to_csv(f'{out_path}/{pid}/{pid}_trials.csv', index=False)

  # calculate the start and end times 
  start = trials.stim_on_timestamp.values[0] - PAD_S
  end = trials.feedback_timestamp.values[-1] + PAD_S

  # TODO: deal with sessions where you need to drop trials so that the start timestamp comes before the first trial

  # calculate the # of frames from start to end at 24fps
  frames = (end - start) * new_fs
  
  # to use the start and end -- these will be the first and last timepoints in the 
  # videos once they are trimmed. So frame 0 starts at start, and then goes forward at 24 fps from that point

  np.save(f'{out_path}/{pid}/{pid}_start_end.npy',[start,end, frames])

