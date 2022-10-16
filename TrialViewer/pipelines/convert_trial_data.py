# Load the trial data files and convert them to just the columns that we want
import numpy as np
import pandas as pd
import os
from os.path import exists


DATA_DIR = './data'
OUT_DIR = './final'

import pickle
with open("selectable.pids", "rb") as fp:   # Unpickling
  selectable_pids = pickle.load(fp)

for pid in selectable_pids:
  trials = pd.read_csv(f'{DATA_DIR}/{pid}/{pid}_trials.csv')

  if not exists(f'{OUT_DIR}/{pid}'):
    os.makedirs(f'{OUT_DIR}/{pid}')

  # load the start/end points
  start_end = np.load(f'{DATA_DIR}/{pid}/{pid}_start_end.npy')
  start = start_end[0]
  end = start_end[1]

  left_ts = np.arange(0,end-start,1/24) + start

  # 2 re-index the trial events
  for i, row in trials.iterrows():
    # replace the timestamps with the index
    start_idx = np.argmin(np.abs(row['start_timestamp']- left_ts))
    stim_on_idx = np.argmin(np.abs(row['stim_on_timestamp'] - left_ts))
    feedback_idx = np.argmin(np.abs(row['feedback_timestamp']- left_ts))
    trials.loc[i,'start_timestamp'] = start_idx
    trials.loc[i,'stim_on_timestamp'] = stim_on_idx
    trials.loc[i,'feedback_timestamp'] = feedback_idx

  trials.to_csv(f'{OUT_DIR}/{pid}/{pid}.trials.csv', index = False)