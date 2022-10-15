# Load the trial data files and convert them to just the columns that we want
import numpy as np
import pandas as pd
from one.api import ONE

one = ONE()

DATA_DIR = './data'

# load CSV file
session_table = pd.read_csv('./session.table.csv')
selectable_pids = []
for i,row in session_table.iterrows():
  if row['selectable']:
    selectable_pids.append(row['pid'])


for pid in selectable_pids:
  left_ts = np.load(f'{DATA_DIR}/{pid}/{pid}_left_times_scaled.npy')
  trials = pd.read_csv(f'{DATA_DIR}/{pid}/{pid}_trials.csv',index_col=0)

  # 2 re-index the trial events
  for i, row in trials.iterrows():
    # replace the timestamps with the index
    start_idx = np.argmin(np.abs(row['start_timestamp']-left_ts))
    stim_on_idx = np.argmin(np.abs(row['stim_on_timestamp']-left_ts))
    feedback_idx = np.argmin(np.abs(row['feedback_timestamp']-left_ts))
    trials.loc[i,'start_timestamp'] = start_idx
    trials.loc[i,'stim_on_timestamp'] = stim_on_idx
    trials.loc[i,'feedback_timestamp'] = feedback_idx

  trials.to_csv(f'{DATA_DIR}/{pid}/{pid}.trials.csv', index = False)