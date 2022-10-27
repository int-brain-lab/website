import numpy as np
import pandas as pd
import requests
import subprocess
import time
from os.path import exists
import datetime
 

PROC_FOLDER = 'D:/ibl-website-videos/proc'
FINAL_FOLDER = 'D:/ibl-website-videos/final'
DATA_FOLDER = './data'

def ffmpeg_time_format(time_s):
  date = datetime.timedelta(seconds = time_s)
  return str(date)

import pickle
with open("selectable.pids", "rb") as fp:   # Unpickling
  selectable_pids = pickle.load(fp)

for pid in selectable_pids:
  print(pid)
  left_ts = np.load(f'D:/ibl-website-videos/proc/{pid}_left_times.npy')
  right_ts = np.load(f'D:/ibl-website-videos/proc/{pid}_right_times.npy')
  body_ts = np.load(f'D:/ibl-website-videos/proc/{pid}_body_times.npy')
  # Load the first trial time and last trial time

  # load the start/end points
  start_end = np.load(f'{DATA_FOLDER}/{pid}/{pid}_start_end.npy')
  start = start_end[0]
  end = start_end[1]

  print((start, end))

  
  # trim all videos to match the timestamps above
  videos = ['left_scaled','right_scaled','body_scaled', 'left_crop']
  timestamps = [left_ts, right_ts, body_ts, left_ts]

  length_t = ffmpeg_time_format(end-start)
  inputs = []

  for i, video in enumerate(videos):
    ts = timestamps[i]

    input = f'{PROC_FOLDER}/{pid}_{video}.mp4'
    out = f'{PROC_FOLDER}/{pid}_{video}_trim.mp4'
    inputs.append(out)
    
    # check if start is actually before ts[0], that would be bad
    if (ts[0] > start):
      print(f'First timestamp {ts[0]} is after video start {start} -- something went wrong?')
      continue
    
    start_t = ffmpeg_time_format(start - ts[0])
    print((start_t, length_t))
    if not exists(out):
      call = subprocess.call(['ffmpeg',
                            '-i', input,
                            '-ss', start_t,
                            '-t', length_t,
                            out])
  
  # now do concatenations
  full_out = f'{FINAL_FOLDER}/{pid}.mp4'
  if not exists(full_out):
    call = subprocess.call(['ffmpeg',
    '-i', inputs[0], '-i', inputs[1], '-i', inputs[2], '-i', inputs[3],
    '-filter_complex', '[0:v][1:v][2:v][3:v]hstack=inputs=4[v]',
    '-map', '[v]',
    full_out])