from ibllib.io.video import get_video_meta, get_video_frame
from brainbox.behavior.dlc import plot_trace_on_frame

from ibllib.qc.camera import CameraQC
import subprocess

import numpy as np
import pandas as pd
from pathlib import Path

import os
from os.path import exists

PROC_DIR = 'D:/ibl-website-videos/proc'
DATA_DIR = './data'

data_path = Path(r'D:\ibl-website-videos\proc')
out_path = Path(r'./data')

import pickle
with open("selectable.pids", "rb") as fp:   # Unpickling
  selectable_pids = pickle.load(fp)


for pid in selectable_pids:
  # eid, probe = one.pid2eid(pid) #'0802ced5-33a3-405e-8336-b65ebc5cb07c'

  ftext = pid + '_left'

  # load crop metadata
  crop_data = pd.read_csv(f'{DATA_DIR}/{pid}/{pid}_left_pupil_rect.csv')

  x0 = crop_data.x0[0]
  y0 = crop_data.y0[0]
  w = 160
  h = 128

  print((x0, y0, w, h))

  times = np.load(f'{PROC_DIR}/{pid}_left_times.npy')

  framerate = 1/np.mean(np.diff(times))


  # crop 
  cropFile = f'{PROC_DIR}/{pid}_left_crop.mp4'
  if not exists(cropFile):
    call = subprocess.call(['ffmpeg',
                  '-r', f'{framerate}',
                  '-i', 'D:\\ibl-website-videos\\raw\\'+ftext+'.mp4',
                  '-vf', f'crop={w}:{h}:{x0}:{y0},scale=160:128', 
                  '-r', '24',
                  cropFile])