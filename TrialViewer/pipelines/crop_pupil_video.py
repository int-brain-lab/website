from one.api import ONE
from ibllib.io.video import get_video_meta, get_video_frame
from brainbox.behavior.dlc import plot_trace_on_frame

from ibllib.qc.camera import CameraQC
import subprocess

import pandas as pd
from pathlib import Path

import os
from os.path import exists

one = ONE()
data_path = Path(r'D:\ibl-website-videos\proc')
out_path = Path(r'./data')

# load CSV file
session_table = pd.read_csv('./session.table.csv')
selectable_pids = []
for i,row in session_table.iterrows():
    if row['selectable']:
        selectable_pids.append(row['pid'])

for pid in selectable_pids:
  # eid, probe = one.pid2eid(pid) #'0802ced5-33a3-405e-8336-b65ebc5cb07c'

  ftext = pid + '_left'

  # load crop metadata
  crop_data = pd.read_csv(os.path.join(out_path, pid, ftext + '_pupil_rect.csv'))

  x0 = crop_data.x0[0]
  y0 = crop_data.y0[0]
  w = crop_data.x1[0] - x0
  h = crop_data.y1[0] - y0

  print((x0, y0, w, h))

  # crop 
  cropFile = 'D:\\ibl-website-videos\\proc\\'+ftext+'_crop'+'.mp4'
  if not exists(cropFile):
    call = subprocess.call(['ffmpeg',
                  '-i', 'D:\\ibl-website-videos\\proc\\'+ftext+'_scaled.mp4',
                  '-vf', f'crop={w}:{h}:{x0}:{y0},fps=24,scale=160:128', 
                  cropFile])