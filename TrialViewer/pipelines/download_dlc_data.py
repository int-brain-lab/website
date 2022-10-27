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
PAD_S = 2

new_fs = 24
new_width = 160
new_height = 128

import pickle
with open("selectable.pids", "rb") as fp:   # Unpickling
  selectable_pids = pickle.load(fp)

skip_pids = ['176b4fe3-f570-4d9f-9e25-a5d218f75c8b','68c98fb2-402b-4670-9b3b-0c77619998ea']

for pid in selectable_pids:
  eid, probe = one.pid2eid(pid) #'0802ced5-33a3-405e-8336-b65ebc5cb07c'

  start_end = np.load(f'{out_path}/{pid}/{pid}_start_end.npy')
  start = start_end[0]
  end = start_end[1]
  frames = start_end[2]
  
  print(f'Computing frame times up to {end-start} for {frames} frames')
  frame_times = np.arange(0,end-start,1/new_fs) + start

  video_params = {
      'left': {
          'fps': 60,
          'width': 1280,
          'height': 1024,
          'resolution': 2
      },
      'right': {
          'fps': 150,
          'width': 640,
          'height': 512,
          'resolution': 1
      },
      'body': {
          'fps': 30,
          'width': 640,
          'height': 512,
          'resolution': 1
      }
  }


  for label in ['left', 'right', 'body']:
    video_path = data_path.joinpath(f'{pid}_{label}_scaled.mp4')
    video_meta = get_video_meta(video_path)

    w_subsamp_factor = video_params[label]['width'] / new_width
    h_subsamp_factor = video_params[label]['height'] / new_height

    try:
      video_data = one.load_object(eid, f'{label}Camera', collection='alf')
    except:
      print(f'Failed for {pid} -- skipping')

      # todo: generate DLC files when videos are missing
      # Save the files
      # np.save(out_path.joinpath(f'{pid}/{pid}_{label}_times_scaled.npy'), frame_times)
      # dlc.to_csv(out_path.joinpath(f'{pid}/{pid}_{label}_dlc_scaled.csv'), index = False)
      # # dlc.to_parquet(data_path.joinpath(f'{eid}_{label}_dlc_scaled.pqt'))
      # np.save(out_path.joinpath(f'{pid}/{pid}_{label}_wheel_scaled.npy'), wheel_pos)

      continue

    if 'dlc' not in video_data.keys():
      print(f'No DLC for {pid} -- skipping')
      continue

    ts = video_data['times']

    # if timestamps don't match we need to recompute
    if video_data['dlc'].shape[0] != video_data['times'].size:
        qc = CameraQC(eid, label, stream=True, n_samples=1)
        qc.load_data(extract_times=True)
        ts = qc.data.timestamps

    print((ts.size, video_data['dlc'].shape[0]))
    if not ts.size == video_data['dlc'].shape[0]:
      print(f'MISALIGNED for {pid} -- skipping')
      continue
    video_data['times'] = ts

    # Make sure the times and video are same size. (This doesn't hold for right video! Mismatch by one frame)
    # assert subsamp_times.size == video_meta['length']


    dlc = pd.DataFrame()
    dlc_positions = copy.copy(video_data.dlc)

    for col in dlc_positions.keys():
        if 'pupil' in col:
            dlc[col] = interpolate.interp1d(video_data.times, dlc_positions[col].values, fill_value=-1, bounds_error=False)(frame_times)
        else:
            if 'likelihood' in col:
                dlc[col] = interpolate.interp1d(video_data.times, dlc_positions[col].values, fill_value=-1, bounds_error=False)(frame_times)
            elif '_x' in col:
                dlc[col] = interpolate.interp1d(video_data.times, dlc_positions[col].values, fill_value=-1, bounds_error=False)(frame_times) / w_subsamp_factor
            elif '_y' in col:
                dlc[col] = interpolate.interp1d(video_data.times, dlc_positions[col].values, fill_value=-1, bounds_error=False)(frame_times) / h_subsamp_factor


    if label == 'left':
        p_pupil_x = np.mean(dlc[['pupil_top_r_x', 'pupil_left_r_x', 'pupil_right_r_x', 'pupil_bottom_r_x']].mean())
        p_pupil_y = np.mean(dlc[['pupil_top_r_y', 'pupil_left_r_y', 'pupil_right_r_y', 'pupil_bottom_r_y']].mean())

        res = video_params[label]['resolution']
        df_pupil = dict()
        # get the center coordinate 
        df_pupil['x0'] = int(p_pupil_x - new_width/2)
        df_pupil['y0'] = int(p_pupil_y - new_height/2)
        df_pupil['x_mu'] = int(p_pupil_x)
        df_pupil['y_mu'] = int(p_pupil_y)
        df_pupil = pd.DataFrame.from_dict([df_pupil])

        df_pupil.to_csv(out_path.joinpath(f'{pid}/{pid}_{label}_pupil_rect.csv'), index = False)

    # check that the positions make sense
    # plot_trace_on_frame(get_video_frame(video_path, 200), dlc, label)

    # interpolate the wheel data at subsampled timestamps
    wheel_data = one.load_object(eid, 'wheel', collection='alf')
    wheel_pos = interpolate.interp1d(wheel_data.timestamps, wheel_data.position, kind='linear')(frame_times)

    # check the interpolation makes sense
    # fig, ax = plt.subplots()
    # ax.plot(wheel_data.timestamps, wheel_data.position, 'x')
    # ax.plot(subsamp_times, wheel_pos, 'o')

    # Save the files
    np.save(out_path.joinpath(f'{pid}/{pid}_{label}_times_scaled.npy'), frame_times)
    dlc.to_csv(out_path.joinpath(f'{pid}/{pid}_{label}_dlc_scaled.csv'), index = False)
    # dlc.to_parquet(data_path.joinpath(f'{eid}_{label}_dlc_scaled.pqt'))
    np.save(out_path.joinpath(f'{pid}/{pid}_{label}_wheel_scaled.npy'), wheel_pos)




