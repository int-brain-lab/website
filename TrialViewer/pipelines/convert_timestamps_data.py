# Convert the timestamp data (video timestamps + DLC)

# The first part of this is that we need to convert the timestamps to match frames across all three videos
# the "ground-truth" frames will come from the LEFT video 

# Now pull the corresponding DLC frames to match this, we want all DLC frames
# if a DLC frame is missing we will replace it with -1

# Load the trial data files and convert them to just the columns that we want
from telnetlib import OUTMRK
import numpy as np
import pandas as pd

DATA_DIR = './data'
OUT_DIR = './final'

def flatten(list_of_list):
  ret = []
  for list in list_of_list:
    ret += list
  return ret

def get_dlc_coord(row, key, suffix):
  val = row[key + suffix]
  like = row[key + '_likelihood']
  if like < 0.1 or np.isnan(val):
    val = -1
  return val

import pickle
with open("selectable.pids", "rb") as fp:   # Unpickling
  selectable_pids = pickle.load(fp)


for pid in selectable_pids:
  print(f'Starting {pid}')

  # trim the timestamps to match the videos
  # load the start/end points
  start_end = np.load(f'{DATA_DIR}/{pid}/{pid}_start_end.npy')
  start = start_end[0]
  end = start_end[1]

  left_ts = np.arange(0,end-start,1/24) + start


  # Load csv files
  dlc_right = pd.read_csv(f'{DATA_DIR}/{pid}/{pid}_right_dlc_scaled.csv')
  dlc_left = pd.read_csv(f'{DATA_DIR}/{pid}/{pid}_left_dlc_scaled.csv')
  dlc_body = pd.read_csv(f'{DATA_DIR}/{pid}/{pid}_body_dlc_scaled.csv')
  # Load wheel
  wheel = np.load(f'{DATA_DIR}/{pid}/{pid}_left_wheel_scaled.npy')
  # we will also be adding all the DLC points to this, we'll get them automatically and we'll get both the _x and _y coordinates for each
  dlc_body_keys = ['tail_start']
  dlc_left_keys = ['cl_nose_tip', 'cl_paw_l', 'cl_paw_r', 'cl_tube_top', 'cl_tongue_end_l', 'cl_tongue_end_r']
  dlc_right_keys = ['cr_nose_tip', 'cr_paw_l', 'cr_paw_r', 'cr_tube_top', 'cr_tongue_end_l', 'cr_tongue_end_r']
  # we'll handle pupil separately because it was cropped out
  dlc_left_pupil_keys = ['pupil_right_r', 'pupil_left_r', 'pupil_top_r', 'pupil_bottom_r']

  # generate the x/y versions
  dlc_body_keys_xy = flatten([(x + '_x', x + '_y') for x in dlc_body_keys])
  dlc_left_keys_xy = flatten([(x + '_x', x + '_y') for x in dlc_left_keys])
  dlc_right_keys_xy = flatten([(x + '_x', x + '_y') for x in dlc_right_keys])
  dlc_left_pupil_keys_xy = flatten([(x + '_x', x + '_y') for x in dlc_left_pupil_keys])

  # 1 re-index videos, dlc, and wheel into one file
  indexes = pd.DataFrame(columns=['left_ts','wheel'] + dlc_body_keys_xy + dlc_left_keys_xy + dlc_right_keys_xy + dlc_left_pupil_keys_xy)
  data_cols = {}

  crop_data = pd.read_csv(f'{DATA_DIR}/{pid}/{pid}_left_pupil_rect.csv')

  pupil_xy = [crop_data.x0[0], crop_data.y0[0]]

  suffixes = ['_x', '_y']

  for key in dlc_body_keys + dlc_left_keys + dlc_right_keys + dlc_left_pupil_keys:
    data_cols[key] = []

  for i, lts in enumerate(left_ts):
    # get indexes

    row_data = [lts, wheel[i]]

    # pull the data columns
    for key in dlc_body_keys:
      for suffix in suffixes:
        row_data.append(get_dlc_coord(dlc_body.iloc[i], key, suffix))
    for key in dlc_left_keys:
      for suffix in suffixes:
        row_data.append(get_dlc_coord(dlc_left.iloc[i], key[3:], suffix))
    for key in dlc_right_keys:
      for suffix in suffixes:
        row_data.append(get_dlc_coord(dlc_right.iloc[i], key[3:], suffix))

    # get the pupil, and offset by the x0/y0 values
    for key in dlc_left_pupil_keys:
      for j, suffix in enumerate(suffixes):
        val = get_dlc_coord(dlc_left.iloc[i], key, suffix)
        if val > -1:
          row_data.append(val - pupil_xy[j])
        else:
          row_data.append(val)
    
    indexes.loc[i] = row_data

  indexes.to_csv(f'{DATA_DIR}/{pid}/indexes.csv')
  for key in indexes.keys():
    dat = indexes[key].values.astype(np.float32)
    with open(f'{OUT_DIR}/{pid}/{pid}.{key}.bytes', 'wb') as file:
      file.write(dat.tobytes())
  print(f'Finished {pid}')
