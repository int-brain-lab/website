# Convert the timestamp data (video timestamps + DLC)

# The first part of this is that we need to convert the timestamps to match frames across all three videos
# the "ground-truth" frames will come from the LEFT video 

# Now pull the corresponding DLC frames to match this, we want all DLC frames
# if a DLC frame is missing we will replace it with -1

# Load the trial data files and convert them to just the columns that we want
import numpy as np
import pandas as pd
from one.api import ONE

one = ONE()

DATA_DIR = './data'

def flatten(list_of_list):
  ret = []
  for list in list_of_list:
    ret += list
  return ret

def get_dlc_coord(row, key):
  val = row[key]
  if np.isnan(val):
    val = -1
  return val

# load CSV file
session_table = pd.read_csv('./session.table.csv')
selectable_pids = []
for i,row in session_table.iterrows():
  if row['selectable']:
    selectable_pids.append(row['pid'])


for pid in selectable_pids:
  # load timestamp files
  body_ts = np.load(f'{DATA_DIR}/{pid}/{pid}_body_times_scaled.npy')
  left_ts = np.load(f'{DATA_DIR}/{pid}/{pid}_left_times_scaled.npy')
  right_ts = np.load(f'{DATA_DIR}/{pid}/{pid}_right_times_scaled.npy')
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
  indexes = pd.DataFrame(columns=['left_ts','right_idx','body_idx','wheel'] + dlc_body_keys_xy + dlc_left_keys_xy + dlc_right_keys_xy + dlc_left_pupil_keys_xy)
  data_cols = {}

  crop_data = pd.read_csv(f'{DATA_DIR}/{pid}/{pid}_left_pupil_rect.csv')

  pupil_xy = [crop_data.x0_ss[0], crop_data.y0_ss[0]]

  suffixes = ['_x', '_y']

  for key in dlc_body_keys + dlc_left_keys + dlc_right_keys + dlc_left_pupil_keys:
    data_cols[key] = []

  for i, lts in enumerate(left_ts):
    # get indexes
    right_idx = np.argmin(np.abs(right_ts-lts))
    body_idx = np.argmin(np.abs(body_ts-lts))

    row_data = [lts, right_idx, body_idx, wheel[i]]

    # pull the data columns
    for key in dlc_body_keys:
      for suffix in suffixes:
        row_data.append(get_dlc_coord(dlc_body.iloc[body_idx], key + suffix))
    for key in dlc_left_keys:
      for suffix in suffixes:
        row_data.append(get_dlc_coord(dlc_left.iloc[i], key[3:] + suffix))
    for key in dlc_right_keys:
      for suffix in suffixes:
        row_data.append(get_dlc_coord(dlc_right.iloc[right_idx], key[3:] + suffix))

    # get the pupil, and offset by the x0/y0 values
    for key in dlc_left_pupil_keys:
      for j, suffix in enumerate(suffixes):
        val = get_dlc_coord(dlc_left.iloc[i], key + suffix) - pupil_xy[j]
        row_data.append(val)
    
  indexes.loc[i] = row_data

  for key in indexes.keys():
    dat = indexes[key].values.astype(np.float32)
    with open(f'{DATA_DIR}/{pid}/{pid}.{key}.bytes', 'wb') as file:
        file.write(dat.tobytes())