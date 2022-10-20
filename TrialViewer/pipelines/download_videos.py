'''Download all of the IBL video files and then compress them to 64x? pixels and 24fps'''
import numpy as np
import pandas as pd
import requests
import subprocess
from os.path import exists
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# setup folders
RAW_FOLDER = 'D:/ibl-website-videos/raw'
PROC_FOLDER = 'D:/ibl-website-videos/proc'
out_path = './data'

new_width = 160
new_height = 128

# get username/password
with open('username.txt') as f:
    username = f.readlines()
    username = username[0]
with open('password.txt') as f:
    password = f.readlines()
    password = password[0]

import pickle
with open("selectable.pids", "rb") as fp:   # Unpickling
  selectable_pids = pickle.load(fp)

# run pids
for pid in selectable_pids:
  eid, probe = one.pid2eid(pid)
  
  if eid == 'ac7d3064-7f09-48a3-88d2-e86a4eb86461':
    continue

  dsets = one.type2datasets(eid, '_iblrig_Camera.raw', details=True)
  videos = ['left','right','body']

  for video in videos:
    dset = next(d for idx, d in dsets.iterrows() if video in d['rel_path'])
    url = one.record2url(dset)
    print(url)
    videoFile = f'{RAW_FOLDER}/{pid}_{video}.mp4'
    if not exists(videoFile):
      open(videoFile, 'wb').write(requests.get(url, auth=(username, password)).content)
    # also get the timestamps
    # To get timestamps for video data so you can find out what times each frame is at
    dsets_time = one.type2datasets(eid, 'camera.times')
    dset = next(d for d in dsets_time if video in d)
    # save timestamps
    times = one.load_dataset(eid, dset)
    if len(times)>0:
      np.save(f'{PROC_FOLDER}/{pid}_{video}_times.npy',times)

      framerate2 = 1/((times[-1]-times[0])/len(times))

      print(f'Input file true framerate is {framerate2}')

    scaledFile = f'{PROC_FOLDER}/{pid}_{video}_scaled.mp4'
    if exists(videoFile) and len(times>0):
      if not exists(scaledFile):
        call = subprocess.call(['ffmpeg',
                      '-r', f'{framerate2}',
                      '-i', videoFile,
                      '-vf', f'scale={new_width}:{new_height}', 
                      '-r', '24',
                      scaledFile])
    else:
      # create a fake timestamps file
      start_end = np.load(f'{out_path}/{pid}/{pid}_start_end.npy')
      start = start_end[0]
      end = start_end[1]
      
      times = np.arange(start,end,1/24)
      np.save(f'{PROC_FOLDER}/{pid}_{video}_times.npy',times)

      # create a fake scaled video
      print(f'Creating fake video for {pid} {video}')
      call = subprocess.call(['ffmpeg',
                            '-loop','1',
                            '-i','./img/video_not_available.png',
                            '-t', f'{end-start}',
                            '-vf', f'scale={new_width}:{new_height}',
                            '-r', '24',
                            scaledFile])