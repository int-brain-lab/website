'''Download all of the IBL video files and then compress them to 64x? pixels and 24fps'''
import numpy as np
import pandas as pd
import requests
import subprocess
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# setup folders
RAW_FOLDER = './videos/raw/'
PROC_FOLDER = './videos/proc/'

# get username/password
with open('username.txt') as f:
    username = f.readlines()
    username = username[0]
with open('password.txt') as f:
    password = f.readlines()
    password = password[0]


# load CSV file
session_table = pd.read_csv('./session.table.csv')
selectable_pids = []
for i,row in session_table.iterrows():
    if row['selectable']:
        selectable_pids.append(row['pid'])

# run pids
for pid in selectable_pids:
  eid, probe = one.pid2eid(pid)

  dsets = one.type2datasets(eid, '_iblrig_Camera.raw', details=True)
  videos = ['left','right','body']

  for video in videos:
    dset = next(d for idx, d in dsets.iterrows() if video in d['rel_path'])
    url = one.record2url(dset)
    print(url)
    ftext = pid + '_' + video
    videoFile = 'D:\\ibl-website-videos\\raw\\' +ftext + ".mp4"
    open(videoFile, 'wb').write(requests.get(url, auth=(username, password)).content)
    # also get the timestamps
    # To get timestamps for video data so you can find out what times each frame is at
    dsets_time = one.type2datasets(eid, 'camera.times')
    dset = next(d for d in dsets_time if video in d)
    # load data
    times = one.load_dataset(eid, dset)
    # with open('./videos/raw/'+ftext+'_times.txt','w') as f:
    #   f.write(str(times[0]))
    np.save('D:\\ibl-website-videos\\proc\\'+ftext+"_times.npy",times)
    call = subprocess.call(['ffmpeg',
                  '-i', 'D:\\ibl-website-videos\\raw\\'+ftext+'.mp4',
                  '-vf', 'fps=24,scale=64:52', 
                  'D:\\ibl-website-videos\\proc\\'+ftext+'_scaled'+'.mp4'])