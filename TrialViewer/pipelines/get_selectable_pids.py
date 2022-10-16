import pandas as pd
import pickle

from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')


# load CSV file
session_table = pd.read_csv('./session.table.csv')
selectable_pids = []
for i,row in session_table.iterrows():
    if row['selectable']:
        selectable_pids.append(row['pid'])

with open("selectable.pids", "wb") as fp:
  pickle.dump(selectable_pids, fp)