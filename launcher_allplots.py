
# %%
from one.api import ONE
from pathlib import Path
from generator import Generator, make_data_js
import joblib

one = one = ONE(cache_dir="/mnt/h0/kb/data/one")
SAVE_PATH = Path("/mnt/h0/kb/viz_figures")
SAVE_PATH.mkdir(exist_ok=True, parents=True)

photometry_data = list(Path('/mnt/h0/kb/data/one/mainenlab/Subjects').rglob('raw_photometry.pqt'))
photometry_sessions = [str(ph.parent.parent.parent) for ph in photometry_data]
photometry_sessions = list(set(photometry_sessions))

def make_session_plots(session):
    errored = []
    try:
        session = Path(session)
        eid = one.path2eid(session)
        g = Generator(eid, one=one, data_path=session, cache_path=SAVE_PATH)
        g.make_all_plots(nums=(1, 2, 3, 4))
    except Exception as err:
        errored.append({session: err})

joblib.Parallel(n_jobs=10)(joblib.delayed(make_session_plots)(session) for session in photometry_sessions)


#  update scipt.js line 666 with the new preprocessings
# symlink the figures folder to the static/cache folder:
# ln -s /mnt/h0/kb/viz_figures /mnt/h0/kb/code_kcenia/website/static/cache
# then at last run the make data js 
# Needed to update data.js to use the correct context data. So basically uncomment the top lines in this file https://github.com/int-brain-lab/website/blob/photometry/static/data.js and comment out line 22
# also changed this line https://github.com/int-brain-lab/website/blob/photometry/static/scripts.js#L690 to reflect the new preprocessing names
make_data_js() 
#%%