from one.api import ONE
from pathlib import Path
from generator import Generator, make_data_js
import joblib
import logging
import traceback

logger = logging.getLogger('ibllib')

one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir="/mnt/h0/kb/data/one")
SAVE_PATH = Path("/mnt/h0/kb/viz_figures")
SAVE_PATH.mkdir(exist_ok=True, parents=True)

photometry_data = list(Path('/mnt/h0/kb/data/one/mainenlab/Subjects').rglob('raw_photometry.pqt'))
photometry_sessions = [str(ph.parent.parent.parent) for ph in photometry_data]
photometry_sessions = list(set(photometry_sessions))
len(photometry_data)
len(photometry_sessions)


def make_session_plots(session):
    errored = []
    session = Path(session)
    eid = one.path2eid(session)
    g = Generator(eid, one=one, data_path=session, cache_path=SAVE_PATH)
    g.make_all_plots(nums=(1, 2, 3, 4))


# make_session_plots(session)
# 51 first movement times error /mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-03448/2021-12-10/
IMIN = 52
for i, session in enumerate(photometry_sessions):
    if i < IMIN:
        continue
    make_session_plots(session)
# joblib.Parallel(n_jobs=10)(joblib.delayed(make_session_plots)(session) for session in photometry_sessions)
