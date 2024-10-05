from one.api import ONE
from pathlib import Path
from generator import Generator, make_data_js
import joblib
from iblutil.util import setup_logger
import logging
import traceback

logger = setup_logger(name='ibl', level='INFO', file=Path('/home/olivier/scratch/photometry.log'))

one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir="/mnt/h0/kb/data/one")
SAVE_PATH = Path("/mnt/h0/kb/viz_figures")
SAVE_PATH.mkdir(exist_ok=True, parents=True)

photometry_data = list(Path('/mnt/h0/kb/data/one/mainenlab/Subjects').rglob('raw_photometry.pqt'))
photometry_sessions = [str(ph.parent.parent.parent) for ph in photometry_data]
photometry_sessions = list(set(photometry_sessions))
len(photometry_data)
len(photometry_sessions)


def make_session_plots(session):
    try:
        session = Path(session)
        eid = one.path2eid(session)
        g = Generator(eid, one=one, data_path=session, cache_path=SAVE_PATH)
        g.make_all_plots(nums=(1, 2, 3, 4))
    except Exception as e:
        logger.error(f"{e} {session}")
        traceback.print_exc()


# make_session_plots(session)
# 51 first movement times error /mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-03448/2021-12-10/
# 227 argmin of empty sequence /mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-06171/2023-10-24/001
# 450 filter critical frequencies must be greater than 0 /mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-06275/2023-07-24/001
# 475 filter critical frequencies must be greater than 0 /mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-06275/2023-07-25/001

IMIN = 475
for i, session in enumerate(photometry_sessions):
    if i < IMIN:
        continue
    print(f"{session}")
    make_session_plots(session)
# joblib.Parallel(n_jobs=10)(joblib.delayed(make_session_plots)(session) for session in photometry_sessions[IMIN:])
