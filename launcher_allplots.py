
# %%
from one.api import ONE
from pathlib import Path
from generator import Generator
import traceback
import joblib
from iblutil.util import setup_logger

logger = setup_logger()


one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir="/mnt/h0/kb/data/one")
SAVE_PATH = Path("/mnt/h0/kb/viz_figures_new")
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
        # logger.error(f"Error in session {session} \n {traceback.format_exc()}")
        # raise err
        errored.append({session: err})

for session in photometry_sessions:
    make_session_plots(session)
# joblib.Parallel(n_jobs=10)(joblib.delayed(make_session_plots)(session) for session in photometry_sessions)
# make_session_plots(session='/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-05236/2023-06-27/001') # good example session

# cf. make_data_js.py for the next steps to put the website online