from iblatlas.atlas import AllenAtlas
from iblatlas.atlas import Insertion
import pandas as pd
from one.api import ONE
one = ONE()
ba = AllenAtlas()

session_df = pd.DataFrame()
for pid in pids:
    try:
        ins = one.alyx.rest('insertions', 'list', id=pid)[0]
        traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
        probe_insertion=ins['id'])[0]
        # ins = one.alyx.rest('insertions', 'list', id=d['id'])[0]
        insert = Insertion.from_dict(traj, ba)
        subject = one.alyx.rest('subjects', 'list', nickname=ins['session_info']['subject'])[0]

        tip = insert.tip
        tip_ccf = ba.xyz2ccf(tip)

        selectable = True
        repeated = False

        data = {
            'pid': ins['id'],
            'eid': ins['session_info']['id'],
            'probe': ins['name'],
            'lab': ins['session_info']['lab'],
            'subject': ins['session_info']['subject'],
            'date': ins['session_info']['start_time'][0:10],
            'dob': subject['birth_date'],
            'probe_model': ins['model'],
            'x': traj['x'],
            'y': traj['y'],
            'z': traj['z'],
            'depth': traj['depth'],
            'theta': traj['theta'],
            'phi': traj['phi'],
            'roll': traj['roll'],
            'ml_ccf_tip': tip_ccf[0],
            'ap_ccf_tip': tip_ccf[1],
            'dv_ccf_tip': tip_ccf[2],
            'selectable': selectable,
            '2022_Q4_IBL_et_al_BWM': True,
            '2022_Q2_IBL_et_al_RepeatedSite': repeated
        }

        session_df = pd.concat([session_df, pd.DataFrame.from_dict([data])])
    except Exception as err:
        print(f'{pid}: {err}')