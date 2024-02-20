from pathlib import Path
import pandas as pd
import numpy as np

data_path = Path("/mnt/home/mfaulkner/ceph")
jobs_path = Path("/mnt/home/mfaulkner/Documents/video_jobs/")
jobs_path.mkdir(exist_ok=True, parents=True)
slurm_template = Path("/mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_video_template.sbatch")


# df = pd.read_parquet(data_path.joinpath('session.table.pqt'))
# eids = df.eid.unique()
eids = ['037d75ca-c90a-43f2-aca6-e86611916779',
 '695a6073-eae0-49e0-bb0f-e9e57a9275b9',
 '91e04f86-89df-4dec-a8f8-fa915c9a5f1a',
 '83d85891-bd75-4557-91b4-1cbb5f8bfc9d',
 'aec5d3cc-4bb2-4349-80a9-0395b76f04e2',
 'fa1f26a1-eb49-4b24-917e-19f02a18ac61',
 'f25642c6-27a5-4a97-9ea0-06652db79fbd',
 '8b1f4024-3d96-4ee7-95f9-8a1dfd4ce4ef',
 '5139ce2c-7d52-44bf-8129-692d61dd6403',
 '4ef13091-1bc8-4f32-9619-107bdf48540c',
 'f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1',
 '8c2f7f4d-7346-42a4-a715-4d37a5208535',
 '09394481-8dd2-4d5c-9327-f2753ede92d7',
 '952870e5-f2a7-4518-9e6d-71585460f6fe']

for i, eid in enumerate(eids):
    batch = np.ceil(i / 10)
    out_path = jobs_path.joinpath(f'batch_{batch}')
    out_path.mkdir(exist_ok=True, parents=True)

    with open(slurm_template, "r") as template:
        fdata = template.read()

    fdata = fdata.replace("zzEIDzz", eid)
    job_fn = out_path / f"viz_video_{eid}.sbatch"
    with open(job_fn, "w") as jout:
        jout.write(fdata)
