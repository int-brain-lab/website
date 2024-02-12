from pathlib import Path
import pandas as pd
import numpy as np

data_path = Path("/mnt/home/mfaulkner/ceph")
jobs_path = Path("/mnt/home/mfaulkner/Documents/video_jobs/")
jobs_path.mkdir(exist_ok=True, parents=True)
slurm_template = Path("/mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_video_template.sbatch")


df = pd.read_parquet(data_path.joinpath('session.table.pqt'))
eids = df.eid.unique()

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
