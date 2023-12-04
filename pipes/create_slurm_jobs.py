from pathlib import Path
import pandas as pd

data_path = Path("/mnt/home/mfaulkner/ceph/viz_data")
jobs_path = Path("/mnt/home/mfaulkner/Documents/viz_jobs/")
slurm_template = Path("/mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_data_template.sbatch")


df = pd.read_parquet(data_path.joinpath('session.table.pqt'))
pids = df.pid.unique()

i = 0
for pid in pids:
    with open(slurm_template, "r") as template:
        fdata = template.read()

    fdata = fdata.replace("zzPIDzz", pid)
    job_fn = jobs_path / f"viz_data_{pid}.sbatch"
    with open(job_fn, "w") as jout:
        jout.write(fdata)
