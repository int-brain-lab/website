from pathlib import Path
import pandas as pd
import numpy as np

data_path = Path("/mnt/home/mfaulkner/ceph")
jobs_path = Path("/mnt/home/mfaulkner/Documents/video_jobs/")
jobs_path.mkdir(exist_ok=True, parents=True)
slurm_template = Path("/mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_video_template.sbatch")


# df = pd.read_parquet(data_path.joinpath('session.table.pqt'))
# eids = df.eid.unique()
eids = ['8b1f4024-3d96-4ee7-95f9-8a1dfd4ce4ef',
 'd7e60cc3-6020-429e-a654-636c6cc677ea',
 '9468fa93-21ae-4984-955c-e8402e280c83',
 '3638d102-e8b6-4230-8742-e548cd87a949',
 '7cec9792-b8f9-4878-be7e-f08103dc0323',
 'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
 'a6fe44a8-07ab-49b8-81f9-e18575aa85cc',
 '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
 '1a507308-c63a-4e02-8f32-3239a07dc578',
 'e012d3e3-fdbc-4661-9ffa-5fa284e4e706',
 '952870e5-f2a7-4518-9e6d-71585460f6fe',
 '6f6d2c8e-28be-49f4-ae4d-06be2d3148c1',
 'd71e565d-4ddb-42df-849e-f99cfdeced52',
 '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
 'aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf',
 '08102cfc-a040-4bcf-b63c-faa0f4914a6f',
 'c3d9b6fb-7fa9-4413-a364-92a54df0fc5d',
 '4ef13091-1bc8-4f32-9619-107bdf48540c',
 'b22f694e-4a34-4142-ab9d-2556c3487086',
 '037d75ca-c90a-43f2-aca6-e86611916779',
 'f304211a-81b1-446f-a435-25e589fe3a5a',
 '25f77e81-c1af-46ab-8686-73ac3d67c4a7',
 '5285c561-80da-4563-8694-739da92e5dd0',
 '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
 '821f1883-27f3-411d-afd3-fb8241bbc39a',
 '0ac8d013-b91e-4732-bc7b-a1164ff3e445',
 '695a6073-eae0-49e0-bb0f-e9e57a9275b9',
 'e8b4fda3-7fe4-4706-8ec2-91036cfee6bd']

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
