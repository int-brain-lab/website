import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import shutil
import sys
import yaml


from brainbox.behavior.dlc import get_speed, likelihood_threshold
from brainbox.io.one import _channels_alf2bunch
from iblatlas.regions import BrainRegions
from ibllib.oneibl.data_handlers import DataHandler
from neurodsp.voltage import destripe
from one.alf.files import add_uuid_string
import one.alf.io as alfio
from one.api import ONE
import spikeglx

sys.path.extend('/mnt/home/mfaulkner/Documents/PYTHON/website')

from generator import make_all_plots, make_captions

pid = sys.argv[1]
print(pid)

one = ONE()
br = BrainRegions()


SDSC_ROOT_PATH = Path("/mnt/sdceph/users/ibl/data")
TEMP_PATH = Path("/mnt/home/mfaulkner/ceph/viz_data")
SAVE_PATH = Path("/mnt/home/mfaulkner/ceph/viz_cache")
SAVE_PATH.mkdir(parents=True, exist_ok=True)


signature_eid = [
    ('_ibl_trials.table.pqt', 'alf'),
    ('_ibl_wheel.position.npy', 'alf'),
    ('_ibl_wheel.timestamps.npy', 'alf'),
    ('_ibl_leftCamera.times.npy', 'alf'),
    ('_ibl_leftCamera.dlc.pqt', 'alf'),
    ('_ibl_leftCamera.features.pqt', 'alf'),
    ('_ibl_leftCamera.ROIMotionEnergy.npy', 'alf'),
    ('licks.times.npy', 'alf'),
]

eid, pname = one.pid2eid(pid)

eid_path = TEMP_PATH.joinpath(eid)


eid_path.mkdir(exist_ok=True, parents=True)

# Transfer over the relevant session data
df = DataHandler(one.eid2path(eid), {'input_files': signature_eid}, one).getData()
for uuid, d in df.iterrows():
    file_path = Path(d['session_path']).joinpath(d['rel_path'])
    file_uuid = add_uuid_string(file_path, uuid)
    file_link = eid_path.joinpath(file_path.name)
    file_link.parent.mkdir(exist_ok=True, parents=True)
    file_link.symlink_to(
        Path(SDSC_ROOT_PATH.joinpath(file_uuid)))

camera = alfio.load_object(eid_path, 'leftCamera')
wheel = alfio.load_object(eid_path, 'wheel')


def create_fake_features(nan_array, keys=None, save=True):
    keys = keys or ['paw_r_speed', 'nose_tip_speed', 'motion_energy', 'pupilDiameter_raw', 'pupilDiameter_smooth']
    features = pd.DataFrame()
    for key in keys:
        features[key] = nan_array

    if save:
        features.to_parquet(eid_path.joinpath('_ibl_leftCamera.computedFeatures.pqt'))

    return features


def create_fake_licks(eid_path, nan_array, save=True):
    lick_file = eid_path.joinpath('licks.times.npy')
    if lick_file.exists():
        lick_file.unlink()

    if save:
        np.save(lick_file, nan_array)


if 'times' not in camera.keys():
    times = np.arange(wheel.timestamps[0], wheel.timestamps[-1], 1 / 60)
    nan_array = np.full_like(times, np.nan)
    # Save fake camera times
    np.save(eid_path.joinpath('_ibl_leftCamera.times.npy'), nan_array)
    # Create and save fake lick times file
    create_fake_licks(eid_path, nan_array)
    # Create and save fake features file
    features = create_fake_features(nan_array)

if 'dlc' not in camera.keys():
    nan_array = np.full_like(camera.times, np.nan)
    # Create and save fake lick times file
    create_fake_licks(eid_path, nan_array)
    # Create and save fake features file
    features = create_fake_features(nan_array)


if camera.times.shape[0] != camera.dlc.shape[0]:
    nan_array = np.full_like(camera.times, np.nan)
    # Create and save fake lick times file
    create_fake_licks(eid_path, nan_array)
    # Create and save fake features file
    features = create_fake_features(nan_array)


nan_array = np.full_like(camera.times, np.nan)
if 'features' not in camera.keys():
    camera.features = pd.DataFrame()
    camera.features['pupilDiameter_raw'] = nan_array
    camera.features['pupilDiameter_smooth'] = nan_array

dlc = likelihood_threshold(camera.dlc)
camera.features['paw_r_speed'] = get_speed(dlc, camera.times, 'left', feature='paw_r')
camera.features['nose_tip_speed'] = get_speed(dlc, camera.times, 'left', feature='nose_tip')

if 'ROIMotionEnergy' not in camera.keys():
    camera.features['motion_energy'] = nan_array
else:
    camera.features['motion_energy'] = camera.ROIMotionEnergy

if not eid_path.joinpath('licks.times.npy').exists():
    create_fake_licks(eid_path, nan_array)

camera.features.to_parquet(eid_path.joinpath('_ibl_leftCamera.computedFeatures.pqt'))


# Now for the probe insertion
signature_pid = [
    ('channels.mlapdv.npy', f'alf/{pname}/pykilosort'),
    ('channels.localCoordinates.npy', f'alf/{pname}/pykilosort'),
    ('channels.brainLocationIds_ccf_2017.npy',f'alf/{pname}/pykilosort'),
    ('spikes.amps.npy', f'alf/{pname}/pykilosort'),
    ('spikes.times.npy', f'alf/{pname}/pykilosort'),
    ('spikes.depths.npy', f'alf/{pname}/pykilosort'),
    ('spikes.clusters.npy', f'alf/{pname}/pykilosort'),
    ('spikes.samples.npy', f'alf/{pname}/pykilosort'),
    ('clusters.waveforms.npy', f'alf/{pname}/pykilosort'),
    ('clusters.waveformsChannels.npy', f'alf/{pname}/pykilosort'),
    ('clusters.channels.npy', f'alf/{pname}/pykilosort'),
    ('clusters.metrics.pqt', f'alf/{pname}/pykilosort'),
    ('_iblqc_ephysChannels.apRMS.npy', f'raw_ephys_data/{pname}'),
    ('*ap.meta', f'raw_ephys_data/{pname}'),
    ('*ap.cbin', f'raw_ephys_data/{pname}'),
    ('*ap.ch', f'raw_ephys_data/{pname}')]


pid_path = TEMP_PATH.joinpath(pid)

df = DataHandler(one.eid2path(eid), {'input_files': signature_pid}, one).getData()
for uuid, d in df.iterrows():
    file_path = Path(d['session_path']).joinpath(d['rel_path'])
    file_uuid = add_uuid_string(file_path, uuid)
    file_link = pid_path.joinpath(file_path.name)
    if file_link.exists():
        continue
    file_link.parent.mkdir(exist_ok=True, parents=True)
    file_link.symlink_to(
        Path(SDSC_ROOT_PATH.joinpath(file_uuid)))


def compute_cluster_average(spike_clusters, spike_var):
    """
    Quickish way to compute the average of some quantity across spikes in each cluster given
    quantity for each spike

    :param spike_clusters: cluster idx of each spike
    :param spike_var: variable of each spike (e.g spike amps or spike depths)
    :return: cluster id, average of quantity for each cluster, no. of spikes per cluster
    """
    clust, inverse, counts = np.unique(spike_clusters, return_inverse=True, return_counts=True)
    _spike_var = sparse.csr_matrix((spike_var, (inverse, np.zeros(inverse.size, dtype=int))))
    spike_var_avg = np.ravel(_spike_var.toarray()) / counts

    return clust, spike_var_avg, counts


ap_file = next(pid_path.glob('*ap.cbin'))
sr = spikeglx.Reader(ap_file)
V_T0 = [600, 60 * 30, 60 * 50]  # sample at 10, 30, 50 min
N_SEC_LOAD = 1
DISPLAY_TIME = 0.05
DISPLAY_START = 0.5

for i, T0 in enumerate(V_T0):

    start = int(T0 * sr.fs)
    end = int((T0 + N_SEC_LOAD) * sr.fs)
    if end > sr.ns:
        raw = np.full((sr.nc - sr.nsync, int(sr.fs * DISPLAY_TIME)), np.nan)
    else:
        raw = sr[start:end, :-sr.nsync].T
        raw = destripe(raw, fs=sr.fs)
        ts = int(DISPLAY_START * sr.fs)
        te = int((DISPLAY_START + DISPLAY_TIME) * sr.fs)
        raw = raw[:, ts:te]

    if i == 0:
        all_raw = raw
    else:
        all_raw = np.dstack((all_raw, raw))
np.save(pid_path.joinpath('raw_ephys_data.npy'), all_raw)

with open(pid_path.joinpath('raw_ephys_info.yaml'), 'w+') as fp:
            yaml.dump(dict(fs=sr.fs, t=V_T0, t_offset=DISPLAY_START, t_display=DISPLAY_TIME, pid=pid, nc=raw.shape[0],
                           dtype="float16"), fp)



spikes = alfio.load_object(pid_path, 'spikes')
clusters = alfio.load_object(pid_path, 'clusters')
channels = _channels_alf2bunch(alfio.load_object(pid_path, 'channels'), brain_regions=br)

metrics = clusters.pop('metrics')
for k in metrics.keys():
    clusters[k] = metrics[k].to_numpy()
for k in channels.keys():
    clusters[k] = channels[k][clusters['channels']]

keys_to_keep = ['cluster_id', 'x', 'y', 'z', 'atlas_id', 'acronym', 'label', 'amp_max', 'channels']

clust_id, amps, _ = compute_cluster_average(spikes.clusters, spikes.amps)
clust_amps = np.full(clusters.cluster_id.size, np.nan)
clust_amps[clust_id] = amps

clust_id, depths, counts = compute_cluster_average(spikes.clusters, spikes.depths)
clust_depths = np.full(clusters.cluster_id.size, np.nan)
clust_depths[clust_id] = depths

fr = counts / (np.max(spikes.times) - np.min(spikes.times))
clust_fr = np.full(clusters.cluster_id.size, np.nan)
clust_fr[clust_id] = fr

clusters_new = pd.DataFrame()
for key in keys_to_keep:
    clusters_new[key] = clusters[key]
clusters_new['amps'] = clust_amps
clusters_new['depths'] = clust_depths
clusters_new['firing_rate'] = clust_fr
clusters_new['pid'] = pid

clusters_new.to_parquet(pid_path.joinpath('clusters.table.pqt'))

good_idx = np.where(clusters['label'] == 1)[0]
idx = np.isin(spikes.clusters, clusters.cluster_id[good_idx])
np.save(pid_path.joinpath('spikes.good.npy'), idx)

metrics = next(pid_path.glob('clusters.metrics*'))
metrics.unlink()

# Make all the plots that we need for the website
make_all_plots(pid, data_path=TEMP_PATH, cache_path=SAVE_PATH)
