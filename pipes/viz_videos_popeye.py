import numpy as np
import pandas as pd
import shutil
import datetime
import subprocess
import time
from pathlib import Path
import sys

from one.api import ONE
import one.alf.io as alfio
from ibllib.io.video import get_video_meta, url_from_eid, get_video_frames_preload
from brainbox.behavior.dlc import likelihood_threshold
from neurodsp.utils import WindowGenerator
from ibllib.oneibl.data_handlers import DataHandler
from one.alf.files import add_uuid_string
import cv2


one = ONE()

eid = sys.argv[1]
print(eid)

SDSC_ROOT_PATH = Path("/mnt/sdceph/users/ibl/data")
ROOT_PATH = Path("/mnt/home/mfaulkner/ceph")
DATA_PATH = ROOT_PATH.joinpath('video_data')
TEMP_DATA_DIR = ROOT_PATH.joinpath('data_temp')
FINAL_DATA_DIR = ROOT_PATH.joinpath('data_final')
TEMP_VIDEO_DIR = ROOT_PATH.joinpath('video_temp')
FINAL_VIDEO_DIR = ROOT_PATH.joinpath('video_final')
NO_VIDEO_IMG = str(ROOT_PATH.joinpath('img', 'video_not_available.png'))


NEW_FS = 24
NEW_WIDTH = 160
NEW_HEIGHT = 128
PAD_S = 0.2

VIDEO_PARAMS = {
    'left': {
        'fps': 60,
        'width': 1280,
        'height': 1024,
    },
    'right': {
        'fps': 150,
        'width': 640,
        'height': 512,
    },
    'body': {
        'fps': 30,
        'width': 640,
        'height': 512,
    },
    'pupil': {
        'fps': 60,
        'width': 160,
        'height': 128,
    }
}

DLC_FEATURES = {
    'left': {'paw_r': (0, 0, 255),
             'paw_l': (125, 125, 0),
             'tongue_end_r': (0, 0, 255),
             'tube_top': (0, 125, 0)},
    'right': {'paw_r': (125, 125, 0),
             'paw_l': (0, 0, 255),
             'tongue_end_r': (0, 0, 255),
             'tube_top': (0, 125, 0)},
    'body': {'tail_start': (0, 255, 255)},
    'pupil': {'pupil_top_r': (125, 0, 125),
              'pupil_bottom_r': (125, 0, 125),
              'pupil_left_r': (125, 0, 125),
              'pupil_right_r': (125, 0, 125)}
}

CAMERAS = ['body', 'left', 'right']
SIGNATURES = ([(f'_iblrig_{cam}Camera.raw.mp4', 'raw_video_data') for cam in CAMERAS] +
              [(f'_ibl_{cam}Camera.times.npy', 'alf') for cam in CAMERAS] +
              [(f'_ibl_{cam}Camera.dlc.pqt', 'alf') for cam in CAMERAS] +
              [('_ibl_trials.table.pqt', 'alf')])


def make_eid_directories(eid):
    TEMP_DATA_DIR.joinpath(eid).mkdir(exist_ok=True, parents=True)
    FINAL_DATA_DIR.joinpath(eid).mkdir(exist_ok=True, parents=True)
    TEMP_VIDEO_DIR.joinpath(eid).mkdir(exist_ok=True, parents=True)
    DATA_PATH.joinpath(eid).mkdir(exist_ok=True, parents=True)


def make_directories():
    TEMP_DATA_DIR.mkdir(exist_ok=True, parents=True)
    FINAL_DATA_DIR.mkdir(exist_ok=True, parents=True)
    TEMP_VIDEO_DIR.mkdir(exist_ok=True, parents=True)
    FINAL_VIDEO_DIR.mkdir(exist_ok=True, parents=True)
    DATA_PATH.mkdir(exist_ok=True, parents=True)


def _run_command(command):
    """
    Run a shell command using subprocess.

    :param command: command to run
    :return: dictionary with keys: process, stdout, stderr
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    info, error = process.communicate()
    return {
        'process': process,
        'stdout': info.decode(),
        'stderr': error.decode()}


def get_data_popeye(eid, one):

    eid_path = DATA_PATH.joinpath(eid)
    eid_path.mkdir(exist_ok=True, parents=True)

    # Transfer over the relevant session data
    df = DataHandler(one.eid2path(eid), {'input_files': SIGNATURES}, one).getData()
    for uuid, d in df.iterrows():
        file_path = Path(d['session_path']).joinpath(d['rel_path'])
        file_uuid = add_uuid_string(file_path, uuid)
        file_link = eid_path.joinpath(file_path.name)
        if file_link.exists():
            continue
        file_link.parent.mkdir(exist_ok=True, parents=True)
        file_link.symlink_to(
            Path(SDSC_ROOT_PATH.joinpath(file_uuid)))


def check_video_status(eid, one):

    video_status = {}

    sess_qc = one.alyx.rest('sessions', 'read', id=eid)['extended_qc']

    for label in CAMERAS:
        video = next(DATA_PATH.joinpath(eid).rglob(f'_iblrig_{label}Camera.raw.mp4'), False)
        times = next(DATA_PATH.joinpath(eid).rglob(f'_ibl_{label}Camera.times.npy'), False)
        dlc = next(DATA_PATH.joinpath(eid).rglob(f'_ibl_{label}Camera.dlc.pqt'), False)
        status = all([video, times, dlc])
        if status:
            video_times = alfio.load_file_content(next(DATA_PATH.joinpath(eid).rglob(f'_ibl_{label}Camera.times.npy')))
            video_meta = get_video_meta(next(DATA_PATH.joinpath(eid).rglob(f'_iblrig_{label}Camera.raw.mp4')))
            if video_meta['length'] != video_times.size:
                status = False
            if sess_qc[f'video{label.capitalize()}'] == 'CRITICAL':
                status = False

        video_status[label] = status

    return video_status


def get_frame_rate(eid, label):
    if label == 'pupil':
        times = alfio.load_file_content(next(DATA_PATH.joinpath(eid).rglob(f'_ibl_leftCamera.times.npy')))
    else:
        times = alfio.load_file_content(next(DATA_PATH.joinpath(eid).rglob(f'_ibl_{label}Camera.times.npy')))

    fps = 1 / np.mean(np.diff(times))

    return fps


# 1. Load in the trials data
def load_trial_data(eid):

    trials = alfio.load_object(DATA_PATH.joinpath(eid), 'trials')
    # Remove trials with nan in either the stim On or the feedback times
    nan_idx = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['feedback_times']))
    trials = {key: val[~nan_idx] for key, val in trials.items()}

    # Find start and end times of trials and the number of frames between them
    start = trials['stimOn_times'][0] - PAD_S
    end = trials['feedback_times'][-1] + PAD_S
    frame_times = np.arange(0, end - start, 1 / NEW_FS) + start
    # Save start and end time
    np.save(TEMP_DATA_DIR.joinpath(eid, f'{eid}_start_end.npy'), np.array([start, end]))

    # Convert times into index
    # start, stim_on, feedback, end
    stim_start = trials['intervals'][:, 0]
    stim_start = np.searchsorted(frame_times, stim_start, 'left')
    stim_on = trials['stimOn_times']
    stim_on = np.searchsorted(frame_times, stim_on, 'left')
    feedback = trials['feedback_times']
    feedback = np.searchsorted(frame_times, feedback, 'left')

    # Construct dataframe with the trials info we need
    trials_df = pd.DataFrame()
    trials_df['start_timestamp'] = stim_start
    trials_df['stim_on_timestamp'] = stim_on
    trials_df['feedback_timestamp'] = feedback
    trials_df['contrast'] = np.nansum(np.c_[trials['contrastLeft'], trials['contrastRight']], axis=1)
    trials_df['feedback'] = trials['feedbackType']

    right_idx = np.where(~np.isnan(trials['contrastRight']))[0]
    stim_side = np.full(trials['contrastLeft'].shape, 'L')
    stim_side[right_idx] = 'R'
    trials_df['stimulus_side'] = stim_side

    # Save dataframe
    trials_df.to_csv(FINAL_DATA_DIR.joinpath(eid, f'{eid}.trials.csv'), index=False)


def load_frame_times(eid):

    # Load start and end time
    start, end = np.load(TEMP_DATA_DIR.joinpath(eid).joinpath(f'{eid}_start_end.npy'))
    frame_times = np.arange(0, end - start, 1 / NEW_FS) + start

    return frame_times, start, end


def load_dlc_data(eid):

    dlc_pupil = alfio.load_file_content(next(DATA_PATH.joinpath(eid).rglob('_ibl_leftCamera.dlc.pqt')))
    dlc_pupil = likelihood_threshold(dlc_pupil)
    p_pupil_x = np.nanmean(
        np.nanmean(np.c_[dlc_pupil['pupil_top_r_x'].values, dlc_pupil['pupil_left_r_x'].values,
                         dlc_pupil['pupil_right_r_x'].values, dlc_pupil['pupil_bottom_r_x'].values], axis=0))
    p_pupil_y = np.nanmean(
        np.nanmean(np.c_[dlc_pupil['pupil_top_r_y'].values, dlc_pupil['pupil_left_r_y'].values,
                         dlc_pupil['pupil_right_r_y'].values, dlc_pupil['pupil_bottom_r_y'].values], axis=0))
    # get the center coordinate
    df_pupil = dict()
    df_pupil['x0'] = int(p_pupil_x - NEW_WIDTH / 2) if not np.isnan(p_pupil_x) else p_pupil_x
    df_pupil['y0'] = int(p_pupil_y - NEW_HEIGHT / 2) if not np.isnan(p_pupil_y) else p_pupil_y
    df_pupil = pd.DataFrame.from_dict([df_pupil])
    df_pupil.to_csv(TEMP_DATA_DIR.joinpath(eid, f'{eid}_pupil_rect.csv'), index=False)

    dlc_pupil = alfio.load_file_content(next(DATA_PATH.joinpath(eid).rglob('_ibl_leftCamera.dlc.pqt')))
    for col in dlc_pupil.keys():
        if 'pupil' in col:
            if '_x' in col:
                dlc_pupil[col] = dlc_pupil[col].values - df_pupil['x0'][0]
            if '_y' in col:
                dlc_pupil[col] = dlc_pupil[col].values - df_pupil['y0'][0]

    dlc_pupil.to_parquet(TEMP_DATA_DIR.joinpath(eid, f'_ibl_pupilCamera.dlc.pqt'))


# Step 3: downscale left, right and body cameras
def downscale_video(eid, label, framerate):

    input_file = str(next(DATA_PATH.joinpath(eid).rglob(f'_iblrig_{label}Camera.raw.mp4')))
    output_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_downscale.mp4'))

    print(f'Downscaling video for {eid} {label}')
    cmd = f'ffmpeg -y -r {framerate} -i {input_file} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} {output_file}'
    _ = _run_command(cmd)

# Step 4: crop out the pupil from the left video
def crop_pupil_video(eid, framerate):

    input_file = str(next(DATA_PATH.joinpath(eid).rglob(f'_iblrig_leftCamera.raw.mp4')))
    output_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_pupil_downscale.mp4'))

    print(f'Cropping pupil for {eid}')
    pupil_df = pd.read_csv(TEMP_DATA_DIR.joinpath(eid, f'{eid}_pupil_rect.csv'))
    x0 = pupil_df.x0[0]
    y0 = pupil_df.y0[0]

    cmd = (f'ffmpeg -y -r {framerate} -i {input_file} -vf crop={NEW_WIDTH}:{NEW_HEIGHT}:{x0}:{y0},scale=160:128 {output_file}')
    _ = _run_command(cmd)


# Step 5: Overlay dlc points onto the videos
def overlay_dlc(eid, label, frame_rate):

    input_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_downscale.mp4'))
    output_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_overlay.mp4'))

    if label == 'pupil':
        dlc = pd.read_parquet(TEMP_DATA_DIR.joinpath(eid, f'_ibl_pupilCamera.dlc.pqt'))
    else:
        dlc = alfio.load_file_content(next(DATA_PATH.joinpath(eid).rglob(f'_ibl_{label}Camera.dlc.pqt')))

    dlc = likelihood_threshold(dlc)
    video_meta = get_video_meta(input_file)

    w_subsamp_factor = VIDEO_PARAMS[label]['width'] / NEW_WIDTH
    h_subsamp_factor = VIDEO_PARAMS[label]['height'] / NEW_HEIGHT

    video_out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                                (video_meta['width'], video_meta['height']), True)

    wg = WindowGenerator(video_meta['length'], 5000, 0)
    for iw, (first, last) in enumerate(wg.firstlast):
        frames = get_video_frames_preload(input_file, np.arange(first, last))
        for i, frame in enumerate(frames):
            image = frame
            for feat in DLC_FEATURES[label].keys():
                x = dlc[f'{feat}_x'][first + i] / w_subsamp_factor
                y = dlc[f'{feat}_y'][first + i] / h_subsamp_factor
                if np.isnan(x) or np.isnan(y):
                    continue
                image = cv2.circle(image, (int(x), int(y)), 4, DLC_FEATURES[label][feat], -1)

            video_out.write(image)

    video_out.release()


# Step 6: Downsample the framerate for all videos
def downsample_video(eid, label, frame_rate):

    input_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_overlay.mp4'))
    output_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_downsample.mp4'))

    print(f'Downsampling video for {eid} {label}')
    cmd = f'ffmpeg -y -r {frame_rate} -i {input_file} -r {NEW_FS} {output_file}'
    _ = _run_command(cmd)


# Step 7: Trim the videos so that they all have the same length
def trim_videos(eid, label):

    _, start, end = load_frame_times(eid)
    length_t = ffmpeg_time_format(end - start)

    if label == 'pupil':
        times = alfio.load_file_content(next(DATA_PATH.joinpath(eid).rglob('_ibl_leftCamera.times.npy')))
    else:
        times = alfio.load_file_content(next(DATA_PATH.joinpath(eid).rglob(f'_ibl_{label}Camera.times.npy')))

    start_t = ffmpeg_time_format(start - times[0])

    input_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_downsample.mp4'))
    output_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_trim.mp4'))

    print(times[0] - start)
    assert (times[0] <= start), f'problem with times for {label}'

    cmd = f'ffmpeg -y -i {input_file} -ss {start_t} -t {length_t} {output_file}'
    _ = _run_command(cmd)


# Step 8: concatenate all the videos into one
def concatenate_videos(eid):

    labels = ['left', 'right', 'body', 'pupil']
    input_files = [str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_trim.mp4')) for label in labels]
    output_file = str(FINAL_VIDEO_DIR.joinpath(f'{eid}.mp4'))

    cmd = (f'ffmpeg -y -i {input_files[0]} -i {input_files[1]} -i {input_files[2]} -i {input_files[3]} '
           f'-filter_complex [0:v][1:v][2:v][3:v]hstack=inputs=4[v] -map [v] {output_file}')
    _ = _run_command(cmd)


# Step 9: clean up some folders
def cleanup_data(eid):
    if Path(FINAL_VIDEO_DIR.joinpath(f'{eid}.mp4')).exists():
        shutil.rmtree(DATA_PATH.joinpath(eid))
        shutil.rmtree(TEMP_DATA_DIR.joinpath(eid))
        shutil.rmtree(TEMP_VIDEO_DIR.joinpath(eid))
    else:
        print('Final concatenated video missing')


def create_fake_video(eid, label):

    times, start, end = load_frame_times(eid)
    length_t = ffmpeg_time_format(end - start)
    start_t = ffmpeg_time_format(start - times[0])

    output_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_trim.mp4'))

    # cmd = (f'ffmpeg -y -loop 1 -i {NO_VIDEO_IMG} -t {end - start} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -pix_fmt yuv420p '
    #        f'-profile:v main -r {NEW_FS} {output_file}')
    #
    # cmd = (f'ffmpeg -y -i {video_file} -ss {start_t} -t {length_t} {trim_file}')

    cmd = (f'ffmpeg -y -loop 1 -i {NO_VIDEO_IMG} -ss {start_t} -t {length_t} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -pix_fmt yuv420p '
           f'-profile:v main -r {NEW_FS} {output_file}')
    _ = _run_command(cmd)

    return output_file


def ffmpeg_time_format(time):
    date = datetime.timedelta(seconds=time)
    return str(date)


def print_elapsed_time(start_time):
    print(f'Elapsed time: {time.time() - start_time}')


def status_with_pupil(labels):

    if 'left' in labels:
        labels.append('pupil')
    return labels


def process_all(eid, one):

    start_time = time.time()

    make_directories()
    make_eid_directories(eid)

    print('Downloading data')
    get_data_popeye(eid, one)
    print_elapsed_time(start_time)

    print('Checking video status')
    video_status = check_video_status(eid, one)
    good_labels = []
    fake_labels = []
    for label, status in video_status.items():
        print(f'{label}: {status}')
        if status:
            good_labels.append(label)
        else:
            fake_labels.append(label)

    fake_labels = status_with_pupil(fake_labels.copy())
    pupil_labels = status_with_pupil(good_labels.copy())
    frame_rates = {}
    for label in pupil_labels:
        frame_rates[label] = np.round(get_frame_rate(eid, label), 2)
    print_elapsed_time(start_time)

    print('Processing trials')
    load_trial_data(eid)
    print_elapsed_time(start_time)

    if len(fake_labels) > 0:
        print(f'Creating fake videos for {fake_labels}')
        output_file = create_fake_video(eid, fake_labels[0])
        if len(fake_labels) > 1:
            for label in fake_labels[1:]:
                shutil.copyfile(Path(output_file), Path(output_file).parent.joinpath(f'{eid}_{label}_trim.mp4'))

    if 'left' in good_labels:
        print('Processing dlc')
        load_dlc_data(eid)
        print_elapsed_time(start_time)

    print('Downscaling videos')
    for label in good_labels:
        downscale_video(eid, label, frame_rates[label])
    print_elapsed_time(start_time)

    if 'left' in good_labels:
        print('Cropping pupil video')
        crop_pupil_video(eid, frame_rates['left'])
        print_elapsed_time(start_time)

    print('Overlaying dlc video')
    for label in pupil_labels:
        overlay_dlc(eid, label, frame_rates[label])
    print_elapsed_time(start_time)

    print('Downsampling video')
    for label in pupil_labels:
        downsample_video(eid, label, frame_rates[label])
    print_elapsed_time(start_time)

    print('Trimming videos')
    for label in pupil_labels:
        trim_videos(eid, label)
    print_elapsed_time(start_time)

    print('Concatenating videos')
    concatenate_videos(eid)
    print_elapsed_time(start_time)

    print('Cleaning up')
    cleanup_data(eid)
    print_elapsed_time(start_time)


process_all(eid, one)
