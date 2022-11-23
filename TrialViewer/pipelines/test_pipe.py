import numpy as np
import pandas as pd
import abc
import copy
import shutil
import datetime
import subprocess
import time
from pathlib import Path

import scipy.interpolate as interpolate

from one.api import ONE
from one import alf
from ibllib.oneibl.data_handlers import RemoteHttpDataHandler, SDSCDataHandler
from ibllib.io.video import get_video_meta, url_from_eid
from brainbox.behavior.dlc import likelihood_threshold


one = ONE()

#ROOT_PATH = Path(r'C:\Users\Mayo\Downloads\FlatIron\dlc_test3')
# ROOT_PATH = Path('/mnt/ibl/resources/viz_videos')
ROOT_PATH = Path('/mnt/s0/Data/viz_videos')
TEMP_DATA_DIR = ROOT_PATH.joinpath('data_temp')
FINAL_DATA_DIR = ROOT_PATH.joinpath('data_final')
TEMP_VIDEO_DIR = ROOT_PATH.joinpath('video_temp')
FINAL_VIDEO_DIR = ROOT_PATH.joinpath('video_final')
NO_VIDEO_IMG = str(ROOT_PATH.joinpath('img', 'video_not_available.png'))
# NO_VIDEO_IMG = str(Path(r'C:\Users\Mayo\iblenv\website\TrialViewer\pipelines\img'))


NEW_FS = 24
NEW_WIDTH = 160
NEW_HEIGHT = 128
PAD_S = 0.5

VIDEO_PARAMS = {
    'left': {
        'fps': 60,
        'width': 1280,
        'height': 1024,
        'resolution': 2
    },
    'right': {
        'fps': 150,
        'width': 640,
        'height': 512,
        'resolution': 1
    },
    'body': {
        'fps': 30,
        'width': 640,
        'height': 512,
        'resolution': 1
    }
}

CAMERAS = ['body', 'left', 'right']
SIGNATURES = ([(f'_iblrig_{cam}Camera.raw.mp4', 'raw_video_data', True) for cam in CAMERAS] +
             [(f'_ibl_{cam}Camera.times.npy', 'alf', True) for cam in CAMERAS] +
             [(f'_ibl_{cam}Camera.dlc.pqt', 'alf', True) for cam in CAMERAS] +
             [('_ibl_wheel.position.npy', 'alf', True),
              ('_ibl_wheel.timestamps.npy', 'alf', True),
              ('_ibl_trials.table.pqt', 'alf', True)])


class VizVideo(abc.ABC):
    def __init__(self, session_path):
        self.session_path = session_path


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


def download_data_local(data_path, one):
    data_handler = RemoteHttpDataHandler(data_path, {'input_files': SIGNATURES}, one=one)
    data_handler.setUp()

    return data_handler, data_path


def download_data_flatiron(data_path, one):
    task = VizVideo(None)
    data_handler = SDSCDataHandler(task, data_path,  {'input_files': SIGNATURES}, one=one)
    data_handler.setUp()

    return data_handler, task.session_path


def check_video_status(eid, one, data_path=None):

    video_status = {}

    wheel = np.bitwise_and(len(one.list_datasets(eid, filename=f'_ibl_wheel.position.npy')) >= 1,
                          len(one.list_datasets(eid, filename=f'_ibl_wheel.timestamps.npy')) >= 1)
    for label in CAMERAS:
        video = len(one.list_datasets(eid, filename=f'_iblrig_{label}Camera.raw.mp4')) >= 1
        times = len(one.list_datasets(eid, filename=f'_ibl_{label}Camera.times.npy')) >= 1
        dlc = len(one.list_datasets(eid, filename=f'_ibl_{label}Camera.dlc.pqt')) >= 1
        status = all([wheel, video, times, dlc])
        if status:
            # Check that the timestamps and video frames match
            if data_path is None:
                video_times = one.load_dataset(eid, f'_ibl_{label}Camera.times.npy')
            else:
                video_times = np.load(data_path.joinpath('alf', f'_ibl_{label}Camera.times.npy'))
            video_meta = get_video_meta(url_from_eid(eid, label, one), one)
            if video_meta['length'] != video_times.size:
                status = False

        video_status[label] = status

    return video_status


def load_trial_data(eid, data_path):

    trials = one.load_object(eid, 'trials', attribute=['table'])
    #trials = alf.io.load_object(data_path.joinpath('alf'), 'trials')
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
    save_path = TEMP_DATA_DIR.joinpath(eid)
    start, end = np.load(save_path.joinpath(f'{eid}_start_end.npy'))
    frame_times = np.arange(0, end - start, 1 / NEW_FS) + start

    return frame_times, start, end


def load_dlc_data(eid, data_path, video_status):

    frame_times, _, _ = load_frame_times(eid)

    dlc_keys = {
        'body': {'prefix': '', 'cols': ['tail_start']},
        'left': {'prefix': 'cl_', 'cols': ['nose_tip', 'paw_l', 'paw_r', 'tube_top', 'tongue_end_l', 'tongue_end_r',
                                          'pupil_right_r', 'pupil_left_r', 'pupil_top_r', 'pupil_bottom_r']},
        'right': {'prefix': 'cr_', 'cols': ['nose_tip', 'paw_l', 'paw_r', 'tube_top', 'tongue_end_l', 'tongue_end_r']}
    }

    dlc_final = pd.DataFrame()
    dlc_final['left_ts'] = frame_times

    # if no dlc then we need to fill it with all -1 s?

    for label, label_info in dlc_keys.items():

        if video_status[label]:

            w_subsamp_factor = VIDEO_PARAMS[label]['width'] / NEW_WIDTH
            h_subsamp_factor = VIDEO_PARAMS[label]['height'] / NEW_HEIGHT

            video_data = one.load_object(eid, f'{label}Camera', collection='alf', attribute=['times', 'dlc'])
            # video_data = alf.io.load_object(data_path.joinpath('alf'), f'{label}Camera')

            dlc = pd.DataFrame()
            dlc_positions = copy.copy(video_data.dlc)

            # if left then we need to find the pupil position
            if label == 'left':
                dlc_pupil = copy.copy(video_data.dlc)
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
                df_pupil['x_mu'] = int(p_pupil_x) if not np.isnan(p_pupil_x) else p_pupil_x
                df_pupil['y_mu'] = int(p_pupil_y) if not np.isnan(p_pupil_y) else p_pupil_y
                df_pupil = pd.DataFrame.from_dict([df_pupil])
                df_pupil.to_csv(TEMP_DATA_DIR.joinpath(eid, f'{eid}_{label}_pupil_rect.csv'), index=False)

            for col in dlc_positions.keys():
                if 'pupil' in col:
                    dlc[col] = interpolate.interp1d(video_data.times, dlc_positions[col].values, fill_value=-1,
                                                    bounds_error=False)(frame_times)
                    if '_x' in col and label == 'left':
                        dlc[col] = dlc[col].values - df_pupil['x0'][0]
                    if '_y' in col and label == 'left':
                        dlc[col] = dlc[col].values - df_pupil['y0'][0]
                else:
                    if 'likelihood' in col:
                        dlc[col] = interpolate.interp1d(video_data.times, dlc_positions[col].values, fill_value=-1,
                                                        bounds_error=False)(frame_times)
                    elif '_x' in col:
                        dlc[col] = interpolate.interp1d(video_data.times, dlc_positions[col].values, fill_value=-1,
                                                        bounds_error=False)(frame_times) / w_subsamp_factor
                    elif '_y' in col:
                        dlc[col] = interpolate.interp1d(video_data.times, dlc_positions[col].values, fill_value=-1,
                                                        bounds_error=False)(frame_times) / h_subsamp_factor

            # Need to remove any that have nan value or have likelihood less that certain value
            dlc = likelihood_threshold(dlc)
            for col in dlc.columns:
                if 'likelihood' not in col:
                    dlc[col][np.isnan(dlc[col].values)] = -1

            for col in label_info['cols']:
                for suffix in ['_x', '_y']:
                    if 'pupil' in col:
                        dlc_final[col + suffix] = dlc[col + suffix]
                    else:
                        dlc_final[label_info['prefix'] + col + suffix] = dlc[col + suffix]

        else:
            for col in label_info['cols']:
                for suffix in ['_x', '_y']:
                    if 'pupil' in col:
                        dlc_final[col + suffix] = -1 * np.ones_like(frame_times)
                    else:
                        dlc_final[label_info['prefix'] + col + suffix] = -1 * np.ones_like(frame_times)

    # interpolate the wheel data at subsampled timestamps
    wheel_data = one.load_object(eid, 'wheel', collection='alf')
    #wheel_data = alf.io.load_object(data_path.joinpath('alf'), 'wheel')
    wheel_pos = interpolate.interp1d(wheel_data.timestamps, wheel_data.position, kind='linear', bounds_error=False,
                                     fill_value=wheel_data.position[0])(frame_times)
    dlc_final['wheel'] = wheel_pos

    # Save the dataframe and write out individual columns to bytes files
    dlc_final.to_csv(TEMP_DATA_DIR.joinpath(eid, 'indexes.csv'))

    for col in dlc_final.columns:
        dat = dlc_final[col].values.astype(np.float32)
        with open(FINAL_DATA_DIR.joinpath(eid, f'{eid}.{col}.bytes'), 'wb') as file:
            file.write(dat.tobytes())


def fake_videos(eid, data_path, video_status, overwrite=False):
    for label, status in video_status.items():
        scaled_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_scaled.mp4'))

        print(f'Creating fake video for {eid} {label}')
        if label == 'body':

            times, start, end = load_frame_times(eid)

            np.save(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_times.npy'), times)

            cmd = (f'ffmpeg -y -loop 1 -i {NO_VIDEO_IMG} -t {end - start} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -pix_fmt yuv420p '
                   f'-profile:v main -r {NEW_FS} {scaled_file}')
            pop = _run_command(cmd)

        else:

            times, start, end = load_frame_times(eid)
            np.save(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_times.npy'), times)
            shutil.copyfile(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_body_scaled.mp4'), Path(scaled_file))


def downsample_video(eid, data_path, video_status, overwrite=False):

    for label, status in video_status.items():
        scaled_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_scaled.mp4'))

        # If the downsampled video file already exists and overwrite is False don't reprocess
        if Path(scaled_file).exists() and not overwrite:
            print(f'Downsampled video for {eid} {label} already exists - will not overwrite')
            continue

        if status:
            print(f'Downsampling video for {eid} {label}')
            video_file = str(data_path.joinpath('raw_video_data', f'_iblrig_{label}Camera.raw.mp4'))
            times = np.load(data_path.joinpath('alf', f'_ibl_{label}Camera.times.npy'))
            np.save(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_times.npy'), times)

            framerate = 1 / np.mean(np.diff(times))

            print(f'Expected framerate: {VIDEO_PARAMS[label]["fps"]}, Detected framerate: {framerate}')

            cmd = (f'ffmpeg -y -r {framerate} -i {video_file} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -r {NEW_FS} {scaled_file}')
            pop = _run_command(cmd)

        else:
            print(f'Creating fake video for {eid} {label}')
            # create the fake video
            times, start, end = load_frame_times(eid)

            np.save(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_times.npy'), times)

            cmd = (f'ffmpeg -y -loop 1 -i {NO_VIDEO_IMG} -t {end - start} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -pix_fmt yuv420p '
                   f'-profile:v main -r {NEW_FS} {scaled_file}')
            pop = _run_command(cmd)


def crop_pupil_video(eid, data_path, video_status, overwrite=False):

    crop_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_left_crop.mp4'))

    # If the cropped video file already exists and overwrite is False don't reprocess
    if Path(crop_file).exists() and not overwrite:
        print(f'Cropped pupil video for {eid} already exists - will not overwrite')
        return

    if video_status['left']:

        print(f'Cropping pupil for {eid}')
        video_file = str(data_path.joinpath('raw_video_data', f'_iblrig_leftCamera.raw.mp4'))

        pupil_df = pd.read_csv(TEMP_DATA_DIR.joinpath(eid, f'{eid}_left_pupil_rect.csv'))

        x0 = pupil_df.x0[0]
        y0 = pupil_df.y0[0]

        if np.isnan(x0) and np.isnan(y0):
            times, start, end = load_frame_times(eid)
            cmd = (f'ffmpeg -y -loop 1 -i {NO_VIDEO_IMG} -t {end - start} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -pix_fmt yuv420p '
                   f'-profile:v main -r {NEW_FS} {crop_file}')
            pop = _run_command(cmd)

            return

        times = np.load(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_left_times.npy'))

        framerate = 1 / np.mean(np.diff(times))

        cmd = (f'ffmpeg -y -r {framerate} -i {video_file} -vf crop={NEW_WIDTH}:{NEW_HEIGHT}:{x0}:{y0},scale=160:128 -r '
               f'{NEW_FS} {crop_file}')
        pop = _run_command(cmd)

    else:
        scaled_file = TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_left_scaled.mp4')
        shutil.copyfile(scaled_file, Path(crop_file))


def ffmpeg_time_format(time):
  date = datetime.timedelta(seconds=time)
  return str(date)


def trim_videos(eid, video_status):

    # TODO check if we can incorporate into last step

    _, start, end = load_frame_times(eid)
    length_t = ffmpeg_time_format(end - start)

    for label, status in video_status.items():
        times = np.load(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_times.npy'))
        start_t = ffmpeg_time_format(start - times[0])

        video_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_scaled.mp4'))
        trim_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_scaled_trim.mp4'))

        print(times[0] - start)
        assert (times[0] <= start), f'problem with times for {label}'
            #raise(f'First timestamp {times[0]} is after video start {start} -- something went wrong?')

        cmd = (f'ffmpeg -y -i {video_file} -ss {start_t} -t {length_t} {trim_file}')
        pop = _run_command(cmd)

        # if left also trim for the cropped pupil video
        if label == 'left':
            video_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_crop.mp4'))
            trim_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_crop_trim.mp4'))

            cmd = (f'ffmpeg -y -i {video_file} -ss {start_t} -t {length_t} {trim_file}')
            pop = _run_command(cmd)


def concatenate_videos(eid, overwrite=False):

    video_inputs = [str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_left_scaled_trim.mp4')),
                    str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_right_scaled_trim.mp4')),
                    str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_body_scaled_trim.mp4')),
                    str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_left_crop_trim.mp4'))]

    video_out = str(FINAL_VIDEO_DIR.joinpath(f'{eid}.mp4'))
    if Path(video_out).exists() and not overwrite:
        print(f'Concatenated video for {eid} already exists - will not overwrite')
    else:
        cmd = (f'ffmpeg -y -i {video_inputs[0]} -i {video_inputs[1]} -i {video_inputs[2]} -i {video_inputs[3]} '
               f'-filter_complex [0:v][1:v][2:v][3:v]hstack=inputs=4[v] -map [v] {video_out}')
        pop = _run_command(cmd)


def make_eid_directories(eid):
    TEMP_DATA_DIR.joinpath(eid).mkdir(exist_ok=True, parents=True)
    FINAL_DATA_DIR.joinpath(eid).mkdir(exist_ok=True, parents=True)
    TEMP_VIDEO_DIR.joinpath(eid).mkdir(exist_ok=True, parents=True)

def make_directories():
    TEMP_DATA_DIR.mkdir(exist_ok=True, parents=True)
    FINAL_DATA_DIR.mkdir(exist_ok=True, parents=True)
    TEMP_VIDEO_DIR.mkdir(exist_ok=True, parents=True)
    FINAL_VIDEO_DIR.mkdir(exist_ok=True, parents=True)


def process_all(eid, one, location=None):

    start_time = time.time()
    session_path = one.eid2path(eid)

    make_eid_directories(eid)

    if location == 'SDSC':
        print('Downloading data')
        handler, data_path = download_data_flatiron(session_path, one)
        print_elapsed_time(start_time)
        print('Checking video status')
        video_status = check_video_status(eid, one, data_path=data_path)
        for label, status in video_status.items():
            print(f'{label}: {status}')
        print_elapsed_time(start_time)
    else:
        print('Checking video status')
        video_status = check_video_status(eid, one)
        for label, status in video_status.items():
          print(f'{label}: {status}')

        print_elapsed_time(start_time)
        print('Downloading data')
        data_path = session_path
        handler, data_path = download_data_local(session_path, one)
        print_elapsed_time(start_time)

    video_status = {'body': False, 'left': True, 'right'}

    print('Processing trials')
    load_trial_data(eid, data_path)
    print_elapsed_time(start_time)

    print('Processing dlc')
    load_dlc_data(eid, data_path, video_status)
    print_elapsed_time(start_time)

    print('Downsampling videos')
    downsample_video(eid, data_path, video_status)
    print_elapsed_time(start_time)

    print('Cropping pupil video')
    crop_pupil_video(eid, data_path, video_status)
    print_elapsed_time(start_time)

    print('Trimming videos')
    trim_videos(eid, video_status)
    print_elapsed_time(start_time)

    print('Concatenating videos')
    concatenate_videos(eid)
    print_elapsed_time(start_time)

    # handler.cleanUp()


def print_elapsed_time(start_time):
    print(f'Elapsed time: {time.time() - start_time}')


errored_eids = []
for eid in eids:
    try:
        process_all(eid, one)
    except Exception as err:
        errored_eids.append({eid: err})




# 0.2 PAD_S
['9468fa93-21ae-4984-955c-e8402e280c83',
 'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
 'a6fe44a8-07ab-49b8-81f9-e18575aa85cc',
 '0ac8d013-b91e-4732-bc7b-a1164ff3e445',
 'aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf',
 '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
 'b22f694e-4a34-4142-ab9d-2556c3487086',
 '3dd347df-f14e-40d5-9ff2-9c49f84d2157',
 '6668c4a0-70a4-4012-a7da-709660971d7a',
 '90d1e82c-c96f-496c-ad4e-ee3f02067f25',
 '36280321-555b-446d-9b7d-c2e17991e090',
 'cf43dbb1-6992-40ec-a5f9-e8e838d0f643',
 'e535fb62-e245-4a48-b119-88ce62a6fe67',
 'f25642c6-27a5-4a97-9ea0-06652db79fbd',
 '4720c98a-a305-4fba-affb-bbfa00a724a4',
 'c3d9b6fb-7fa9-4413-a364-92a54df0fc5d',
 '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
 'e8b4fda3-7fe4-4706-8ec2-91036cfee6bd',
 '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
 'e012d3e3-fdbc-4661-9ffa-5fa284e4e706',
 'd7e60cc3-6020-429e-a654-636c6cc677ea',
 '9e9c6fc0-4769-4d83-9ea4-b59a1230510e',
 '09b2c4d1-058d-4c84-9fd4-97530f85baf6',
 '56bc129c-6265-407a-a208-cc16d20a6c01',
 '1a507308-c63a-4e02-8f32-3239a07dc578',
 '6f6d2c8e-28be-49f4-ae4d-06be2d3148c1',
 '7cec9792-b8f9-4878-be7e-f08103dc0323',
 'dd4da095-4a99-4bf3-9727-f735077dba66',
 '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
 '3638d102-e8b6-4230-8742-e548cd87a949',
 '5139ce2c-7d52-44bf-8129-692d61dd6403',
 '49368f16-de69-4647-9a7a-761e94517821',
 '5285c561-80da-4563-8694-739da92e5dd0',
 'ff96bfe1-d925-4553-94b5-bf8297adf259',
 'f304211a-81b1-446f-a435-25e589fe3a5a',
 '821f1883-27f3-411d-afd3-fb8241bbc39a']

# 0.18 pad
['8b1f4024-3d96-4ee7-95f9-8a1dfd4ce4ef']


# CREATE FAKE VIDEOS COS TIMINGS ARE OFF
eids = ['e5fae088-ed96-4d9b-82f9-dfd13c259d52',
        '4d8c7767-981c-4347-8e5e-5d5fffe38534',
        'dd4da095-4a99-4bf3-9727-f735077dba66',
        'c728f6fd-58e2-448d-aefb-a72c637b604c',
        'f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1',
        'fa8ad50d-76f2-45fa-a52f-08fe3d942345',
        '09394481-8dd2-4d5c-9327-f2753ede92d7',
        'd832d9f7-c96a-4f63-8921-516ba4a7b61f',
        'dcceebe5-4589-44df-a1c1-9fa33e779727',
        '65f5c9b4-4440-48b9-b914-c593a5184a18',
        '4ddb8a95-788b-48d0-8a0a-66c7c796da96',
        '695a6073-eae0-49e0-bb0f-e9e57a9275b9',
        '09394481-8dd2-4d5c-9327-f2753ede92d7']


# TODO
# rerun because timestamps have been patched
eids = ['259927fd-7563-4b03-bc5d-17b4d0fa7a55',
'e49d8ee7-24b9-416a-9d04-9be33b655f40', - need to concatenate again
#'5139ce2c-7d52-44bf-8129-692d61dd6403',
'66d98e6e-bcd9-4e78-8fbb-636f7e808b29',
'ebc9392c-1ecb-4b4b-a545-4e3d70d23611',
'32d27583-56aa-4510-bc03-669036edad20',
#'09394481-8dd2-4d5c-9327-f2753ede92d7',
#'952870e5-f2a7-4518-9e6d-71585460f6fe',
#'695a6073-eae0-49e0-bb0f-e9e57a9275b9',
'03063955-2523-47bd-ae57-f7489dd40f15']

#all fake
'09394481-8dd2-4d5c-9327-f2753ede92d7',
'695a6073-eae0-49e0-bb0f-e9e57a9275b9',

#body fake
'952870e5-f2a7-4518-9e6d-71585460f6fe',


['09394481-8dd2-4d5c-9327-f2753ede92d7',
'4d8c7767-981c-4347-8e5e-5d5fffe38534',
'6a601cc5-7b79-4c75-b0e8-552246532f82',
'8c2f7f4d-7346-42a4-a715-4d37a5208535',
'9a629642-3a9c-42ed-b70a-532db0e86199',
'571d3ffe-54a5-473d-a265-5dc373eb7efc',
'872ce8ff-9fb3-485c-be00-bc5479e0095b',
'7082d8ff-255a-47d7-a839-bf093483ec30',
'a82800ce-f4e3-4464-9b80-4c3d6fade333',
'aad23144-0e52-4eac-80c5-c4ee2decb198',
'ac7d3064-7f09-48a3-88d2-e86a4eb86461',
'af55d16f-0e31-4073-bdb5-26da54914aa2',
'b81e3e11-9a60-4114-b894-09f85074d9c3',
'c728f6fd-58e2-448d-aefb-a72c637b604c',
'cea755db-4eee-4138-bdd6-fc23a572f5a1',
'dd4da095-4a99-4bf3-9727-f735077dba66',
'e5fae088-ed96-4d9b-82f9-dfd13c259d52',
'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d',
'f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1',
'fa8ad50d-76f2-45fa-a52f-08fe3d942345',
 '695a6073-eae0-49e0-bb0f-e9e57a9275b9',
'e5fae088-ed96-4d9b-82f9-dfd13c259d52',
        '4d8c7767-981c-4347-8e5e-5d5fffe38534',
        'dd4da095-4a99-4bf3-9727-f735077dba66',
        'c728f6fd-58e2-448d-aefb-a72c637b604c',
        'f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1',
        'fa8ad50d-76f2-45fa-a52f-08fe3d942345',
        '09394481-8dd2-4d5c-9327-f2753ede92d7',
        'd832d9f7-c96a-4f63-8921-516ba4a7b61f',
        'dcceebe5-4589-44df-a1c1-9fa33e779727',
        '65f5c9b4-4440-48b9-b914-c593a5184a18',
        '4ddb8a95-788b-48d0-8a0a-66c7c796da96',
        '695a6073-eae0-49e0-bb0f-e9e57a9275b9',
        '09394481-8dd2-4d5c-9327-f2753ede92d7']


