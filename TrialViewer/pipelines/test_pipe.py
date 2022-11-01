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

# ROOT_PATH = Path(r'C:\Users\Mayo\Downloads\FlatIron\dlc_test2')
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
PAD_S = 0.2

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

    trials = one.load_object(eid, 'trials')
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

            video_data = one.load_object(eid, f'{label}Camera', collection='alf')
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
                        dlc[col] = dlc[col] - df_pupil['x0']
                    if '_y' in col and label == 'left':
                        dlc[col] = dlc[col] - df_pupil['y0']
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
            # framerate = 1 / ((times[-1] - times[0]) / len(times))

            print(f'Expected framerate: {VIDEO_PARAMS[label]["fps"]}, Detected framerate: {framerate}')

            cmd = (f'ffmpeg -r {framerate} -i {video_file} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -r {NEW_FS} {scaled_file}')
            pop = _run_command(cmd)

            # call = subprocess.call(['ffmpeg',
            #                         '-r', f'{framerate}',
            #                         '-i', video_file,
            #                         '-vf', f'scale={NEW_WIDTH}:{NEW_HEIGHT}',
            #                         '-r', '24',
            #                         scaled_file])
        else:
            print(f'Creating fake video for {eid} {label}')
            # create the fake video
            times, start, end = load_frame_times(eid)

            np.save(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_times.npy'), times)

            cmd = (f'ffmpeg -loop 1 -i {NO_VIDEO_IMG} -t {end - start} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -r '
                   f'{NEW_FS} {scaled_file}')
            pop = _run_command(cmd)

            # call = subprocess.call(['ffmpeg',
            #                         '-loop', '1',
            #                         '-i', NO_VIDEO_IMG,
            #                         '-t', f'{end - start}',
            #                         '-vf', f'scale={NEW_WIDTH}:{NEW_HEIGHT}',
            #                         '-r', '24',
            #                         scaled_file])


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
            cmd = (f'ffmpeg -loop 1 -i {NO_VIDEO_IMG} -t {end - start} -vf scale={NEW_WIDTH}:{NEW_HEIGHT} -r '
                   f'{NEW_FS} {crop_file}')
            pop = _run_command(cmd)

            return

        times = np.load(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_left_times.npy'))

        framerate = 1 / np.mean(np.diff(times))

        cmd = (f'ffmpeg -r {framerate} -i {video_file} -vf crop={NEW_WIDTH}:{NEW_HEIGHT}:{x0}:{y0},scale=160:128 -r '
               f'{NEW_FS} {crop_file}')
        pop = _run_command(cmd)

        # call = subprocess.call(['ffmpeg',
        #                         '-r', f'{framerate}',
        #                         '-i', video_file,
        #                         '-vf', f'crop={NEW_WIDTH}:{NEW_HEIGHT}:{x0}:{y0},scale=160:128',
        #                         '-r', '24',
        #                         crop_file])
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

        if (times[0] > start):
            print(f'First timestamp {times[0]} is after video start {start} -- something went wrong?')
            continue

        cmd = (f'ffmpeg -i {video_file} -ss {start_t} -t {length_t} {trim_file}')
        pop = _run_command(cmd)

        # call = subprocess.call(['ffmpeg',
        #                         '-i', video_file,
        #                         '-ss', start_t,
        #                         '-t', length_t,
        #                         trim_file])

        # if left also trim for the cropped pupil video
        if label == 'left':
            video_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_crop.mp4'))
            trim_file = str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_{label}_crop_trim.mp4'))

            cmd = (f'ffmpeg -i {video_file} -ss {start_t} -t {length_t} {trim_file}')
            pop = _run_command(cmd)
            # call = subprocess.call(['ffmpeg',
            #                         '-i', video_file,
            #                         '-ss', start_t,
            #                         '-t', length_t,
            #                         trim_file])


def concatenate_videos(eid, overwrite=False):

    video_inputs = [str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_left_scaled_trim.mp4')),
                    str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_right_scaled_trim.mp4')),
                    str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_body_scaled_trim.mp4')),
                    str(TEMP_VIDEO_DIR.joinpath(eid, f'{eid}_left_crop_trim.mp4'))]

    video_out = str(FINAL_VIDEO_DIR.joinpath(f'{eid}.mp4'))
    if Path(video_out).exists() and not overwrite:
        print(f'Concatenated video for {eid} already exists - will not overwrite')
    else:
        cmd = (f'ffmpeg -i {video_inputs[0]} -i {video_inputs[1]} -i {video_inputs[2]} -i {video_inputs[3]} '
               f'-filter_complex [0:v][1:v][2:v][3:v]hstack=inputs=4[v] -map [v] {video_out}')
        pop = _run_command(cmd)

        # call = subprocess.call(['ffmpeg',
        #                         '-i', video_inputs[0], '-i', video_inputs[1], '-i', video_inputs[2], '-i', video_inputs[3],
        #                         '-filter_complex', '[0:v][1:v][2:v][3:v]hstack=inputs=4[v]',
        #                         '-map', '[v]',
        #                         video_out])

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
        #video_status = {'body': True, 'left': True, 'right': True}
        print_elapsed_time(start_time)
        print('Downloading data')
        data_path = session_path
        #handler, data_path = download_data_local(session_path, one)
        print_elapsed_time(start_time)

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

    handler.cleanUp()


def print_elapsed_time(start_time):
    print(f'Elapsed time: {time.time() - start_time}')


errored_eids = []
for eid in eids:
    try:
        process_all(eid, one)
    except Exception as err:
        errored_eids.append(eid)



['5569f363-0934-464e-9a5b-77c8e67791a1', # True
 '16693458-0801-4d35-a3f1-9115c7e5acfd', # False
 '8c552ddc-813e-4035-81cc-3971b57efe65', #True
 '07dc4b76-5b93-4a03-82a0-b3d9cc73f412', # True
 '0ac8d013-b91e-4732-bc7b-a1164ff3e445', # True
 '465c44bd-2e67-4112-977b-36e1ac7e3f8c', # True
 '6c6b0d06-6039-4525-a74b-58cfaa1d3a60' # True]
