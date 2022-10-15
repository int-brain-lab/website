# Folder structure

The code downloads all of the full resolution videos to a local folder (D:\ibl-website-videos)

# Download videos

Run `download_videos.py` to download the video data and downsample it

# Download trial data

Run `download_trial_data.py` to download the metadata and DLC and re-sample these to match the videos

# Crop pupil video

Run `crop_pupil_video.py` to crop the pupil video out of the original LEFT video

# Convert trial metadata

Run `convert_trial_data.py` to convert the trial metadata files to a CSV file for use with Unity 

# Convert timestamps metadata

Run `convert_timestamps_data.py` to run through all of the timestamp and DLC files and save them as NPY files for use with Unity