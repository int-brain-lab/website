# Folder structure

The code downloads all of the full resolution videos to a local folder (D:\ibl-website-videos)

# Convert session table to list of selectable pids

Run `get_selectable_pids.py`

# Download trial data

Run `download_trial_data.py` to download the metadata and DLC and re-sample these to match the videos

# Download videos

Run `download_videos.py` to download the video data and downsample it

# Download dlc data

Run `download_dlc_data.py` to download the metadata and DLC and re-sample these to match the videos

# Crop pupil video

Run `crop_pupil_video.py` to crop the pupil video out of the original LEFT video

# Trim and concatenate videos

Run `trim_concat_videos.py`

# Push videos to server

Copy all the final video files from `ibl-website-videos/final` to the server at `/var/www/ibl_website/trialviewer_data/WebGL/`

# Convert trial metadata

Run `convert_trial_data.py` to convert the trial metadata files to a CSV file for use with Unity 

# Convert timestamps metadata

Run `convert_timestamps_data.py` to run through all of the timestamp and DLC files and save them as NPY files for use with Unity

# Push trial and timestamp data to Unity

Copy the entire `final` folder from the pipelines output to the Unity `AddressablesAssets` folder and re-build the Addressables. Then copy the **catalog** and **.bundle** files to the server at `/var/www/ibl_website/trialviewer_data/WebGL/`

After re-building Addressables you have to re-build the WebGL build as well and re-deploy that to the server at `/var/www/ibl_website/trialviewer_data/TrialViewerBuild/`
