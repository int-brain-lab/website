#!/usr/bin/bash
#SBATCH --job-name="video_data_zzPIDzz"
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00   # Allow it to run for max 2 hours
#SBATCH -o /mnt/home/mfaulkner/Documents/video_logs/viz_data_zzEIDzz.log
#SBATCH -e /mnt/home/mfaulkner/Documents/video_logs/err_zzEIDzz.log
#SBATCH --mail-type=fail
#SBATCH --mail-user=mayo.faulkner@internationalbrainlab.org

module load python/3.10.8
module load ffmpeg

/mnt/home/mfaulkner/Documents/PYTHON/envs/ibllib/bin/python /mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_videos_popeye.py zzEIDzz