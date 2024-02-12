#!/usr/bin/bash
#SBATCH --job-name="viz_video_f88d4dd4-ccd7-400e-9035-fa00be3bcfa8"
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00   # Allow it to run for 2 hours
#SBATCH -o /mnt/home/mfaulkner/Documents/viz_video_f88d4dd4-ccd7-400e-9035-fa00be3bcfa8.log
#SBATCH -e /mnt/home/mfaulkner/Documents/err_f88d4dd4-ccd7-400e-9035-fa00be3bcfa8.log
#SBATCH --mail-type=begin   # send email when job begins
#SBATCH --mail-type=end     # send email when job ends
#SBATCH --mail-type=fail
#SBATCH --mail-user=mayo.faulkner@internationalbrainlab.org

module load python/3.10.8
module load ffmpeg

/mnt/home/mfaulkner/Documents/PYTHON/envs/ibllib/bin/python /mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_videos_popeye.py f88d4dd4-ccd7-400e-9035-fa00be3bcfa8