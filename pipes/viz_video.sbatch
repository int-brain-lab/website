#!/usr/bin/bash
#SBATCH --job-name="viz_video_fa8ad50d-76f2-45fa-a52f-08fe3d942345"
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00   # Allow it to run for 2 hours
#SBATCH -o /mnt/home/mfaulkner/Documents/viz_video_fa8ad50d-76f2-45fa-a52f-08fe3d942345.log
#SBATCH -e /mnt/home/mfaulkner/Documents/err_fa8ad50d-76f2-45fa-a52f-08fe3d942345.log
#SBATCH --mail-type=begin   # send email when job begins
#SBATCH --mail-type=end     # send email when job ends
#SBATCH --mail-type=fail
#SBATCH --mail-user=mayo.faulkner@internationalbrainlab.org

module load python/3.10.8
module load ffmpeg

/mnt/home/mfaulkner/Documents/PYTHON/envs/ibllib/bin/python /mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_videos_popeye.py fa8ad50d-76f2-45fa-a52f-08fe3d942345