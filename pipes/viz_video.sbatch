#!/usr/bin/bash
#SBATCH --job-name="viz_video_0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=genx
#SBATCH --time=02:00:00   # Allow it to run for 2 hours
#SBATCH -o /mnt/home/mfaulkner/Documents/viz_video_0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63.log
#SBATCH -e /mnt/home/mfaulkner/Documents/err_0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63.log
#SBATCH --mail-type=begin   # send email when job begins
#SBATCH --mail-type=end     # send email when job ends
#SBATCH --mail-type=fail
#SBATCH --mail-user=mayo.faulkner@internationalbrainlab.org

module load python/3.10.8
module load ffmpeg

/mnt/home/mfaulkner/Documents/PYTHON/envs/ibllib/bin/python /mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_videos_popeye.py 0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63