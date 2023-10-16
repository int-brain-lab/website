#!/usr/bin/bash
#SBATCH --job-name="viz_data_9117969a-3f0d-478b-ad75-98263e3bfacf"
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00   # Allow it to run for 5 hours
#SBATCH -o /mnt/home/mfaulkner/Documents/viz_website/viz_data_9117969a-3f0d-478b-ad75-98263e3bfacf.log
#SBATCH --mail-type=begin   # send email when job begins
#SBATCH --mail-type=end     # send email when job ends
#SBATCH --mail-type=fail
#SBATCH --mail-user=mayo.faulkner@internationalbrainlab.org

module load python/3.10.8

/mnt/home/mfaulkner/Documents/PYTHON/envs/ibllib/bin/python /mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_data_popeye.py 9117969a-3f0d-478b-ad75-98263e3bfacf