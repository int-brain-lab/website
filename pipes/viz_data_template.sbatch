#!/usr/bin/bash
#SBATCH --job-name="viz_data_zzPIDzz"
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00   # Allow it to run for max 2 hours
#SBATCH -o /mnt/home/mfaulkner/Documents/viz_logs/viz_data_zzPIDzz.log
#SBATCH -e /mnt/home/mfaulkner/Documents/viz_logs/err_zzPIDzz.log
#SBATCH --mail-type=begin   # send email when job begins
#SBATCH --mail-type=end     # send email when job ends
#SBATCH --mail-type=fail
#SBATCH --mail-user=mayo.faulkner@internationalbrainlab.org

module load python/3.10.8

/mnt/home/mfaulkner/Documents/PYTHON/envs/ibllib/bin/python /mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_data_popeye.py zzPIDzz