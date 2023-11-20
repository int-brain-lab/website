# File: job.slurm

# the file with the list of files to process
pid_list=$1

# the job array index
# the task ID is automatically set by Slurm
i=$SLURM_ARRAY_TASK_ID

# Get the pid for this job
pid =$(sed -n "${i}p" ${pid_list})

# Slurm params
#SBATCH -p ccX      # or "-p genx" if your job won't fill a node
#SBATCH --job-name="viz_data_$pid"
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00   # Allow it to run for max 2 hours
#SBATCH -o /mnt/home/mfaulkner/Documents/viz_website/viz_data_$pid.log
#SBATCH -e /mnt/home/mfaulkner/Documents/viz_website/err_$pid.log
#SBATCH --mail-type=begin   # send email when job begins
#SBATCH --mail-type=end     # send email when job ends
#SBATCH --mail-type=fail
#SBATCH --mail-user=mayo.faulkner@internationalbrainlab.org

module load python/3.10.8

echo "About to process $pid"
/mnt/home/mfaulkner/Documents/PYTHON/envs/ibllib/bin/python /mnt/home/mfaulkner/Documents/PYTHON/website/pipes/viz_data_popeye.py $pid
