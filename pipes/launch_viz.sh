#!/bin/bash
# File: launch_viz.sh

# Recommendation: keep scripts in $HOME, and data in ceph
projdir="$HOME/ceph"  # project directory

# Read in the pid list and get the number of pids
pid_list="$projdir/pid_list.txt"
nfiles=$(wc -l $pid_list)

# Launch a Slurm job array with $nfiles entries
# Limit to 8 jobs running at once
sbatch --array=1-$nfiles%8 job_pid.slurm $pid_list