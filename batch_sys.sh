#!/bin/bash
# Geoff Dolinger
#
# Example with an array of experiments
#  The --array line says that we will execute 4 experiments (numbered 0,1,2,3).
#   You can specify ranges or comma-separated lists on this line
#  For each experiment, the SLURM_ARRAY_TASK_ID will be set to the experiment number
#   In this case, this ID is used to set the name of the stdout/stderr file names
#   and is passed as an argument to the python program
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=8G
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results/sys_exp%04a_stdout.txt
#SBATCH --error=results/sys_exp%04a_stderr.txt
#SBATCH --time=08:00:00
#SBATCH --job-name=CFAR_SYS
#SBATCH --mail-user=geoffrey.dolinger@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/gdoli86/cfar
#SBATCH --array=0-19
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf
hostname 

python System_Parser_SC.py @sys.txt --exp_index $SLURM_ARRAY_TASK_ID
#python System_Parser_SC.py @sys.txt --exp_index 0
