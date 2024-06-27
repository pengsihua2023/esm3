#!/bin/bash
#SBATCH --job-name=ESM3           # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name, i.e., gpu_p 
#SBATCH --gres=gpu:A100:1             # Requests one GPU device 
#SBATCH --ntasks=64                    # Run a single task      
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem=900gb                    # Job memory request
#SBATCH --time=7-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=ESM3.%j.out         # Standard output log
#SBATCH --error=ESM3.%j.err          # Standard error log
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sp96859@uga.edu  # Where to send mail


cd $SLURM_SUBMIT_DIR
ml Miniconda3/23.5.2-0
module load CUDA/11.7.0

export HUGGINGFACE_HUB_TOKEN=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


echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ESM3

/home/sp96859/.conda/envs/ESM3/bin/python sihua-esm3.py
