#!/bin/bash
#SBATCH --job-name=bayesian          # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sathish.kabatkar_ravindranth@boehringer-ingelheim.com    # Where to send mail 
#SBATCH --nodes=1                    # Run all processes on a single node  
#SBATCH --ntasks=1                   # Run a single task       
#SBATCH --cpus-per-task=48           # Number of CPU cores per task
#SBATCH --mem=48GB                   # Job memory request
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=job_%j.log          # Standard output and error log
pwd; date

echo "Logistic regression on PE dataset"

srun hostname
srun sleep 30
source tryenv/bin/activate
srun sleep 30
srun python Models/Freq_elasticnet_reg.py
date
