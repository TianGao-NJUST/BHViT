#!/bin/bash
#SBATCH --job-name              gt
#SBATCH --cpus-per-task         16        #maximum cpu limit for each v100 GPU is 6 , each a100 GPU is 8
#SBATCH --gres                  gpu:2
#SBATCH --mem                   90G      #maximum memory limit for each v100 GPU is 90G , each a100 GPU is 40G
#SBATCH --output                output.txt
#SBATCH --partition             a100_batch 
source activate base
./scripts/run_BHViT.sh $1 &
watch -n 1 nvidia-smi >> nvidia-smi.out &
wait