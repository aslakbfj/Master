#!/bin/bash

#SBATCH --nodes=1			                # Use 1 node
#SBATCH --ntasks=8			                # 1 core (CPU)
#SBATCH --job-name=train_explainn           # Name of job
#SBATCH --mem=80G 			                # Default memory per CPU is 3GB
#SBATCH --partition=gpu                     # Use GPU partition
#SBATCH --gres=gpu:1                        # Use one GPU
#SBATCH --output=./slurm_analyses/job%j.log # Stdout and stderr file


source activate explainn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


 weight_file = os.listdir("CAM_filters_TF_binding/CAM_TF_num_cnns_"+str(num_cnns)+"/")[0]