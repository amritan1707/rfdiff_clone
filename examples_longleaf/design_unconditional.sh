#!/bin/bash

#SBATCH -J rfdiff_unconditional
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 00-01:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

# Here, we're making some unconditional designs
# We specify the path for the outputs
# We tell RFdiffusion that designs should be 100-200 residues in length (randomly sampled each design)
# We generate 10 such designs

source ~/.bashrc
conda activate rf_diff
python /proj/kuhl_lab/RFdiffusion/scripts/run_inference.py \
inference.output_prefix=example_outputs/rfdiff_unconditional/rfdiff_model \
'contigmap.contigs=[100-200]' \
inference.num_designs=10
