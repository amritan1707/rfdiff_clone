#!/bin/bash

#SBATCH -J rfdiff_motifscaffolding
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 00-01:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

# Here, we're running one of the motif-scaffolding benchmark examples
# Specifically, we're scaffolding site 5 from RSV-F protein
# We specify the output path and input pdb (the RSV-F protein)
# We specify the protein we want to build, with the contig input:
#   - 10-40 residues (randomly sampled)
#   - residues 163-181 (inclusive) on the A chain of the input
#   - 10-40 residues (randomly sampled)
# We generate 10 designs

source ~/.bashrc
conda activate rf_diff
python /proj/kuhl_lab/RFdiffusion/scripts/run_inference.py \
inference.output_prefix=example_outputs/rfdiff_motifscaffolding/rfdiff_model \
inference.input_pdb=input_pdbs/5TPN.pdb \
'contigmap.contigs=[10-40/A163-181/10-40]' \
inference.num_designs=10
