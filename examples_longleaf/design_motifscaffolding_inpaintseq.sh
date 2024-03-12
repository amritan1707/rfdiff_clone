#!/bin/bash

#SBATCH -J rfdiff_motifscaffolding_inpaintseq
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 00-01:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

# Here, we're running one of the motif-scaffolding benchmark examples
# Specifically, we're scaffolding site 5 from RSV-F protein
# This is equivalent to the example in design_motifscaffolding.sh, except that we're adding the contigmap.inpaint_seq flag
# The logic here is that only some of the amino acids are important for the function of the motif, and others can be redesigned.
# This can promote better packing, as RFdiffusion is not forced to pack against specific (but unimportant) residues on the back-side of the motif  
# We specify the output path and input pdb (the RSV-F protein)
# We specify the protein we want to build, with the contig input:
#   - 10-40 residues (randomly sampled)
#   - residues 163-181 (inclusive) on the A chain of the input
#   - 10-40 residues (randomly sampled)
# We generate 10 designs
# We then specify that residues 163-168 (inclusive), 170-171 (inclusive) and 179 (inclusive) on the A chain of the input, should be masked in the input

source ~/.bashrc
conda activate rf_diff
python /proj/kuhl_lab/RFdiffusion/scripts/run_inference.py \
inference.output_prefix=example_outputs/rfdiff_motifscaffolding_inpaintseq/rfdiff_model \
inference.input_pdb=input_pdbs/5TPN.pdb \
'contigmap.contigs=[10-40/A163-181/10-40]' \
inference.num_designs=10 \
'contigmap.inpaint_seq=[A163-168/A170-171/A179]'
