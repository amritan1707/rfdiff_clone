#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re
import os, time, pickle, sys
import torch
from omegaconf import OmegaConf
import hydra
import logging
sys.path.append("/work/users/a/m/amritan/")
from rfdiff_clone.rfdiffusion.util import writepdb_multi, writepdb
from rfdiff_clone.rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
import time


def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(config_path, input_pdb_path, contigs=None, contigs_2=None, provide_seq=None, num_designs=10, design_run=False):
    start_time = time.time()
    conf = OmegaConf.load(config_path)
    print("done loading config")
    
    time1 = time.time()
    config_time = time1 - start_time
    print(f"Config loaded in {config_time:.2f} seconds")

    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler and target/contig.
    print("Initializing sampler")
    sampler = iu.sampler_selector(conf)

    time2 = time.time()
    sampler_time = time2 - time1
    print(f"Sampler initialized in {sampler_time:.2f} seconds")
    
    # Make sure out_prefix has a base directory
    if not design_run and os.path.dirname(sampler.inf_conf.output_prefix) == "":
        sampler.inf_conf.output_prefix = os.path.join(".", sampler.inf_conf.output_prefix)
    
    # Loop over number of designs to sample.
    print("input_pdb_path", input_pdb_path)
    sampler.inf_conf.input_pdb = input_pdb_path

    #print(sampler.contig_conf)
    if contigs is not None:
        sampler.contig_conf.contigs = contigs
    if contigs_2 is not None:
        sampler.extra.contigs_2 = contigs_2
    if provide_seq is not None:
        sampler.contig_conf.provide_seq = provide_seq
    
    sampler.inf_conf.num_designs = num_designs
    
    time3 = time.time()
    path_time = time3 - time2
    #print(f"Paths initialized in {path_time:.2f} seconds")
    
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match(".*_(\d+)\.pdb$", e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1
        
    time4 = time.time()
    pdb_time = time4 - time3
    #print(f"PDBs initialized in {pdb_time:.2f} seconds")

    pdbs, trbs, traj_xts, traj_x0s = [], [], [], []
    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue

        #read input pdb file here and generate features
        x_init, seq_init = sampler.sample_init()
        print("finished sample_init")
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        # Loop over number of reverse diffusion time steps.
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            if sampler.contig_conf.contigs_2 is not None:
                if t == int(sampler.t_step_input_2):
                    print("at time step " + str(t) + ": combining masks")
                    sampler.mask_str[0,:] = torch.logical_and(sampler.mask_str.squeeze(), sampler.mask_str_2.squeeze())
                    sampler.diffusion_mask = sampler.mask_str
            #   sampler.
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])  # remove singleton leading dimension

        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(
            denoised_xyz_stack,
            [
                0,
            ],
        )
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(
            px0_xyz_stack,
            [
                0,
            ],
        )

        # For logging -- don't flip
        plddt_stack = torch.stack(plddt_stack)

        # Save outputs
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]

        # Output glycines, except for motif region
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )  # 7 is glycine

        bfacts = torch.ones_like(final_seq.squeeze())
        # make bfact=0 for diffused coordinates
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        # pX0 last step
        if design_run:
            out = None
        else:
            out = f"{out_prefix}.pdb"

        # Now don't output sidechains
        pdb = writepdb(
            out,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
            design_run = design_run
        )

        # run metadata
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "CPU",
            time=time.time() - start_time,
        )
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        if not design_run:
            with open(f"{out_prefix}.trb", "wb") as f_out:
                pickle.dump(trb, f_out)

        if sampler.inf_conf.write_trajectory:
            # trajectory pdbs
            traj_prefix = (
                os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            )
            
            if not design_run:
                os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
                xt_out = f"{traj_prefix}_Xt-1_traj.pdb"
                x0_out = f"{traj_prefix}_pX0_traj.pdb"
            else:
                xt_out = None
                x0_out = None              
            
            traj_xt = writepdb_multi(
                xt_out, 
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
                design_run=design_run,
            ) 
            
            traj_x0 = writepdb_multi(
                x0_out, 
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
                design_run=design_run,
            )

        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")
        print(f"Finished design in {(time.time()-start_time):.2f} seconds")
        pdbs.append(pdb)
        trbs.append(trb)
        traj_xts.append(traj_xt)
        traj_x0s.append(traj_x0)
    
    return pdbs, trbs, traj_xts, traj_x0s

if __name__ == "__main__":
    config_path = "./partial.yaml"
    input_pdb_path = "./8tim.pdb"
    a, b, c, d = main(config_path, input_pdb_path, contigs=None, design_run=False, num_designs=100)
    print(len(a))