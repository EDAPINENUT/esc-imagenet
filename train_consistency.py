import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from diffusers.models import AutoencoderKL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from sit import SiT_models
from loss.consistency_loss import ConsistencyLoss

from dataset import LMDBLatentsDataset
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import math
import swanlab
from swanlab.integration.accelerate import SwanLabTracker
from utils import array2grid

logger = get_logger(__name__)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def ema_and_scale_fn(step, total_steps, start_ema, start_scales, end_scales):
    """
    Compute the ema and scale for the current step as in consistency models
    """
    scales = np.ceil(
        np.sqrt(
            (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
            + start_scales**2
        )
        - 1
    ).astype(np.int32)
    scales = np.maximum(scales, 1)
    c = -np.log(start_ema) * start_scales
    target_ema = np.exp(-c / scales)
    scales = scales + 1
    return float(target_ema), int(scales)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

@torch.no_grad()
def update_ema_distill(ema_model, tgt_model, model, ema_decay=0.99995, tgt_decay=0.):
    """
    Step the EMA model towards the current model.
    """
    update_ema(ema_model, model, ema_decay)
    if tgt_decay > 0:
        update_ema(tgt_model, model, tgt_decay)
    else:
        if hasattr(model, 'module'):
            tgt_model.load_state_dict(model.module.state_dict())
        else:
            tgt_model.load_state_dict(model.state_dict())
    
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # Set accelerator
    if args.debug:
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    swanlab.login("4YqR9oTwsE51dPgeM84eg")
    tracker = SwanLabTracker(
        project_name="esc-imagenet",
        run_name="esc-imagenet",
        config=args,
        experiment_name=args.exp_name,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        log_with=tracker
    )

    accelerator.init_trackers(
        project_name="esc-imagenet",
        config=vars(copy.deepcopy(args))
    )

    torch.backends.cudnn.benchmark = True
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        
        # Log all args for reference
        logger.info("Training arguments:")
        for arg, value in sorted(args_dict.items()):
            logger.info(f"  {arg}: {value}")
            
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    
    # Define block_kwargs from args
    block_kwargs = {
        "fused_attn": False,
        "qk_norm": False,
    }

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model_tgt = deepcopy(model).to(device)
    requires_grad(model_tgt, False) # For torch jvp
    
    # Create loss function with all MeanFlow parameters
    loss_fn = ConsistencyLoss(
        path_type=args.path_type, 
        # Add MeanFlow specific parameters
        time_sampler=args.time_sampler,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        loss_type=args.loss_type,
        adaptive_p=args.adaptive_p,
        cfg_omega=args.cfg_omega,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = LMDBLatentsDataset(args.data_dir, flip_prob=0.5)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
    args.max_train_steps = args.epochs * steps_per_epoch // accelerator.num_processes
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    update_ema(model_tgt, model, decay=0)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    model_tgt.eval() 
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
        )
        model.load_state_dict(ckpt['model'])
        model_tgt.load_state_dict(ckpt['model_tgt'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # here is a trick from IMM. https://github.com/lumalabs/imm/blob/main/training/encoders.py
    latents_scale = torch.tensor(
        [0.18125, 0.18125, 0.18125, 0.18125]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    local_path = './ckpt/stabilityai/sd-vae-ft-ema'
    if not os.path.exists(local_path):
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(local_path).to(device)
    vae.eval()
    with torch.no_grad():
        sample_batch_size = 64
        gt_raw_latents, _ = next(iter(train_dataloader))
        while gt_raw_latents.size(0) < sample_batch_size:
            additional_latents, _ = next(iter(train_dataloader))
            gt_raw_latents = torch.cat([gt_raw_latents, additional_latents], dim=0)
        gt_raw_latents = gt_raw_latents[:sample_batch_size]
        posterior = DiagonalGaussianDistribution(gt_raw_latents)
        gt_raw_latents = posterior.sample()
        gt_raw_latents = gt_raw_latents * latents_scale + latents_bias
        gt_raw_images = vae.decode((gt_raw_latents - latents_bias) / latents_scale).sample
        gt_raw_images = (gt_raw_images + 1) / 2.
        gt_raw_images = array2grid(gt_raw_images.detach().cpu())
        if accelerator.is_main_process:
            sample_path = f"{save_dir}/samples/"
            os.makedirs(sample_path, exist_ok=True)
            Image.fromarray(gt_raw_images).save(f"{sample_path}/gt.png")
        accelerator.log({
            "image/gt": swanlab.Image(gt_raw_images)
        })
    
    z_fake = torch.randn_like(gt_raw_latents)
    y_fake = torch.randint(0, args.num_classes, (sample_batch_size,), device=device)

    for epoch in range(args.epochs):
        model.train()
        for moments, labels in train_dataloader:
            moments = moments.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                posterior = DiagonalGaussianDistribution(moments)
                x = posterior.sample()
                x = x * latents_scale + latents_bias

            tgt_decay, scales = ema_and_scale_fn(
                global_step, args.max_train_steps, args.ema_start, args.scale_min, args.scale_max
            )
            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels, scales=scales)
                loss, loss_ref = loss_fn(model, model_tgt, x, model_kwargs)
                loss_mean = loss.mean()
                loss_mean_ref = loss_ref.mean()
                loss = loss_mean                
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema_distill(ema, model_tgt, model, ema_decay=args.ema_decay, tgt_decay=tgt_decay)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0 or global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "model_tgt": model_tgt.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from sampler import cfg_sampler
                with torch.no_grad():
                    samples = cfg_sampler(ema, z_fake, num_steps=1, cfg_scale=1.0, y=y_fake, scheduler=loss_fn.flow_scheduler)
                    samples = vae.decode((samples - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    samples = array2grid(samples.detach().cpu())
                    if accelerator.is_main_process:
                        Image.fromarray(samples).save(f"{sample_path}/{global_step:07d}.png")
                    accelerator.log({
                        "image/samples": swanlab.Image(samples)
                    })

            logs = {
                "loss_ref": accelerator.gather(loss_mean_ref).mean().detach().item(), 
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs)
            
            # Log to file periodically
            if accelerator.is_main_process and global_step % 100 == 0:
                logger.info(f"Step {global_step}: mse = {logs['loss_ref']:.4f}, grad_norm = {logs['grad_norm']:.4f}")

            if global_step >= args.max_train_steps:
                break
        
        # Log epoch completion
        if accelerator.is_main_process:
            logger.info(f"Completed epoch {epoch+1}/{args.epochs}")
            
        if global_step >= args.max_train_steps:
            break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="MeanFlow Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, default="cm-debug")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

    # model
    parser.add_argument("--model", type=str, default="SiT-B/4")
    parser.add_argument("--num-classes", type=int, default=1000)

    # dataset
    parser.add_argument("--data-dir", type=str, default="/wutailin/image_data/imagenet_vq_lmdb/train")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--checkpointing-steps", type=int, default=20000)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # basic loss
    parser.add_argument("--path-type", type=str, default="edm", choices=["edm", "linear", "cosine"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--loss-type", default="adaptive", type=str, choices=["uniform", "adaptive"], help="Loss weighting type")
    
    # MeanFlow specific parameters
    parser.add_argument("--time-sampler", type=str, default="progressive", choices=["progressive"], 
                       help="Time sampling strategy")
    parser.add_argument("--sigma-max", type=float, default=80.0, help="sigma_max in edm")
    parser.add_argument("--sigma-min", type=float, default=0.002, help="sigma_min in edm")
    parser.add_argument("--scale-max", type=float, default=200, help="scale_max in edm")
    parser.add_argument("--scale-min", type=float, default=2, help="scale_min in edm")
    parser.add_argument("--ema-start", type=float, default=0.9, help="ema_start in edm")
    parser.add_argument("--adaptive-p", type=float, default=1.0, help="Power param for adaptive weighting")
    parser.add_argument("--cfg-omega", type=float, default=1.5, help="CFG omega param, default 1.0 means no CFG")
    
    # ESC specific parameters
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--tgt-decay", type=float, default=0.0, help="Target model decay rate")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
