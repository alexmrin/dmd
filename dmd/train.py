"""
Main training entry point for DMD training.
Part of this file is taken and adopted from devrimcavusoglu/std. See
the original file below
https://github.com/devrimcavusoglu/std/blob/main/std/main.py
"""

import datetime
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from neptune import Run
from torch.backends import cudnn
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from accelerate import Accelerator
from accelerate.utils import set_seed

from dmd import NEPTUNE_CONFIG_PATH, PROJECT_ROOT
from dmd.dataset.cifar_pairs import CIFARPairs
from dmd.fid import FID
from dmd.loss import DenoisingLoss, GeneratorLoss
from dmd.modeling_utils import load_edm, load_dmd_model
from dmd.training.training_loop import train_one_epoch
from dmd.utils.common import create_experiment, seed_everything
from dmd.utils.logging import CheckpointHandler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    from timm.utils import ApexScaler

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from fvcore.nn import FlopCountAnalysis, flop_count, flop_count_table, parameter_count
    from utils import sfc_flop_jit

    has_fvcore = True
except ImportError:
    has_fvcore = False


def train(
    generator: torch.nn.Module,
    mu_fake: torch.nn.Module,
    mu_real: torch.nn.Module,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    loss_g: TorchLoss,
    loss_d: TorchLoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    accelerator: Accelerator,
    max_norm: float = 10,
    amp_autocast=None,
    neptune_run: Optional[Run] = None,
    cudnn_benchmark: bool = True,
    is_distributed: bool = False,
    print_freq: int = 10,
    im_save_freq: int = 300,
    checkpoint_handler: Optional[CheckpointHandler] = None,
):
    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    if cudnn_benchmark:
        cudnn.benchmark = True

    fid = FID(data_loader_test, device=device)

    for epoch in range(epochs):
        if is_distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            generator,
            mu_fake,
            mu_real,
            data_loader_train,
            loss_g,
            loss_d,
            optimizer_g,
            optimizer_d,
            device,
            epoch,
            max_norm=max_norm,
            neptune_run=neptune_run,
            output_dir=checkpoint_handler.checkpoint_dir,
            print_freq=print_freq,
            im_save_freq=im_save_freq,
            accelerator=accelerator
        )

        # lr_scheduler.step(epoch)
        model_dict = {
            "model_g": generator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "model_d": mu_fake.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            # "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            # "model_ema": get_state_dict(model_ema),
            # "args": args,
        }
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            # **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }
        test_fid = fid(generator)
        if neptune_run is not None:
            neptune_run["test/fid"].append(test_fid)
        print(f"Test FID: {test_fid}")
        log_stats["test_fid"] = test_fid
        checkpoint_handler.save(model_dict, log_stats, test_fid, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def run(
    model_path: str,
    data_path: str,
    epochs: int,
    output_dir: str = None,
    batch_size: int = 56,
    eval_batch_size: int = 128,
    num_workers: int = 1,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    dmd_loss_timesteps: int = 1000,
    dmd_loss_lambda: float = 0.25,
    log_neptune: bool = False,
    neptune_run_id: Optional[str] = None,
    resume_from_checkpoint: bool = False,
    cudnn_benchmark: bool = True,
    max_norm: float = 10.0,
    print_steps: int = 10,
    im_save_steps: int = 300,
    model_save_steps: int = 1600,
    seed: int = 42,
) -> None:
    if not resume_from_checkpoint and output_dir is None:
        raise ValueError("`output_dir` must be given when `resume_from_checkpoint` is `False`.")
    if resume_from_checkpoint and output_dir is None:
        warnings.warn("`output_dir` is set to `model_path` when `resume_from_checkpoint` is `True`.")
        output_dir = Path(model_path).parent
    output_dir = Path(output_dir)
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Use accelerator's set_seed instead
    set_seed(seed)

    # Prepare dataloader
    data_path = Path(data_path).resolve()
    training_dataset = CIFARPairs(data_path)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if accelerator.is_main_process:
        test_dataset = CIFAR10(
            root=(PROJECT_ROOT / "data").as_posix(), train=True, download=True, transform=transforms.ToTensor()
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        test_dataset = CIFAR10(
            root=(PROJECT_ROOT / "data").as_posix(), train=True, download=False, transform=transforms.ToTensor()
        )
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay, "betas": betas}
    mu_real = load_edm(model_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl", device=device)
    if resume_from_checkpoint:
        generator, generator_optimizer, mu_fake, diffuser_optimizer = load_dmd_model(model_path=model_path, device=device, for_training=True, optimizer_kwargs=optimizer_kwargs)
    else:
        mu_fake = load_edm(model_path=model_path, device=device)
        generator = load_edm(model_path=model_path, device=device)

        # Create optimizers
        generator_optimizer = AdamW(params=generator.parameters(), **optimizer_kwargs)
        diffuser_optimizer = AdamW(params=mu_fake.parameters(), **optimizer_kwargs)

    # Create losses
    generator_loss = GeneratorLoss(timesteps=dmd_loss_timesteps, lambda_reg=dmd_loss_lambda)
    diffusion_loss = DenoisingLoss()

    # Prepare for distributed training
    generator, mu_fake, mu_real, generator_optimizer, diffuser_optimizer, train_loader, test_loader = accelerator.prepare(
        generator, mu_fake, mu_real, generator_optimizer, diffuser_optimizer, train_loader, test_loader
    )

    checkpoint_handler = CheckpointHandler(
        checkpoint_dir=output_dir, lower_is_better=True
    )

    neptune_run = None
    if log_neptune and accelerator.is_main_process:
        neptune_run = create_experiment(NEPTUNE_CONFIG_PATH, run_id=neptune_run_id)
        neptune_run["training_args"] = {
            "model_path": model_path,
            "data_path": data_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "dmd_loss_timesteps": dmd_loss_timesteps,
            "dmd_loss_lambda": float(dmd_loss_lambda),
            "device": str(device),
            "cudnn_benchmark": cudnn_benchmark,
            "max_norm": max_norm,
            "print_steps": print_steps,
            "im_save_steps": im_save_steps,
            "model_save_steps": model_save_steps,
        }

    # start training
    train(
        generator=generator,
        mu_real=mu_real,
        mu_fake=mu_fake,
        data_loader_train=train_loader,
        data_loader_test=test_loader,
        device=device,
        loss_g=generator_loss,
        loss_d=diffusion_loss,
        optimizer_g=generator_optimizer,
        optimizer_d=diffuser_optimizer,
        epochs=epochs,
        neptune_run=neptune_run,
        cudnn_benchmark=cudnn_benchmark,
        max_norm=max_norm,
        print_freq=print_steps,
        im_save_freq=im_save_steps,
        checkpoint_handler=checkpoint_handler,
        accelerator=accelerator,
    )

    if neptune_run and accelerator.is_main_process:
        neptune_run.stop()