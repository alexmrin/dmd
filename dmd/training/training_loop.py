"""
Main training loop.
Part of this file taken and adopted from devrimcavusoglu/std repository. See
the original file below
https://github.com/devrimcavusoglu/std/blob/main/std/engine.py
"""
import math
import sys
from contextlib import suppress
from pathlib import Path
from typing import List, Optional

import torch
from neptune import Run
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from accelerate import Accelerator

from dmd.modeling_utils import encode_labels, forward_diffusion, get_fixed_generator_sigma
from dmd.utils.array import torch_to_pillow
from dmd.utils.common import image_grid
from dmd.utils.logging import MetricLogger


def _save_intermediate_images(
    output_dir: str,
    all_images: List[torch.Tensor],
    prefix: str,
):
    pims = []
    for images in all_images:
        pims.extend(torch_to_pillow(images))
    grid = image_grid(pims, rows=len(all_images), cols=all_images[0].shape[0])
    images_output_dir = Path(output_dir)
    images_output_dir.mkdir(exist_ok=True, parents=True)
    image_path = images_output_dir / f"{prefix}.png"
    grid.save(image_path)
    return grid


def update_parameters(model, loss, optimizer, max_norm):
    optimizer.zero_grad()

    # this attribute is added by timm on one optimizer (adahessian)
    is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    loss.backward(create_graph=is_second_order)
    if max_norm is not None:
        clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()


def train_one_epoch(
    generator: torch.nn.Module,
    mu_fake: torch.nn.Module,
    mu_real: torch.nn.Module,
    data_loader_train: DataLoader,
    loss_g: TorchLoss,
    loss_d: TorchLoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    output_dir: str = None,
    max_norm: float = 10,
    neptune_run: Optional[Run] = None,
    print_freq: int = 10,
    im_save_freq: int = 300,
    accelerator: Accelerator = None,
):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Set G and mu_fake to train mode, mu_real should be frozen
    generator.requires_grad_(True).train()
    mu_fake.requires_grad_(True).train()
    mu_real.requires_grad_(False).eval()

    metric_logger = MetricLogger(delimiter="  ", neptune_run=neptune_run)
    header = "Epoch: [{}]".format(epoch)

    i = 0
    for pairs in metric_logger.log_every(data_loader_train, print_freq, header):
        y_ref = pairs["image"].to(device, non_blocking=True).to(torch.float32).clip(-1, 1)
        z_ref = pairs["latent"].to(device, non_blocking=True).to(torch.float32)
        z = torch.randn_like(y_ref, device=device)
        generator_sigma = get_fixed_generator_sigma(z.shape[0], device=device)
        z = z * generator_sigma[0, 0]
        z_ref = z_ref * generator_sigma[0, 0]
        class_idx = pairs["class_id"].to(device, non_blocking=True)
        class_ids = encode_labels(class_idx, generator.label_dim)

        # Update generator
        x = generator(z, generator_sigma, class_labels=class_ids)
        x_ref = generator(z_ref, generator_sigma, class_labels=class_ids)
        l_g = loss_g(mu_real, mu_fake, x, x_ref, y_ref, class_ids)
        if not math.isfinite(l_g.item()):
            print(f"Generator Loss is {l_g.item()}, stopping training")
            sys.exit(1)

        accelerator.backward(l_g)
        if max_norm is not None:
            accelerator.clip_grad_norm_(generator.parameters(), max_norm)
        optimizer_g.step()
        optimizer_g.zero_grad()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metric_logger.log_neptune("loss_g", l_g.item())

        # Update mu_fake
        t = torch.randint(1, 1000, [x.shape[0]])
        l_d = loss_d(mu_fake, x, t, class_ids)
        if not math.isfinite(l_d.item()):
            print(f"Diffusion Loss is {l_d.item()}, stopping training")
            sys.exit(1)

        accelerator.backward(l_d)
        if max_norm is not None:
            accelerator.clip_grad_norm_(mu_fake.parameters(), max_norm)
        optimizer_d.step()
        optimizer_d.zero_grad()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metric_logger.log_neptune("loss_d", l_d.item())

        if i % im_save_freq == 0 and accelerator.is_main_process:
            images_epoch_dir = images_dir / f"epoch_{epoch}"
            images_epoch_dir.mkdir(exist_ok=True)
            with torch.no_grad():
                x_t, sigma_t = forward_diffusion(x, t)
                real_pred = mu_real(x_t, sigma_t, class_labels=class_ids)
                fake_pred = mu_fake(x_t, sigma_t, class_labels=class_ids)
            grid = _save_intermediate_images(
                images_epoch_dir, [x, real_pred, fake_pred, x_ref, y_ref], f"iter_{i}"
            )
            metric_logger.log_neptune(f"images", grid)

        if accelerator.is_main_process:
            metric_logger.update(loss_g=l_g.item(), loss_d=l_d.item())
        i += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if accelerator.is_main_process:
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}