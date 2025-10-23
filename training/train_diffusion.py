import torch
from schedulers.ddim import DDIM
from model.diffusion import UNet
from loguru import logger
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from typing import List
from utils.viz import plot_condition_vs_images_grid, plot_training_visualization_grid
import os
from itertools import chain
import time
from itertools import islice
import torch.distributed as dist
from utils.metrics import squareness_metric, alignment_metric
from utils.viz import tensor_to_binary_mask
dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def validate_img_diffusion_model(
    config,
    model,
    scheduler,
    val_dataloader,
    dataset,
    null_condition=0,
    device="cuda",
    ddp=False,
    master_process=True,
    autocast_dtype='bfloat16',
    loss_config={},
    n_samples_for_metrics=1000,
    guidance_scale=1.0,
    seed=42,
    **kwargs,
):
    model.eval()
    val_loss = 0.0
    num_batches = 0
    quality_scores = []
    alignment_scores = []
    total_metric_samples = 0

    with torch.no_grad():
        val_iter = tqdm(val_dataloader, desc="Validation", leave=False) if master_process else val_dataloader

        for ellipse_imgs, square_imgs in val_iter:
            ellipse_imgs = ellipse_imgs.to(device, non_blocking=True)
            square_imgs = square_imgs.to(device, non_blocking=True)

            posterior = square_imgs
            t = torch.randint(1, scheduler.diffusion_steps, size=[ellipse_imgs.size(0), 1], device=device)
            x_t, eps = scheduler.noise(posterior, t)

            with torch.autocast(device_type=device, dtype=dtype_map[autocast_dtype]):
                noise_pred = model(x_t, t, ellipse_imgs)
                loss = calc_loss(
                    x_t,
                    t,
                    noise_pred,
                    eps,
                    scheduler=scheduler,
                    dataset=dataset,
                    posterior=posterior,
                    **loss_config,
                )

            val_loss += loss.item()
            num_batches += 1
            if master_process:
                val_iter.set_postfix(val_loss=loss.item())

            # Metrics collection (only up to n_samples_for_metrics)
            remaining = n_samples_for_metrics - total_metric_samples
            if remaining <= 0:
                continue

            to_sample = min(remaining, ellipse_imgs.size(0))
            pred_square_imgs = scheduler.sample(
                model,
                n_samples=to_sample,
                condition=ellipse_imgs[:to_sample],
                guidance_scale=guidance_scale,
                seed=seed,
                return_dict=True,
            )['x_0']  # B x 1 x H x W

            
            for curve_img, pred_square in zip(ellipse_imgs[:to_sample], pred_square_imgs):
                pred_square = dataset.to_image(pred_square)  # Convert to image format
                square_binary = tensor_to_binary_mask(pred_square)
                curve_binary = tensor_to_binary_mask(curve_img)

                quality = squareness_metric(square_binary, **config['metrics']['quality'])
                alignment = alignment_metric(square_binary, curve_binary, **config['metrics']['alignment'])

                quality_scores.append(quality)
                alignment_scores.append(alignment)

            total_metric_samples += to_sample

    # Convert losses and metrics to tensors for DDP reduction
    val_loss_tensor = torch.tensor(val_loss, device=device)
    num_batches_tensor = torch.tensor(num_batches, device=device)
    quality_tensor = torch.tensor(quality_scores, device=device)
    alignment_tensor = torch.tensor(alignment_scores, device=device)
    metric_count_tensor = torch.tensor(len(quality_scores), device=device)

    if ddp:
        # Reduce losses
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

        # Reduce metrics
        dist.all_reduce(metric_count_tensor, op=dist.ReduceOp.SUM)

        # Pad to max length for reduction
        max_len = max(quality_tensor.numel(), alignment_tensor.numel())
        pad = lambda x: torch.cat([x, torch.zeros(max_len - x.numel(), device=device)])
        quality_tensor = pad(quality_tensor)
        alignment_tensor = pad(alignment_tensor)

        dist.all_reduce(quality_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(alignment_tensor, op=dist.ReduceOp.SUM)

        avg_quality = (quality_tensor.sum() / metric_count_tensor).item()
        avg_alignment = (alignment_tensor.sum() / metric_count_tensor).item()
    else:
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    avg_val_loss = (val_loss_tensor / num_batches_tensor).item()
    return {
        "val_loss": avg_val_loss,
        "avg_quality": avg_quality,
        "avg_alignment": avg_alignment,
        "n_samples": total_metric_samples,
    }

def sample(
    model,
    scheduler,
    dataloader,
    dataset,
    n_samples: int,
    n_seeds_per_sample: int,
    guidance_scale: float | List[float],
    seed: int,
    log_to_wandb: bool,
    wandb_title: str = 'Validation Samples',
    timesteps_to_log: List[int] = [],
    **kwargs
):
    model.eval()
    results = []
    
    def _tensors_lists_differ(t1, t2):
        return not all(torch.equal(a, b) for a, b in zip(t1, t2))

    # Make sure guidance_scale is a list
    if isinstance(guidance_scale, float):
        guidance_scale = [guidance_scale]
        
    logged_images = []
    x_0_pred_images = []
    for sample_idx in range(n_samples):
        # Assume each sample is (condition_img, gt_square_img)
        condition_img, gt_square_img = dataloader.dataset[sample_idx]

        # Add batch dim and move to device
        condition = condition_img.unsqueeze(0).to(model.device)      # [1, C, H, W]
        gt_square_img = gt_square_img.unsqueeze(0).to(model.device)  # [1, C, H, W]
        gt_square_img = dataset.to_image(gt_square_img)  # Convert to image format
        
        sample_results = []
        all_x_0 = []
        all_x_0_to_image = []
        all_x_0_prediction = []  # Store all x_0 results for this sample
        all_x_0_prediction_to_image = []  # Store all x_0 results for this sample
        for scale in guidance_scale:
            output = scheduler.sample(
                model,
                n_samples=n_seeds_per_sample,
                condition=condition,
                guidance_scale=scale,
                seed=seed,
                return_dict=True,
            )
            sample_results.append(output)
            all_x_0.append(output["x_0"])  # Store raw x_0 tensors
            all_x_0_to_image.append(dataset.to_image(output["x_0"]))  # (B, C, H, W)
            
            all_x_0_prediction.append(output["x_0_traj"])
            all_x_0_prediction_to_image.append(dataset.to_image(output["x_0_traj"]))
        results.append(sample_results)

        preds_to_plot = all_x_0
        x_axis_vals = guidance_scale
        if _tensors_lists_differ(all_x_0, all_x_0_to_image):
            preds_to_plot = list(chain.from_iterable(zip(all_x_0, all_x_0_to_image)))
            x_axis_vals = [x for x in guidance_scale for _ in range(2)]
            
        fig_or_path = plot_condition_vs_images_grid(
            condition_img,
            gt=gt_square_img,
            predictions=preds_to_plot,
            x_axis=x_axis_vals,
            draw_condition_on_pred=True,
            title=f"Sample={sample_idx}"
        )
        if log_to_wandb:
            logged_images.append(wandb.Image(fig_or_path, caption=f"Sample={sample_idx}"))
            
        for i,scale in enumerate(guidance_scale):
            x_0_predictions = all_x_0_prediction[i] # (B, T, C, H, W)
            x_0_predictions_to_image = all_x_0_prediction_to_image[i]  # (B, T, C, H, W)
            # index out timesteps into a list
            
            x_0_predictions = [x_0_predictions[:, scheduler.diffusion_steps -t-1] for t in timesteps_to_log]
            x_0_predictions_to_image = [x_0_predictions_to_image[:, scheduler.diffusion_steps -t-1] for t in timesteps_to_log]

            preds_to_plot = x_0_predictions
            x_axis_vals = timesteps_to_log
            if _tensors_lists_differ(x_0_predictions, x_0_predictions_to_image):
                preds_to_plot = list(chain.from_iterable(zip(x_0_predictions, x_0_predictions_to_image)))
                x_axis_vals = [x for x in timesteps_to_log for _ in range(2)]


            fig_or_path = plot_condition_vs_images_grid(
                condition_img,
                gt=gt_square_img,
                predictions=preds_to_plot,
                x_axis=x_axis_vals,
                draw_condition_on_pred=True,
                title=f"x_0 predictions CFG={scale} Sample={sample_idx}",
                column_prefix="t="
            )
            if log_to_wandb:
                x_0_pred_images.append(wandb.Image(fig_or_path, caption=f"x_0 predictions CFG={scale} Sample={sample_idx}"))
    if log_to_wandb:
        wandb.log({wandb_title: logged_images})  # single W&B panel with multiple images  
        wandb.log({wandb_title+ " x_0_predictions": x_0_pred_images})  # single W&B panel with multiple images
    return results  # shape: [n_samples][len(guidance_scale)] (each is a dict if return_dict=True)

import torch.nn as nn

def calc_loss(
    x_t,
    t,
    noise_pred,
    eps,
    scheduler: DDIM,
    dataset,
    posterior,
    loss_func,
    loss_target,
    piecewise_loss,
    weight_x_0_by_timestep,
    **kwargs
):
    if loss_func == "l2":
        criterion = nn.MSELoss(reduction='none')
    elif loss_func == "l1":
        criterion = nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_func}")

    if loss_target == 'eps':
        loss_per_elem = criterion(noise_pred, eps)
    elif loss_target == 'x_0':
        _, x_0_pred,_,_ = scheduler.denoise_ddim(
            x_t,
            t,
            noise_pred,
        )
        loss_per_elem = criterion(x_0_pred, posterior)

        if weight_x_0_by_timestep:
            if torch.is_tensor(t):
                t = t.view(t.shape + (1,) * (loss_per_elem.dim() - t.dim()))
            weights = (scheduler.baralphas[t] / (1 - scheduler.baralphas[t]))
            if loss_func == "l1":
                weights = weights ** 0.5
            loss_per_elem = loss_per_elem * weights

    if not piecewise_loss:
        return loss_per_elem.mean()

    square_imgs = dataset.to_image(posterior)
    square_mask = (square_imgs == -1)

    loss_square = loss_per_elem[square_mask].mean()
    loss_background = loss_per_elem[~square_mask].mean()

    return 0.5 * loss_square + 0.5 * loss_background

def train_img_diffusion_model(
    config,
    model: UNet,
    scheduler:DDIM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    dataset,
    optimizer,
    lr_scheduler,
    outputs_dir,
    grad_accum_steps=1,
    num_epochs=20,
    p_null=0.5,
    null_condition=0,
    validation_config={},
    sampling_config={},
    train_viz_config={},
    loss_config={},
    checkpoint_every_n_epochs=10,
    log_every_n_steps=100,
    autocast_dtype='bfloat16',
    clip_grad_norm=False,
    ddp=False,
    master_process=True,
    ddp_world_size=1,
    clean_prev_checkpoints=True,
    device=None,
    **kwargs,
    ):
       

 
    train_step = 0
    # MEMORY FIX: Use manual iterator reset instead of cycle() to avoid memory accumulation with large datasets
    dataloader_iterator = iter(train_dataloader)

    for epoch in range(num_epochs):
        model.train()
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        num_steps_per_epoch = len(train_dataloader) // grad_accum_steps
      
        for step in range(num_steps_per_epoch):

            t0 = time.time()
            optimizer.zero_grad()
            step_loss = 0.0
            for micro_step in range(grad_accum_steps):
                try:
                    ellipse_imgs, square_imgs = next(dataloader_iterator)
                except StopIteration:
                    # MEMORY FIX: Reset iterator when exhausted instead of using cycle()
                    dataloader_iterator = iter(train_dataloader)
                    ellipse_imgs, square_imgs = next(dataloader_iterator)
                ellipse_imgs = ellipse_imgs.to(device)
                square_imgs = square_imgs.to(device)

                t = torch.randint(1, scheduler.diffusion_steps, size=[ellipse_imgs.size(0), 1]).to(device)
                posterior = square_imgs  # Learn only square

                x_t, eps = scheduler.noise(posterior, t)

                with torch.autocast(device_type=device, dtype=dtype_map[autocast_dtype]):
                    if p_null > 0:
                        mask = torch.rand(ellipse_imgs.size(0), device=device) <= p_null
                        noise_pred = torch.empty_like(eps)
                        noise_pred[~mask] = model(x_t[~mask], t[~mask], ellipse_imgs[~mask])
                        unconditioned_labels = torch.full_like(ellipse_imgs, null_condition, device=device)
                        noise_pred[mask] = model(x_t[mask], t[mask], unconditioned_labels[mask])
                    else:
                        noise_pred = model(x_t, t, ellipse_imgs)

                    loss = calc_loss(
                        x_t,
                        t,
                        noise_pred,
                        eps,
                        **loss_config,
                        scheduler=scheduler,
                        dataset=dataset,
                        posterior=posterior,
                    )
                loss = loss / grad_accum_steps
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  # Only sync gradients on the last micro-step
                loss.backward()
                step_loss += loss.detach()
            if ddp:
                dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
            grad_norm = None
            if clip_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += step_loss.item()
            if master_process and (train_step + 1) % log_every_n_steps == 0:
                wandb.log(
                    {
                        "batch_loss": step_loss.item(),
                        "grad_norm": grad_norm,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    },
                    step=train_step
                )

            
            if (train_step + 1) % train_viz_config['plot_every_n_steps']== 0:
                image_pixel_space = dataset.to_image(posterior)
                train_fig = plot_training_visualization_grid(
                    ellipse_imgs,
                    square_imgs,
                    posterior,
                    image_pixel_space,
                    eps,
                    x_t,
                    t,
                    **train_viz_config,
                )
                if master_process:
                    wandb.log({
                        "Training Viz": wandb.Image(train_fig, caption=f"Train step {train_step}"),
                        },step=train_step)
            train_step+=1
            torch.cuda.synchronize()  # Ensure all operations are complete before measuring time
            t1 = time.time()
            dt = (t1 - t0)
            imgs_per_sec = grad_accum_steps*ellipse_imgs.size(0)*ddp_world_size / dt
            if master_process:
                logger.info(f"Step {step}, Loss: {loss.item():.4f}, lr: {lr_scheduler.get_last_lr()[0]:.6f}, Grad Norm: {grad_norm:.4f}, Time: {dt*1000:.2f} ms, Images/sec: {imgs_per_sec:.2f}")
        avg_loss = epoch_loss / len(train_dataloader)
        if master_process:
            wandb.log({"epoch": epoch + 1, "epoch_loss": avg_loss}, step=train_step)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % validation_config['validate_every_n_epochs'] == 0:
            logger.info(f"Validating model at epoch {epoch + 1}")
            res = validate_img_diffusion_model(
                config,
                model,
                scheduler,
                val_dataloader,
                dataset,
                null_condition=null_condition,
                device=device,
                ddp=ddp,
                master_process=master_process,
                autocast_dtype=autocast_dtype,
                loss_config=loss_config,
                **validation_config,
            )
            avg_val_loss = res['val_loss']
            if master_process:
                wandb.log(res, step=train_step)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
            
        if master_process and (epoch + 1) % sampling_config['sample_every_n_epochs'] == 0:
            logger.info(f"Sampling images for epoch {epoch + 1}")
            sample(
                model,
                scheduler,
                val_dataloader,
                dataset,
                wandb_title="Validation Samples",
                **sampling_config
            )

            sample(
                model,
                scheduler,
                train_dataloader,
                dataset,
                wandb_title="Training Samples",
                **sampling_config
            )
        if master_process and (epoch + 1) % checkpoint_every_n_epochs == 0:
            model.eval()
            logger.info(f"Saving model checkpoint for epoch {epoch + 1}")
            checkpoints_dir = os.path.join(outputs_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            model_save_path = os.path.join(checkpoints_dir, f"checkpoint_{epoch + 1}.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "config": config
            }, model_save_path)
            logger.info(f"Model saved to {model_save_path}")
            
            if clean_prev_checkpoints:
                # Remove older checkpoints
                for filename in os.listdir(checkpoints_dir):
                    if filename.startswith("checkpoint_") and filename != f"checkpoint_{epoch + 1}.pth":
                        os.remove(os.path.join(checkpoints_dir, filename))
                        logger.info(f"Removed old checkpoint: {filename}")
                        