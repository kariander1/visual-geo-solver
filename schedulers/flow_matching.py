import torch
import numpy as np
from tqdm import tqdm


class FlowMatching:
    """Flow matching (rectified flow) scheduler.

    Learns a velocity field v that transports samples along straight paths
    from noise to data.  Drop-in replacement for DDIM — same interface for
    ``noise()``, ``sample()``, and the attributes used by the training loop
    (``diffusion_steps``, ``device``).

    Interpolation:  x_t = (1 - sigma) * x_data + sigma * eps,  sigma = t / T
        t = 1  -> nearly clean,  t = T-1  -> nearly noise  (matches DDIM)
    Velocity target:  v = x_data - eps
    x_0 prediction:   x_0 = x_t + sigma * v
    """

    def __init__(self, diffusion_steps=50, device=None, **kwargs):
        self.diffusion_steps = diffusion_steps
        self.device = device

    # ------------------------------------------------------------------
    # Noising (training)
    # ------------------------------------------------------------------
    def noise(self, x_data, t):
        """Linear interpolation + velocity target.

        Args:
            x_data: clean data  [B, ...]
            t:      integer timesteps  [B, 1]  in [1, diffusion_steps-1]

        Returns:
            x_t:      noised sample  (same shape as x_data)
            velocity: target  v = x_data - eps  (same shape as x_data)
        """
        eps = torch.randn(size=x_data.shape, device=x_data.device)

        B = x_data.shape[0]
        shape = [B] + [1] * (x_data.ndim - 1)
        sigma = (t / self.diffusion_steps).float().view(shape)

        x_t = (1 - sigma) * x_data + sigma * eps
        velocity = x_data - eps
        return x_t, velocity

    # ------------------------------------------------------------------
    # Sampling timestep schedules (copied from DDIM — general-purpose)
    # ------------------------------------------------------------------
    def _get_sampling_timesteps(self, sampling_steps, sampling_schedule):
        """Build a list of timesteps for the sampling loop.

        Returns List[int] in descending order (high -> low noise), ending at 1.
        """
        full = list(range(self.diffusion_steps - 1, 0, -1))  # [T-1 .. 1]
        N = len(full)

        if sampling_steps is None or sampling_steps >= N:
            return full

        if sampling_schedule.startswith("first_mid_last"):
            parts = sampling_schedule.split(":")
            first, mid, last = [int(x) for x in parts[1].split(",")]
            first = min(first, N)
            last = min(last, N - first)
            head = full[:first]
            tail = full[-last:] if last > 0 else []
            mid_start = first
            mid_end = N - last
            if mid > 0 and mid_end > mid_start:
                mid_indices = np.linspace(mid_start, mid_end - 1, mid, dtype=int)
                middle = [full[i] for i in mid_indices]
            else:
                middle = []
            return head + middle + tail
        elif sampling_schedule.startswith("first_last"):
            parts = sampling_schedule.split(":")
            if len(parts) == 2:
                first, last = [int(x) for x in parts[1].split(",")]
            else:
                first = int(sampling_steps * 0.6)
                last = sampling_steps - first
            first = min(first, N)
            last = min(last, N - first)
            head = full[:first]
            tail = full[-last:] if last > 0 else []
            return head + tail
        elif sampling_schedule == "cosine":
            t = np.linspace(0, np.pi, sampling_steps)
            indices = np.round((1 - np.cos(t)) / 2 * (N - 1)).astype(int)
            indices = np.unique(np.clip(indices, 0, N - 1))
            return [full[i] for i in indices]
        elif sampling_schedule == "cosine_head":
            t = np.linspace(0, np.pi / 2, sampling_steps)
            indices = np.round((1 - np.cos(t)) * (N - 1)).astype(int)
            indices = np.unique(np.clip(indices, 0, N - 1))
            return [full[i] for i in indices]
        elif sampling_schedule == "cosine_tail":
            t = np.linspace(0, np.pi / 2, sampling_steps)
            indices = np.round(np.sin(t) * (N - 1)).astype(int)
            indices = np.unique(np.clip(indices, 0, N - 1))
            return [full[i] for i in indices]
        elif sampling_schedule == "quadratic":
            t = np.linspace(0, 1, sampling_steps)
            indices = np.round(t**2 * (N - 1)).astype(int)
            indices = np.unique(np.clip(indices, 0, N - 1))
            return [full[i] for i in indices]
        else:  # "uniform"
            indices = np.linspace(0, N - 1, sampling_steps, dtype=int)
            return [full[i] for i in indices]

    # ------------------------------------------------------------------
    # Sampling (Euler ODE integration from noise -> data)
    # ------------------------------------------------------------------
    def sample(
        self,
        model,
        n_samples=1000,
        n_features=9,
        condition: torch.Tensor = 0,
        guidance_scale=1,
        x_T=None,
        seed=None,
        return_dict=True,
        index_in=[],
        source_distribution=None,
        index_out=[],
        sampling_steps=None,
        sampling_schedule="uniform",
    ):
        """Euler ODE integration from pure noise to data."""
        with torch.no_grad():
            # ---- initialise x_t from noise ----
            if x_T is None:
                mu, sigma = 0, 1
                if isinstance(condition, torch.Tensor) and condition is not None:
                    condition = condition.to(self.device)
                    if condition.dim() == 3:
                        condition = condition.unsqueeze(0)
                    target_shape = (
                        (n_samples,) + condition.shape[1:]
                        if condition.dim() >= 2
                        else (n_samples, 1)
                    )
                else:
                    target_shape = (n_samples, n_features)
                if source_distribution is not None:
                    mu = source_distribution[0]
                    sigma = source_distribution[1]
                if seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    x_t = torch.empty(target_shape, device=self.device).normal_(
                        mean=mu, std=sigma, generator=generator
                    )
                    if len(index_in) > 0:
                        x_t = x_t[index_in]
                        n_samples = len(index_in)
                    if len(index_out) > 0:
                        mask = torch.ones(x_t.size(0), dtype=torch.bool, device=x_t.device)
                        mask[index_out] = False
                        x_t = x_t[mask]
                        n_samples = x_t.shape[0]
                else:
                    x_t = torch.empty(target_shape, device=self.device).normal_(
                        mean=mu, std=sigma
                    )
            else:
                x_t = x_T
                n_samples = x_T.shape[0]

            # ---- timestep schedule ----
            timestep_list = self._get_sampling_timesteps(sampling_steps, sampling_schedule)

            # ---- trajectory storage (matches DDIM return structure) ----
            x_t_traj = [x_t]
            x_0_traj = []
            guidance_scale_traj = []
            noise_pred_traj = []
            noise_pred_cond_traj = []
            noise_pred_scaled_traj = []
            x_t_scaled_traj = []

            total_steps = len(timestep_list)
            for step_idx, t in enumerate(
                tqdm(timestep_list, desc=f"Sampling ({total_steps} steps)", unit="step")
            ):
                prev_t = timestep_list[step_idx + 1] if step_idx + 1 < len(timestep_list) else 0
                t_tensor = torch.full([n_samples, 1], t, device=self.device)

                # ---- build condition tensor ----
                condition = condition.to(self.device)
                n_conditions = condition.shape[0]
                if n_conditions == 1:
                    c_tensor = condition.repeat(n_samples, *[1 for _ in range(condition.ndim - 1)])
                elif n_conditions < n_samples:
                    assert n_samples % n_conditions == 0, (
                        "n_samples must be divisible by number of conditions"
                    )
                    c_tensor = condition.repeat_interleave(n_samples // n_conditions, dim=0)
                elif n_conditions == n_samples:
                    c_tensor = condition
                else:
                    raise ValueError(
                        f"Condition batch size ({n_conditions}) > n_samples ({n_samples})"
                    )

                null_tensor = torch.full_like(c_tensor, 0, device=self.device)

                # ---- velocity prediction (with optional CFG) ----
                v_cond = model(x_t, t_tensor, c_tensor)
                if guidance_scale != 1:
                    v_uncond = model(x_t, t_tensor, null_tensor)
                    v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_pred = v_cond

                # ---- x_0 prediction at current sigma ----
                sigma_curr = t / self.diffusion_steps
                x_0_pred = x_t + sigma_curr * v_pred

                # ---- Euler step ----
                sigma_prev = prev_t / self.diffusion_steps
                d_sigma = sigma_curr - sigma_prev          # positive
                x_t = x_t + d_sigma * v_pred

                # ---- store trajectories ----
                x_t_traj.append(x_t)
                x_0_traj.append(x_0_pred)
                guidance_scale_traj.append(
                    torch.full([n_samples, 1], guidance_scale, device=self.device)
                )
                noise_pred_traj.append(v_pred)
                noise_pred_cond_traj.append(v_cond)
                noise_pred_scaled_traj.append(v_pred)
                x_t_scaled_traj.append(x_t)

            # ---- final sample ----
            x_0 = x_t
            x_0_traj.append(x_0)

            x_t_traj = torch.stack(x_t_traj, dim=1)
            x_0_traj = torch.stack(x_0_traj, dim=1)
            noise_pred_traj = torch.stack(noise_pred_traj, dim=1)
            noise_pred_cond_traj = torch.stack(noise_pred_cond_traj, dim=1)
            noise_pred_scaled_traj = torch.stack(noise_pred_scaled_traj, dim=1)
            x_t_scaled_traj = torch.stack(x_t_scaled_traj, dim=1)
            guidance_scale_traj = torch.stack(guidance_scale_traj, dim=1)

            if return_dict:
                return {
                    "x_0": x_0,
                    "x_t_traj": x_t_traj,
                    "x_0_traj": x_0_traj,
                    "guidance_scale_traj": guidance_scale_traj,
                    "noise_pred_traj": noise_pred_traj,
                    "noise_pred_cond_traj": noise_pred_cond_traj,
                    "noise_pred_scaled_traj": noise_pred_scaled_traj,
                    "x_t_scaled_traj": x_t_scaled_traj,
                }
            return (
                x_0,
                x_t_traj,
                x_0_traj,
                guidance_scale_traj,
                noise_pred_traj,
                noise_pred_cond_traj,
                x_t_scaled_traj,
            )

    def __repr__(self):
        return f"FlowMatching(diffusion_steps={self.diffusion_steps}, device={self.device})"
