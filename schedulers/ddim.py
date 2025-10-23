import torch
from tqdm import tqdm

class DDIM:
    def __init__(
        self,
        diffusion_steps=50,
        beta_start =0.008,
        eta=0.0,
        device=None):
        self.diffusion_steps = diffusion_steps
        self.beta_start = beta_start
        self.timesteps = torch.tensor(range(0, self.diffusion_steps), device=device)
        self.schedule = torch.cos((self.timesteps / self.diffusion_steps + self.beta_start) / (1 + self.beta_start) * torch.pi / 2)**2

        self.baralphas = self.schedule / self.schedule[0]
        self.betas = 1 - self.baralphas / torch.concatenate([self.baralphas[0:1], self.baralphas[0:-1]])
        self.alphas = 1 - self.betas
        self.eta  = eta
        self.device = device
        
    def noise(self, x_0: torch.Tensor, t: torch.Tensor):
        eps = torch.randn(size=x_0.shape, device=x_0.device)
        
        B = x_0.shape[0]
        shape = [B] + [1] * (x_0.ndim - 1)

        sqrt_alpha = self.baralphas[t].sqrt().view(shape)
        sqrt_one_minus_alpha = (1 - self.baralphas[t]).sqrt().view(shape)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * eps
        return x_t, eps
    
    def denoise_ddim(self, x_t, t, noise_pred, prev_timestep=None, noise_pred_uncond=None):
        if torch.is_tensor(t):
            # expand dims to match x_t
            t = t.view(t.shape + (1,) * (x_t.dim() - t.dim()))

        if prev_timestep is None:
            prev_timestep = t-1
        alpha_prod_t = self.baralphas[t]
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_t_prev = self.baralphas[prev_timestep]
        alpha_prod_t_prev[t<0] = 1.0
        variance = self.betas[t]
        std_dev_t = self.eta * variance ** (0.5)

        x_0_pred = (x_t - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        x_t_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred
        x_t_prev = alpha_prod_t_prev ** (0.5) * x_0_pred + x_t_direction
        x_t_scaled =  ((alpha_prod_t_prev / alpha_prod_t)**0.5) * x_t
        noise_pred_scaled =              (
                (1 - alpha_prod_t_prev - std_dev_t**2)**0.5 -
                ((alpha_prod_t_prev * beta_prod_t / alpha_prod_t)**0.5)
            ) * noise_pred

        return x_t_prev, x_0_pred, x_t_scaled, noise_pred_scaled

    
    def sample(self, model, n_samples=1000, n_features=9, condition: torch.Tensor=0, guidance_scale=1, x_T=None, seed=None, return_dict=True, index_in=[], source_distribution=None, index_out=[]):
        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            if x_T is None:
                mu = 0
                sigma = 1
                if isinstance(condition, torch.Tensor) and condition is not None:
                    condition = condition.to(self.device)
                    if condition.dim() ==3:
                        condition = condition.unsqueeze(0)  # remove channel dimension if it exists
                    target_shape = (n_samples,) + condition.shape[1:] if condition.dim() >= 2 else (n_samples, 1)
                else:
                    # Fallback to flat vectors
                    target_shape = (n_samples, n_features)
                if source_distribution is not None:
                    mu = source_distribution[0]
                    sigma = source_distribution[1]
                if seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    x_t = torch.empty(target_shape, device=self.device).normal_(mean=mu, std=sigma, generator=generator)
                    if len(index_in) > 0:
                        x_t = x_t[index_in]
                        n_samples = len(index_in)
                    if len(index_out) > 0:
                        # remove indicies in index_out
                        # Create a mask for the indices to keep
                        mask = torch.ones(x_t.size(0), dtype=torch.bool, device=x_t.device)
                        mask[index_out] = False  # Mark indices in index_out as False
                        
                        # Use the mask to filter x_t
                        x_t = x_t[mask]
                        n_samples = x_t.shape[0]
                else:
                    x_t = torch.empty((n_samples, n_features), device=self.device).normal_(mean=mu, std=sigma)
            else:
                x_t = x_T
                n_samples = x_T.shape[0]
            x_t_traj = [x_t]
            x_0_traj = []
            x_t_scaled_traj = []
            guidance_scale_traj = []
            noise_pred_traj = []
            noise_pred_scaled_traj = []
            noise_pred_cond_traj = []
            for t in tqdm(range(self.diffusion_steps-1, 0, -1)):
                t_tensor = torch.full([n_samples, 1], t, device=self.device)

                # Ensure condition is on the correct device
                condition = condition.to(self.device)

                # Determine how many conditions are given
                n_conditions = condition.shape[0]

                # If only one condition is given, repeat it
                if n_conditions == 1:
                    c_tensor = condition.repeat(n_samples, *[1 for _ in range(condition.ndim - 1)])
                # If multiple conditions and fewer than n_samples, repeat them evenly
                elif n_conditions < n_samples:
                    assert n_samples % n_conditions == 0, "n_samples must be divisible by number of conditions"
                    repeat_factor = n_samples // n_conditions
                    c_tensor = condition.repeat_interleave(repeat_factor, dim=0)
                # If already matching shape
                elif n_conditions == n_samples:
                    c_tensor = condition
                else:
                    raise ValueError(f"Condition batch size ({n_conditions}) cannot exceed number of samples ({n_samples})")

                null_tensor = torch.full_like(c_tensor, 0, device=self.device)

                
                noise_pred_conditional = model(x_t, t_tensor, c_tensor)
                # Apply classifier-free guidance
                if guidance_scale != 1:
                    noise_pred_unconditional = model(x_t, t_tensor, null_tensor)
                    noise_pred = noise_pred_unconditional + guidance_scale * (noise_pred_conditional - noise_pred_unconditional)
                    x_t, x_0_hat, x_t_scaled, noise_pred_scaled = self.denoise_ddim(x_t, t, noise_pred, noise_pred_uncond=noise_pred_unconditional)
                else:
                    noise_pred = noise_pred_conditional
                    x_t, x_0_hat, x_t_scaled, noise_pred_scaled = self.denoise_ddim(x_t, t, noise_pred)

                guidance_scale_traj+= [torch.full([n_samples, 1], guidance_scale, device=self.device)]
                x_0_traj += [x_0_hat]
                x_t_traj += [x_t]
                x_t_scaled_traj += [x_t_scaled]
                noise_pred_traj += [noise_pred]
                noise_pred_scaled_traj += [noise_pred_scaled]
                noise_pred_cond_traj += [noise_pred_conditional]
            # reaching last timestep obtains orginal sample x_0
            x_0 = x_t
            x_0_traj += [x_0]
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
                    "x_t_scaled_traj": x_t_scaled_traj}
            return x_0, x_t_traj, x_0_traj,  guidance_scale_traj, noise_pred_traj, noise_pred_cond_traj, x_t_scaled_traj
   