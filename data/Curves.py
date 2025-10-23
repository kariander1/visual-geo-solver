import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.ndimage import distance_transform_edt
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from utils.viz import draw_square, is_self_intersecting
from omegaconf import DictConfig
from utils.viz import plot_sample_img_grid
import hydra
from scipy.ndimage import binary_fill_holes


class CurveImageDataset(Dataset):
    def __init__(
        self,
        num_samples,
        image_size,
        dataset_path='',
        use_distance_transform=False,
        use_signed_distance_transform=False,
        save_on_generate=False,
        H_range=(6, 15),
        num_points=500,
        side_length_range=(0.3, 0.6),
        rotation_range=(0, 2 * np.pi),
        shift_range=0.5,
        circles_ratio=0.1,
        use_spline=True,
        square_kwargs={},
        curve_kwargs={},
        max_squares=5,
        fill_square=False,
        **kwargs
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.dataset_path = Path(dataset_path)
        self.use_distance_transform = use_distance_transform
        self.use_signed_distance_transform = use_signed_distance_transform
        self.save_on_generate = save_on_generate
        self.max_dist = 2**0.5 * image_size
        self.eps = 1  # in pixels
        self.square_kwargs = square_kwargs
        self.curve_kwargs = curve_kwargs
        self.max_squares = max_squares
        self.gen_params = {
            "H_range": H_range,
            "num_points": num_points,
            "radius_range": side_length_range,
            "rotation_range": rotation_range,
            "use_spline": use_spline,
            "shift_range": shift_range,
            "circles_ratio": circles_ratio,
            "max_squares": max_squares,
            }
        if self.save_on_generate:
            self.generator = self._generate_curve_and_square(
                **self.gen_params
            )
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            for i in tqdm(range(num_samples), desc="Pre-generating samples"):
                sample = self._generate_sample()
                self._save_sample_as_png(sample, i)

        self._build_transform(image_size)
        
        # Define inverse (to go back to [0,255] image)
        self.fill_square = fill_square
        self.to_image = self._no_op
        if self.use_distance_transform:
            self.to_image = self._distance_to_image
            
        if self.use_signed_distance_transform:
            self.to_image = self._signed_distance_to_image
    def __len__(self):
        return self.num_samples

    def _build_transform(self, image_size: int):
        """Helper to build torchvision transform pipeline."""
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def _no_op(self, distance_map):
        return distance_map

    def _distance_to_image(self, distance_map):
        image = torch.ones_like(distance_map)
        mask = ((distance_map + 1) / 2 * self.max_dist <= self.eps)
        image[mask] = -1
        return image

    def _signed_distance_to_image(self, distance_map):
        image = torch.ones_like(distance_map)
        mask = abs(distance_map * self.max_dist) <= self.eps
        image[mask] = -1
        return image

    def change_size(self, new_size):
        self.image_size = new_size
        self._build_transform(new_size)

    def _fill_square_interior(self, square_img: torch.Tensor) -> torch.Tensor:
        """
        Given a square image tensor with -1 on the perimeter and 1 in background,
        fill the interior of the square with -1 as well.
        """
        # Convert to numpy and create binary mask of the square (-1 values)
        square_np = square_img.squeeze().cpu().numpy()  # shape: [H, W]
        perimeter_mask = square_np == -1

        # Fill the holes inside the perimeter
        filled_mask = binary_fill_holes(perimeter_mask)

        # Convert back: filled region gets -1, rest stays 1
        filled_square = np.where(filled_mask, -1.0, 1.0).astype(np.float32)

        # Convert to tensor
        return torch.from_numpy(filled_square).unsqueeze(0)  # add channel dimension back

    def _load_and_process(self, curve_img, square_img):
        # Convert to PIL if needed (e.g., raw tensors from generator)
        if isinstance(curve_img, torch.Tensor):
            curve_img = TF.to_pil_image((curve_img + 1) / 2)
        if isinstance(square_img, torch.Tensor):
            square_img = TF.to_pil_image((square_img + 1) / 2)

        # Apply transform
        curve_tensor = self.transform(curve_img)
        square_tensor = self.transform(square_img)

        # Post-process square
        if self.fill_square:
            square_tensor = self._fill_square_interior(square_tensor)

        if self.use_distance_transform:
            square_tensor = self._apply_distance_transform(square_tensor)
        elif self.use_signed_distance_transform:
            square_tensor = self._apply_signed_distance_transform(square_tensor)

        return curve_tensor, square_tensor

    def __getitem__(self, idx):
        curve_path = self.dataset_path / f"curve_{idx}.png"
        square_path = self.dataset_path / f"square_{idx}.png"

        return self.load_image_pair(curve_path, square_path)

    def load_image_pair(self, curve_path, square_path):
        """
        Load a pair of images from the given paths and apply the dataset transformations.
        """
        curve_img = Image.open(curve_path).convert('L')
        square_img = Image.open(square_path).convert('L')
        return self._load_and_process(curve_img, square_img)

    def flush(self):
        while True:
            try:
                _, _ = next(self.generator)
            except StopIteration:
                return

    def generate_sample(self, seed=None):
        curve_tensor_img, square_tensor_img = self._generate_sample(seed=seed)
        return self._load_and_process(curve_tensor_img, square_tensor_img)

    def _generate_sample(self, seed=None):
        try:
            curve_tensor, square_tensor = next(self.generator)
        except StopIteration:
            self.generator = self._generate_curve_and_square(**self.gen_params, seed=seed)
            curve_tensor, square_tensor = next(self.generator)
            


        curve_img = draw_periodic_spline_image(curve_tensor, self.image_size, **self.curve_kwargs)
        square_img = np.full((self.image_size, self.image_size), 255, dtype=np.uint8)
        draw_square(square_img, square_tensor, self.image_size, **self.square_kwargs)

        curve_tensor_img = (torch.tensor(curve_img).unsqueeze(0) / 255.0) * 2 - 1
        square_tensor_img = (torch.tensor(square_img).unsqueeze(0) / 255.0) * 2 - 1

        return curve_tensor_img, square_tensor_img

    def _apply_distance_transform(self, square_tensor_img):
        binary_mask = (square_tensor_img.numpy() != -1)
        distance_map = distance_transform_edt(binary_mask)
        square_tensor_img = torch.from_numpy(distance_map).float()
        square_tensor_img = (square_tensor_img / self.max_dist) * 2 - 1  # normalize to [-1, 1]

        return square_tensor_img

    def _apply_signed_distance_transform(self, square_tensor_img):
        """
        Computes a signed distance transform for the square image:
        - Inside the square: negative distances
        - Outside the square: positive distances
        """
        binary_mask = (square_tensor_img.numpy() != -1).astype(np.uint8)

        # Positive distances: from background to nearest edge of square
        dist_outside = distance_transform_edt(binary_mask == 0)

        # Negative distances: from inside square to nearest edge
        dist_inside = distance_transform_edt(binary_mask == 1)

        # Signed distance: negative inside, positive outside
        signed_dist = dist_inside-dist_outside
        signed_dist = signed_dist / self.max_dist
        signed_tensor = torch.from_numpy(signed_dist).float()

        return signed_tensor
    
    def _save_sample_as_png(self, sample, idx):
        curve_img, square_img = sample
        TF.to_pil_image((curve_img + 1) / 2).save(self.dataset_path / f"curve_{idx}.png")
        TF.to_pil_image((square_img + 1) / 2).save(self.dataset_path / f"square_{idx}.png")
        
    
    def save_sample_as_png(self, sample, save_path: str):
        curve_img, square_img = sample
        TF.to_pil_image((curve_img + 1) / 2).save(save_path / "curve.png")
        TF.to_pil_image((square_img + 1) / 2).save(save_path / "square.png")


    def _generate_curve_and_square(
        self,
        H_range,
        num_points,
        radius_range,
        rotation_range,
        use_spline,
        shift_range,
        circles_ratio,
        max_squares,
        seed=None
    ):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            if not hasattr(self, "_rng") or not isinstance(self._rng, np.random.Generator):
                self._rng = np.random.default_rng()
            rng = self._rng

        is_circle = rng.random() < circles_ratio
        while True:
            t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

            num_squares = rng.integers(1, max_squares + 1)
            if is_circle:
                num_squares = 1

            square_centers   = rng.uniform(-0.7, 0.7, size=(num_squares, 2))
            square_rotations = rng.uniform(rotation_range[0], rotation_range[1], size=num_squares)

            square_points_all = []

            for center, rotation in zip(square_centers, square_rotations):
                target_radius = rng.uniform(radius_range[0], radius_range[1])
                base = np.array([
                    [-1, -1],
                    [1, -1],
                    [1, 1],
                    [-1, 1]
                ]) * (target_radius / np.sqrt(2))  # scale square to match circumscribed radius

                # Apply rotation
                rot_mat = np.array([
                    [np.cos(rotation), -np.sin(rotation)],
                    [np.sin(rotation),  np.cos(rotation)]
                ])
                rotated_square = base @ rot_mat.T

                # Shift to new center
                rotated_square += center
                square_points_all.append(rotated_square)

            square_points = np.concatenate(square_points_all, axis=0)  # shape: (4*num_squares, 2)

            if is_circle:
                r_final = np.full_like(t, target_radius)

                square_points_all = []
                center_angles = rng.uniform(0, 2 * np.pi, size=num_squares)
                for theta_center in center_angles:
                    angle_offsets = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                    rotation = rng.uniform(0, 2*np.pi)
                    square_angles = (theta_center + angle_offsets + rotation) % (2 * np.pi)
                    
                    x = target_radius * np.cos(square_angles)
                    y = target_radius * np.sin(square_angles)
                    square = np.stack([x, y], axis=1)
                    square_points_all.append(square)

                square_points = np.concatenate(square_points_all, axis=0)
            else:
                H = int(rng.integers(H_range[0], H_range[1] + 1))
                rho = rng.random(H) * np.logspace(-0.5, -2.5, H)
                phi = rng.random(H) * 2 * np.pi
                r_base = np.ones_like(t)
                for h in range(1, H + 1):
                    r_base += rho[h - 1] * np.sin(h * t + phi[h - 1])

                # --- Convert square points to polar (angle + radius) ---
                x_pts, y_pts = square_points[:, 0], square_points[:, 1]
                square_radii = np.sqrt(x_pts**2 + y_pts**2)
                square_angles = np.arctan2(y_pts, x_pts) % (2 * np.pi)

                # --- Compute corrections at those angles ---
                square_indices = [np.argmin(np.abs(t - angle)) for angle in square_angles]
                r_current = r_base[square_indices]
                delta = square_radii - r_current

                # Make periodic spline
                sort_idx = np.argsort(square_angles)
                square_angles_sorted = square_angles[sort_idx]
                delta_sorted = delta[sort_idx]

                # Append for periodicity (ensuring strictly increasing domain)
                square_angles_sorted = np.append(square_angles_sorted, square_angles_sorted[0] + 2 * np.pi)
                delta_sorted = np.append(delta_sorted, delta_sorted[0])

                # Now safely fit periodic spline
                correction_spline = CubicSpline(square_angles_sorted, delta_sorted, bc_type='periodic')
                r_final = r_base + correction_spline(t)

            # Convert to Cartesian
            x = r_final * np.cos(t)
            y = r_final * np.sin(t)

            # Interpolate to smooth
            if use_spline:
                tck, _ = splprep([x, y], s=0, per=True)
                u_fine = np.linspace(0, 1, 1000)
                x, y = splev(u_fine, tck)
            else:
                x = np.append(x, x[0])
                y = np.append(y, y[0])

            if is_circle or not is_self_intersecting(x, y):
                break

        # Convert curve to tensor
        curve_tensor = torch.tensor(np.stack([x, y], axis=1))

        # Random global shift
        shift = rng.uniform(-shift_range, shift_range, size=(2,))
        curve_tensor += torch.tensor(shift)

        # Prepare square tensors and yield each square
        num_squares = square_points.shape[0] // 4
        for i in range(num_squares):
            square_pts = square_points[4*i:4*(i+1), :]
            square_tensor = torch.tensor(square_pts)
            square_tensor += torch.tensor(shift)

            # Yield each square individually with the full curve
            yield _fit_to_unit_box_with_padding(curve_tensor.clone(), square_tensor.clone())



def _fit_to_unit_box_with_padding(curve_tensor, square_tensor, padding_ratio=0.05):
    """
    Scales and shifts the input tensors to fit as large as possible inside [-1, 1]^2,
    leaving uniform padding on all sides. This does NOT enforce centering.

    Args:
        curve_tensor (Tensor): Shape (N, 2)
        square_tensor (Tensor): Shape (4, 2)
        padding_ratio (float): Padding ratio (e.g., 0.1 leaves 10% margin)

    Returns:
        (curve_tensor_scaled, square_tensor_scaled)
    """
    all_points = torch.cat([curve_tensor, square_tensor], dim=0)

    # 1. Get bounding box
    min_xy = all_points.min(dim=0).values
    max_xy = all_points.max(dim=0).values
    size_xy = max_xy - min_xy

    # 2. Compute target box size after padding
    target_size = 2.0 * (1.0 - padding_ratio)  # total span from -1+pad to 1-pad
    scale = target_size / size_xy.max()  # uniform scaling to fit max dimension

    # 3. Scale and shift
    scaled_points = (all_points - min_xy) * scale  # shift to origin, scale
    final_min = scaled_points.min(dim=0).values
    final_max = scaled_points.max(dim=0).values

    # 4. Shift to center inside [-1, 1] with padding
    shift_to_center = -((final_min + final_max) / 2)
    scaled_and_shifted = scaled_points + shift_to_center

    # 5. Split back
    curve_tensor_scaled = scaled_and_shifted[:curve_tensor.shape[0]]
    square_tensor_scaled = scaled_and_shifted[curve_tensor.shape[0]:]

    return curve_tensor_scaled, square_tensor_scaled




def draw_periodic_spline_image(curve_tensor, image_size, color=0, thickness_range=[1], **kwargs):
    img = np.full((image_size, image_size), 255, dtype=np.uint8)
    pts = (curve_tensor * (image_size // 2) + image_size // 2).numpy().astype(np.int32)
    thickness = np.random.choice(thickness_range)
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color=color, thickness=thickness)
    cv2.line(img, tuple(pts[-1]), tuple(pts[0]), color=color, thickness=thickness)
    return img


# Main
@hydra.main(config_path="../configs", config_name="config_debug", version_base=None)
def main(config: DictConfig):

    seed = config['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = CurveImageDataset(
    **config['dataset'],)
    plot_sample_img_grid(dataset, save_path='sample_grid.png',grid_size=8)

    

if __name__ == "__main__":
    main()