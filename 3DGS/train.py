"""
3D Gaussian Splatting Trainer - Rewritten following gsplat official example.
Uses step_pre_backward / step_post_backward pattern for proper densification.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import gsplat
from gsplat import DefaultStrategy
from gsplat.rendering import rasterization

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset
from utils.general_utils import mkdir_p
from utils.sh_utils import RGB2SH

from math import exp


# ---------------- SSIM (standard 3DGS: 11x11 Gaussian window, sigma=1.5) ----------------
def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    win_2d = g[:, None] * g[None, :]
    return win_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM over (C, H, W) or (B, C, H, W) tensors in [0, 1]. Returns scalar."""
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    C = img1.shape[1]
    win = _gaussian_window(window_size, sigma, C, img1.device, img1.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(img1, win, groups=C, padding=pad)
    mu2 = F.conv2d(img2, win, groups=C, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, win, groups=C, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, win, groups=C, padding=pad) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, win, groups=C, padding=pad) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ---------------- Standard 3DGS PLY I/O ----------------
def write_3dgs_ply(splats, ply_path):
    """Write splats as standard 3DGS PLY (Inria-compatible).

    Property layout per Gaussian (variable count for SH degree):
        x, y, z                       (3)   means
        nx, ny, nz                    (3)   placeholder normals (zero)
        f_dc_0..2                     (3)   sh0 (DC), order [R, G, B]
        f_rest_0..{3*K_rest-1}        (3*K_rest)  shN, transposed: [R bands... | G bands... | B bands...]
        opacity                       (1)   raw logit (sigmoid applied at render)
        scale_0..2                    (3)   raw log-scale (exp applied at render)
        rot_0..3                      (4)   quat (w, x, y, z) — gsplat convention

    Total floats/pt: 17 + 3*K_rest  (= 17 for sh_degree 0; 62 for sh_degree 3)
    """
    means     = splats["means"].detach().cpu().numpy().astype(np.float32)
    scales    = splats["scales"].detach().cpu().numpy().astype(np.float32)
    quats     = splats["quats"].detach().cpu().numpy().astype(np.float32)
    opacities = splats["opacities"].detach().cpu().numpy().reshape(-1).astype(np.float32)
    sh0       = splats["sh0"].detach().cpu().numpy().astype(np.float32)   # (N, 1, 3)
    shN       = splats["shN"].detach().cpu().numpy().astype(np.float32)   # (N, K-1, 3)

    N = means.shape[0]
    K_rest = shN.shape[1]
    sh_degree = int(round((1 + K_rest) ** 0.5)) - 1   # K = K_rest + 1 = (deg+1)^2

    sh0_flat = sh0.reshape(N, 3)                                          # (N, 3)
    # 3DGS Inria order: transpose (N, K_rest, 3) → (N, 3, K_rest) → flatten
    shN_flat = shN.transpose(0, 2, 1).reshape(N, 3 * K_rest)              # (N, 3*K_rest)

    rows = np.concatenate([
        means,                                            # 3
        np.zeros((N, 3), dtype=np.float32),               # 3 (nx, ny, nz)
        sh0_flat,                                         # 3 (f_dc)
        shN_flat,                                         # 3*K_rest (f_rest)
        opacities.reshape(N, 1),                          # 1
        scales,                                           # 3
        quats,                                            # 4
    ], axis=1)
    n_props = rows.shape[1]

    with open(ply_path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for ax in (b"x", b"y", b"z", b"nx", b"ny", b"nz"):
            f.write(b"property float " + ax + b"\n")
        for i in range(3):
            f.write(f"property float f_dc_{i}\n".encode())
        for i in range(3 * K_rest):
            f.write(f"property float f_rest_{i}\n".encode())
        f.write(b"property float opacity\n")
        for i in range(3):
            f.write(f"property float scale_{i}\n".encode())
        for i in range(4):
            f.write(f"property float rot_{i}\n".encode())
        f.write(b"end_header\n")
        rows.tofile(f)

    print(f"[PLY] Saved: {ply_path}  ({N} Gaussians, {n_props} floats/pt, sh_degree={sh_degree})")


def create_gaussians_with_optimizers(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    init_scales: torch.Tensor,
    init_opacity: float = 0.5,
    sh_degree: int = 0,  # 0 = DC only; 3 = full 16 SH bands
    means_lr: float = 1.6e-4,
    scale_lr: float = 5e-3,
    opacity_lr: float = 5e-2,
    quat_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = None,   # if None, use sh0_lr / 20 (3DGS convention)
    device: str = "cuda",
) -> tuple:
    """Initialize Gaussians from point cloud. Returns: (splats ParameterDict, optimizers dict).

    SH coefficients: K = (sh_degree + 1)^2.  sh0 is the DC term (initialized from RGB),
    shN is the rest (initialized to zero — only used once gsplat's sh_degree warmup reaches them).
    """
    points = points.to(device).float()
    rgbs = torch.clamp(rgbs.to(device).float(), 0, 1)

    N = points.shape[0]
    K = (sh_degree + 1) ** 2  # total SH coefficients per channel

    means = points
    scales = torch.log(init_scales.to(device))   # gsplat expects log-scales

    # Identity rotation. gsplat uses (w, x, y, z); identity = (1, 0, 0, 0).
    quats = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0

    init_opacity = float(np.clip(init_opacity, 0.001, 0.999))
    logit_opacity = float(np.log(init_opacity / (1 - init_opacity)))
    opacities = torch.ones((N, 1), device=device) * logit_opacity

    # SH: DC from RGB, rest zero
    sh0 = RGB2SH(rgbs).unsqueeze(1)                     # (N, 1, 3)
    shN = torch.zeros((N, K - 1, 3), device=device)     # (N, K-1, 3); empty (K-1=0) when sh_degree=0

    splats = nn.ParameterDict({
        "means":     nn.Parameter(means),
        "scales":    nn.Parameter(scales),
        "quats":     nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "sh0":       nn.Parameter(sh0),
        "shN":       nn.Parameter(shN),
    })

    if shN_lr is None:
        shN_lr = sh0_lr / 20.0   # 3DGS convention: higher-order SH learns 20× slower

    optimizers = {
        "means":     Adam([{"params": splats["means"]}],     lr=means_lr,   eps=1e-15),
        "scales":    Adam([{"params": splats["scales"]}],    lr=scale_lr,   eps=1e-15),
        "quats":     Adam([{"params": splats["quats"]}],     lr=quat_lr,    eps=1e-15),
        "opacities": Adam([{"params": splats["opacities"]}], lr=opacity_lr, eps=1e-15),
        "sh0":       Adam([{"params": splats["sh0"]}],       lr=sh0_lr,     eps=1e-15),
        "shN":       Adam([{"params": splats["shN"]}],       lr=shN_lr,     eps=1e-15),
    }

    print(f"[init] sh_degree={sh_degree} (K={K}) → sh0:{tuple(sh0.shape)}, shN:{tuple(shN.shape)}, "
          f"sh0_lr={sh0_lr}, shN_lr={shN_lr}")

    return splats, optimizers

def generate_sky_points(dataset, num_points=100000, depth_range=(50.0, 80.0), device="cuda"):
    """Generates 3D points for sky regions using unprojection."""
    all_points = []
    all_colors = []
    
    # Sample from frames to ensure coverage
    indices = np.linspace(0, len(dataset)-1, 100, dtype=int)
    K_inv = torch.inverse(dataset.get_intrinsics_torch().to(device))
    
    for idx in indices:
        mask = dataset.get_mask(idx)
        if mask is None: continue
        
        image = dataset.get_image(idx).to(device)
        pose_c2w = dataset.get_poses_torch()[idx].to(device)
        
        # Identify sky pixels
        sky_coords = torch.where(mask == 255)
        if len(sky_coords[0]) == 0: continue
        
        # Sample points from the sky
        num_to_sample = num_points // len(indices)
        sel = torch.randint(0, len(sky_coords[0]), (num_to_sample,))
        y, x = sky_coords[0][sel].to(device), sky_coords[1][sel].to(device)
        
        # Random depth initialization
        depths = torch.rand(num_to_sample, device=device) * (depth_range[1] - depth_range[0]) + depth_range[0]
        
        # Unproject: P_world = R * (K_inv * p_pix * depth) + t
        pix_h = torch.stack([x.float(), y.float(), torch.ones_like(x).float()], dim=-1)
        p_cam = (K_inv @ pix_h.unsqueeze(-1)).squeeze(-1) * depths.unsqueeze(-1)
        p_world = (pose_c2w[:3, :3] @ p_cam.unsqueeze(-1)).squeeze(-1) + pose_c2w[:3, 3]
        
        all_points.append(p_world)
        all_colors.append(image[:, y, x].T)

    return torch.cat(all_points), torch.cat(all_colors)

class Trainer:
    """3DGS Trainer following gsplat official pattern."""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: dict,
        force_cpu: bool = False,
    ):
        """Initialize trainer."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        mkdir_p(str(self.output_dir))
        
        self.config = config
        self.device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Trainer] Device: {self.device}")
        
        # Load dataset
        print("[Trainer] Loading dataset...")
        self.dataset = Dataset(data_dir)
        # Pointcloud data
        pc_points, pc_colors = self.dataset.get_pointcloud()
        num_pc = pc_points.shape[0]
        self.poses_c2w = self.dataset.get_poses_torch().to(self.device)
        self.K = self.dataset.get_intrinsics_torch().to(self.device)
        
        print(f"[Trainer] Points: {len(pc_points)}, Cameras: {len(self.poses_c2w)}")

        # Add sky points at random depth
        print("[Trainer] Initializing additional sky Gaussians...")
        # Generate Sky data
        sky_points, sky_colors = generate_sky_points(self.dataset, num_points=10000)
        num_sky = sky_points.shape[0]

        init_scale_lidar = config.get('init_scale_lidar', 0.1)
        init_scale_sky = config.get('init_scale_sky', 10.0)
        pc_scales = torch.ones((num_pc, 3)) * init_scale_lidar
        sky_scales = torch.ones((num_sky, 3)) * init_scale_sky

        # Combine both sets
        combined_points = torch.cat([pc_points.to(self.device), sky_points], dim=0)
        combined_colors = torch.cat([pc_colors.to(self.device), sky_colors], dim=0)
        combined_scales = torch.cat([pc_scales, sky_scales], dim=0)
        
        # Create Gaussians and optimizers
        init_opacity = config.get('init_opacity', 0.5)
        self.sh_degree_max = int(config.get('sh_degree', 0))
        self.splats, self.optimizers = create_gaussians_with_optimizers(
            points=combined_points,
            rgbs=combined_colors,
            init_scales=combined_scales,
            init_opacity=init_opacity,
            sh_degree=self.sh_degree_max,
            means_lr=config['lr_xyz'],
            scale_lr=config['lr_scaling'],
            opacity_lr=config['lr_opacity'],
            quat_lr=config['lr_rotation'],
            sh0_lr=config.get('lr_color', 2.5e-3),
            device=str(self.device),
        )
        
        print(f"[Trainer] Initialized {len(self.splats['means'])} Gaussians")
        
        # Setup learning rate scheduler (only for means which has schedule)
        self.scheduler = ExponentialLR(self.optimizers["means"], gamma=config['lr_decay'])
        
        # Setup strategy
        self.strategy = DefaultStrategy(
            prune_opa=config.get('prune_opa', 0.005),
            grow_grad2d=config.get('grow_grad2d', 0.0001),
            grow_scale3d=config.get('grow_scale3d', 0.01),
            grow_scale2d=config.get('grow_scale2d', 0.05),
            prune_scale3d=config.get('prune_scale3d', 0.15),
            prune_scale2d=config.get('prune_scale2d', 0.15),
            refine_start_iter=config.get('refine_start_iter', 500),
            refine_stop_iter=config.get('refine_stop_iter', 15000),
            refine_every=config.get('refine_every', 100),
            reset_every=config.get('reset_every', 3000),
            verbose=True,
        )
        
        # Initialize strategy state
        self.strategy_state = self.strategy.initialize_state()
        print(f"[Trainer] Strategy: densification {self.strategy.refine_start_iter}-{self.strategy.refine_stop_iter} iters, every {self.strategy.refine_every}")
        
        # Tensorboard
        self.tb_writer = SummaryWriter(str(self.output_dir / "runs"))
        self.iteration = 0
    
    def _active_sh_degree(self) -> int:
        """SH-degree warmup: every `sh_degree_interval` iters, unlock one more band."""
        interval = max(1, int(self.config.get('sh_degree_interval', 1000)))
        return min(self.iteration // interval, self.sh_degree_max)

    def rasterize_splats(self, camtoworld: torch.Tensor, K: torch.Tensor, width: int, height: int):
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)

        viewmat = torch.linalg.inv(camtoworld)
        K_batch = K.unsqueeze(0)
        viewmat_batch = viewmat.unsqueeze(0)

        # SH coefficients: (N, K, 3); gsplat evaluates view-dependent color when sh_degree is given
        colors_sh = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        sh_deg_now = self._active_sh_degree()

        # render_mode "RGB+ED" → returns 4-channel: [RGB, expected_depth] in one pass
        render_features, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_sh,
            viewmats=viewmat_batch,
            Ks=K_batch,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1000.0,
            sh_degree=sh_deg_now,
            render_mode="RGB+ED",
        )

        self.last_info = info

        image_rgb   = render_features[0, ..., :3]   # (H, W, 3)
        image_depth = render_features[0, ..., 3]    # (H, W)   expected depth
        image_alpha = render_alphas[0, ..., 0]      # (H, W)   accumulated opacity / silhouette

        return image_rgb.permute(2, 0, 1), image_depth, image_alpha
    
    def train_step(self, frame_indices: np.ndarray) -> float:
        """Run one training step."""
        # Zero gradients for all optimizers
        for opt in self.optimizers.values():
            opt.zero_grad()

        num_frames = len(frame_indices)
        sum_l1 = 0.0
        sum_dssim = 0.0
        sum_depth = 0.0
        n_depth_frames = 0

        w_ssim      = self.config.get('w_ssim', 0.2)        # standard 3DGS: 0.2
        w_depth     = self.config.get('w_depth', 0.0)
        w_scale_reg = self.config.get('w_scale_reg', 0.0)
        scale_thr   = self.config.get('scale_reg_threshold', 0.5)  # meters

        # Render all frames in batch
        for frame_idx in frame_indices:
            target_image = self.dataset.get_image(frame_idx).to(self.device)  # [3, H, W]
            mask = self.dataset.get_mask(frame_idx).to(self.device)
            # Mask: 0 for objects (keep), 255 for sky (ignore)
            loss_mask_2d = (mask == 0)                                 # (H, W) bool
            loss_mask    = loss_mask_2d.float().unsqueeze(0)           # (1, H, W) for RGB

            pose_c2w = self.poses_c2w[frame_idx]

            # Render
            rendered_rgb, rendered_depth, _ = self.rasterize_splats(
                pose_c2w, self.K,
                self.dataset.image_width, self.dataset.image_height,
            )

            # ----- RGB L1 loss (sky-masked) -----
            l1_val = F.l1_loss(rendered_rgb * loss_mask, target_image * loss_mask)
            sum_l1 = sum_l1 + l1_val

            # ----- D-SSIM (1 - SSIM) on full image; standard 3DGS does NOT mask SSIM -----
            if w_ssim > 0:
                ssim_val = ssim(rendered_rgb.clamp(0, 1), target_image)
                sum_dssim = sum_dssim + (1.0 - ssim_val)

            # ----- Depth L1 loss (only where LiDAR has a point AND not sky) -----
            if w_depth > 0:
                lidar_depth = self.dataset.get_lidar_depth(frame_idx, device=str(self.device))
                depth_valid = (lidar_depth > 0) & loss_mask_2d
                if depth_valid.any():
                    d_loss = F.l1_loss(rendered_depth[depth_valid], lidar_depth[depth_valid])
                    sum_depth = sum_depth + d_loss
                    n_depth_frames += 1

        avg_l1    = sum_l1 / num_frames
        avg_dssim = (sum_dssim / num_frames) if w_ssim > 0 else torch.tensor(0.0, device=self.device)
        avg_depth = (sum_depth / n_depth_frames) if n_depth_frames > 0 else torch.tensor(0.0, device=self.device)

        # ----- Scale regularization (over ALL Gaussians, once per step) -----
        if w_scale_reg > 0:
            scales_actual = torch.exp(self.splats["scales"])              # (N, 3) meters
            # Soft hinge: only penalize axes that exceed threshold
            scale_reg = F.relu(scales_actual - scale_thr).mean()
        else:
            scale_reg = torch.tensor(0.0, device=self.device)

        # Standard 3DGS: L = (1 - λ) * L1 + λ * (1 - SSIM); λ = w_ssim
        total_loss = (1.0 - w_ssim) * avg_l1 + w_ssim * avg_dssim \
                     + w_depth * avg_depth + w_scale_reg * scale_reg

        # Tensorboard log per-component
        self.tb_writer.add_scalar('loss/l1',        float(avg_l1),    self.iteration)
        self.tb_writer.add_scalar('loss/dssim',     float(avg_dssim), self.iteration)
        self.tb_writer.add_scalar('loss/depth',     float(avg_depth), self.iteration)
        self.tb_writer.add_scalar('loss/scale_reg', float(scale_reg), self.iteration)
        self.tb_writer.add_scalar('sh_degree',      self._active_sh_degree(), self.iteration)

        # Pre-backward step
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.iteration,
            info=self.last_info,
        )

        # Backward
        total_loss.backward()
        
        # Optimizer steps for all parameters
        for opt in self.optimizers.values():
            opt.step()
        
        # Post-backward step (handles split/clone/prune densification)
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.iteration,
            info=self.last_info,
            packed=False,
        )
        
        return total_loss.item()

    def train(self, num_epochs: int, batch_size: int = 4):
        """Run training loop."""
        print(f"\n[Trainer] Starting training: {num_epochs} epochs, batch_size={batch_size}\n")
        
        num_batches = (len(self.dataset) + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(self.dataset))
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0.0
            
            for batch_idx in pbar:
                # Get batch
                start = batch_idx * batch_size
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]
                
                # Train step
                loss = self.train_step(batch_indices)
                epoch_loss += loss
                
                # Log
                pbar.set_postfix({'loss': f'{loss:.6f}'})
                self.tb_writer.add_scalar('loss/train', loss, self.iteration)
                self.tb_writer.add_scalar('gs_count', len(self.splats["means"]), self.iteration)
                
                self.iteration += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")

            # Save side-by-side renderings for visual inspection
            if ((epoch + 1) % self.config.get('vis_interval', 10) == 0) or (epoch == 0):
                self.save_visualizations(
                    epoch + 1, n=200, stride=2,
                    resolution_ratio=0.5,
                )

            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 5) == 0:
                self.save_checkpoint(epoch + 1)

            # === Depth-guided densification (warm-up phase) ===
            # Inject Gaussians at silhouette holes using DA3 metric depth, but only
            # within [start_depth_densify_epoch, end_depth_densify_epoch] and every
            # `depth_densify_interval` epochs. `epoch` is 0-indexed here.
            start_e = self.config.get('start_depth_densify_epoch', -1)
            end_e   = self.config.get('end_depth_densify_epoch', -1)
            interval = max(1, self.config.get('depth_densify_interval', 5))
            if 0 <= start_e <= epoch <= end_e and ((epoch - start_e) % interval == 0):
                n_added = self.densify_from_depth()
                self.tb_writer.add_scalar('densify/depth_injected', n_added, self.iteration)
                self.tb_writer.add_scalar('gs_count_after_inject', len(self.splats["means"]), self.iteration)

            # LR schedule
            self.scheduler.step()

    @torch.no_grad()
    def densify_from_depth(self):
        """Inject Gaussians at silhouette holes using DA3 metric depth.

        For a sample of frames:
          1. Render alpha (accumulated opacity).
          2. Mark pixels where alpha < threshold AND not sky AND DA3 conf high enough
             AND DA3 depth in valid range.
          3. Unproject those pixels to world space using DA3 metric depth + camera pose.
          4. Append the new points as Gaussians (extends params, optimizers, strategy state).

        Returns: number of injected Gaussians.
        """
        cfg = self.config
        sil_thr     = cfg.get('silhouette_threshold', 0.5)
        conf_pct    = cfg.get('depth_densify_conf_percentile', 50)
        init_scale  = cfg.get('depth_densify_init_scale', 0.15)
        init_op     = cfg.get('depth_densify_init_opacity', 0.1)
        max_depth   = cfg.get('depth_densify_max_depth', 100.0)
        min_depth   = cfg.get('depth_densify_min_depth', 1.0)
        n_frames    = cfg.get('depth_densify_frames_per_pass', 50)
        max_total   = cfg.get('depth_densify_max_points_per_pass', 100000)

        if len(self.dataset.da3_depth_filenames) == 0:
            print("[densify_from_depth] No DA3 depth available, skipping")
            return 0

        n_frames = min(n_frames, len(self.dataset))
        frame_indices = np.linspace(0, len(self.dataset) - 1, n_frames, dtype=int)
        per_frame_cap = max(100, max_total // max(1, n_frames))

        K_inv = torch.inverse(self.K)
        new_means_list, new_colors_list = [], []
        considered = 0

        # ---- per-filter survival counters (debug)
        n_total = n_alpha = n_alpha_ns = n_alpha_ns_conf = n_all = 0
        a_min, a_max, a_sum, a_n = 1e9, -1e9, 0.0, 0
        d_min, d_max, d_sum, d_n = 1e9, -1e9, 0.0, 0
        c_min, c_max, c_sum, c_n = 1e9, -1e9, 0.0, 0
        conf_skipped = False

        for frame_idx in frame_indices:
            da3_depth, da3_conf = self.dataset.get_da3_depth(frame_idx, device=str(self.device))
            if da3_depth is None:
                continue

            pose_c2w = self.poses_c2w[frame_idx]
            _, _, rendered_alpha = self.rasterize_splats(
                pose_c2w, self.K,
                self.dataset.image_width, self.dataset.image_height,
            )  # (H, W)

            sky = self.dataset.get_mask(frame_idx)
            not_sky = (sky.to(self.device) == 0) if sky is not None else torch.ones_like(rendered_alpha, dtype=torch.bool)

            # Per-frame conf threshold (percentile over non-sky pixels).
            # If conf has no variation (e.g. DA3 metric model has no real conf head
            # → all zeros), strict `>` would reject everything. Skip the filter then.
            ns_conf = da3_conf[not_sky]
            if ns_conf.numel() == 0:
                continue
            if float(ns_conf.max() - ns_conf.min()) < 1e-6 or conf_pct <= 0:
                m_conf = torch.ones_like(da3_conf, dtype=torch.bool)
                conf_skipped = True
            else:
                conf_thr = torch.quantile(ns_conf, conf_pct / 100.0)
                m_conf = da3_conf > conf_thr

            # ---- per-filter survival
            n_total      += rendered_alpha.numel()
            m_alpha       = rendered_alpha < sil_thr
            n_alpha      += int(m_alpha.sum())
            m_alpha_ns    = m_alpha & not_sky
            n_alpha_ns   += int(m_alpha_ns.sum())
            m_with_conf   = m_alpha_ns & m_conf
            n_alpha_ns_conf += int(m_with_conf.sum())

            valid = m_with_conf & (da3_depth > min_depth) & (da3_depth < max_depth)
            n_all += int(valid.sum())

            a_min = min(a_min, float(rendered_alpha.min()))
            a_max = max(a_max, float(rendered_alpha.max()))
            a_sum += float(rendered_alpha.mean()); a_n += 1
            d_min = min(d_min, float(da3_depth.min()))
            d_max = max(d_max, float(da3_depth.max()))
            d_sum += float(da3_depth.mean()); d_n += 1
            c_min = min(c_min, float(da3_conf.min()))
            c_max = max(c_max, float(da3_conf.max()))
            c_sum += float(da3_conf.mean()); c_n += 1

            ys, xs = torch.where(valid)
            considered += len(ys)
            if len(ys) == 0:
                continue

            if len(ys) > per_frame_cap:
                sel = torch.randperm(len(ys), device=self.device)[:per_frame_cap]
                ys, xs = ys[sel], xs[sel]

            depths = da3_depth[ys, xs]
            pix_h = torch.stack([xs.float(), ys.float(), torch.ones_like(xs).float()], dim=-1)  # (N, 3)
            p_cam = (K_inv @ pix_h.unsqueeze(-1)).squeeze(-1) * depths.unsqueeze(-1)             # (N, 3)
            p_world = (pose_c2w[:3, :3] @ p_cam.unsqueeze(-1)).squeeze(-1) + pose_c2w[:3, 3]

            image = self.dataset.get_image(frame_idx).to(self.device)  # (3, H, W)
            colors = image[:, ys, xs].T                                # (N, 3)

            new_means_list.append(p_world)
            new_colors_list.append(colors)

        # ---- always print survival pipeline (helps tune thresholds)
        conf_label = "conf SKIPPED (no variation)" if conf_skipped else f"conf>p{conf_pct}"
        print(f"[densify_from_depth] survival: total={n_total} → "
              f"alpha<{sil_thr}={n_alpha} ({100.*n_alpha/max(n_total,1):.2f}%) → "
              f"& not_sky={n_alpha_ns} → "
              f"& {conf_label}={n_alpha_ns_conf} → "
              f"& depth∈[{min_depth},{max_depth}]={n_all}")
        if a_n > 0:
            print(f"  alpha stats: min={a_min:.3f} max={a_max:.3f} mean={a_sum/a_n:.3f}")
        if d_n > 0:
            print(f"  da3_depth stats: min={d_min:.2f} max={d_max:.2f} mean={d_sum/d_n:.2f} m")
        if c_n > 0:
            print(f"  da3_conf  stats: min={c_min:.4f} max={c_max:.4f} mean={c_sum/c_n:.4f}")

        if not new_means_list:
            print(f"[densify_from_depth] no candidate pixels (considered={considered})")
            return 0

        new_means = torch.cat(new_means_list, dim=0)
        new_colors = torch.cat(new_colors_list, dim=0)

        if len(new_means) > max_total:
            sel = torch.randperm(len(new_means), device=self.device)[:max_total]
            new_means = new_means[sel]
            new_colors = new_colors[sel]

        self._inject_gaussians(new_means, new_colors, init_scale, init_op)
        print(f"[densify_from_depth] injected {len(new_means)} (silhouette<{sil_thr}, "
              f"considered={considered}, total now {len(self.splats['means'])})")
        return len(new_means)

    @torch.no_grad()
    def _inject_gaussians(self, new_means: torch.Tensor, new_colors: torch.Tensor,
                          init_scale: float = 0.15, init_opacity: float = 0.1):
        """Append new Gaussians to splats; extend optimizer state; reset strategy state.

        new_colors: (N, 3) RGB in [0, 1] — typically the source-frame pixel color at the
        unprojected location.  sh0 is initialized from this via RGB2SH; shN starts at zero.
        """
        N_new = new_means.shape[0]
        if N_new == 0:
            return
        device = self.device

        new_colors = torch.clamp(new_colors, 0, 1)
        new_scales_log = torch.log(torch.ones((N_new, 3), device=device) * init_scale)
        # Identity rotation, gsplat (w, x, y, z) convention
        new_quats = torch.zeros((N_new, 4), device=device)
        new_quats[:, 0] = 1.0
        op = float(np.clip(init_opacity, 0.001, 0.999))
        logit_op = float(np.log(op / (1 - op)))
        new_opacities = torch.ones((N_new, 1), device=device) * logit_op
        new_sh0 = RGB2SH(new_colors).unsqueeze(1)                    # (N, 1, 3) ← pixel color
        K_minus_1 = self.splats["shN"].shape[1]                      # match current SH band count
        new_shN = torch.zeros((N_new, K_minus_1, 3), device=device)  # (N, K-1, 3) zero-init

        new_params = {
            "means":     new_means,
            "scales":    new_scales_log,
            "quats":     new_quats,
            "opacities": new_opacities,
            "sh0":       new_sh0,
            "shN":       new_shN,
        }

        for key, new_val in new_params.items():
            old_param = self.splats[key]
            opt = self.optimizers[key]

            # Extend Adam state buffers along dim 0
            old_state = opt.state.get(old_param, {})
            new_state = {}
            for sk, sv in old_state.items():
                if torch.is_tensor(sv) and sv.dim() > 0 and sv.shape[0] == old_param.shape[0]:
                    z = torch.zeros((N_new, *sv.shape[1:]), device=sv.device, dtype=sv.dtype)
                    new_state[sk] = torch.cat([sv, z], dim=0)
                else:
                    new_state[sk] = sv

            new_data = torch.cat([old_param.data, new_val], dim=0)
            new_param = nn.Parameter(new_data)

            self.splats[key] = new_param
            opt.param_groups[0]['params'] = [new_param]
            opt.state.clear()
            opt.state[new_param] = new_state

        # Strategy state holds per-Gaussian counters (grad2d/count/...). Sizes now mismatch,
        # so reset — gsplat will re-initialise on the next step_post_backward.
        self.strategy_state = self.strategy.initialize_state()
        # Optimizers were updated in-place (same opt object, param_groups list unchanged);
        # the scheduler holding ExponentialLR(optimizers["means"]) keeps working.

    @torch.no_grad()
    def save_visualizations(self, epoch: int, n: int = 5, stride: int = 1,
                            resolution_ratio: float = 1.0):
        """Render `n` frames spaced by `stride` and save [GT | rendered] PNGs.

        resolution_ratio: scale factor for saved images (0 < r <= 1).
            Note: rendering still happens at full resolution (gsplat needs it
            to match GT for accurate visual comparison); only the saved file
            is downscaled via INTER_AREA after combining.
        """
        vis_dir = self.output_dir / "vis" / f"epoch_{epoch:04d}"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Pick n indices spaced by stride, clipped to dataset length
        max_idx = len(self.dataset) - 1
        sample_idxs = [min(i * stride, max_idx) for i in range(n)]

        for idx in sample_idxs:
            target = self.dataset.get_image(idx).to(self.device)             # (3, H, W)
            pose_c2w = self.poses_c2w[idx]
            rendered_rgb, _, rendered_alpha = self.rasterize_splats(
                pose_c2w, self.K,
                self.dataset.image_width, self.dataset.image_height,
            )
            rendered_rgb = rendered_rgb.clamp(0, 1)
            # alpha (H, W) ∈ [0, 1]  → 3-channel grayscale (white = full coverage, black = hole)
            alpha_3ch = rendered_alpha.clamp(0, 1).unsqueeze(0).expand(3, -1, -1)

            combined = torch.cat([target, rendered_rgb, alpha_3ch], dim=2)   # (3, H, 3W) — GT | render | alpha
            img_np = (combined.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            if resolution_ratio < 1.0:
                h, w = img_np.shape[:2]
                img_np = cv2.resize(
                    img_np,
                    (max(1, int(w * resolution_ratio)), max(1, int(h * resolution_ratio))),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(str(vis_dir / f"frame_{idx:04d}.png"), img_np)

    def save_checkpoint(self, epoch: int):
        """Save checkpoint."""
        ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'iteration': self.iteration,
            'splats': {k: v.data for k, v in self.splats.items()},
        }, ckpt_path)
        print(f"[Trainer] Saved checkpoint: {ckpt_path}")
    
    def save_ply(self, filename: str = "gaussians.ply"):
        """Export as standard 3DGS PLY (compatible with shN; readable by Inria viewer)."""
        ply_path = self.output_dir / filename
        write_3dgs_ply(self.splats, ply_path)

    def load_checkpoint(self, ckpt_path: str):
        """Load a saved checkpoint and resume training."""
        print(f"[Trainer] Loading checkpoint from: {ckpt_path}")
        # Load the data to the current device
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # Restore the iteration count
        self.iteration = checkpoint.get('iteration', 0)
        
        # Restore parameter data
        # We wrap the saved tensors back into nn.Parameters to maintain gradient flow
        splat_data = checkpoint['splats']
        for k in self.splats.keys():
            if k in splat_data:
                self.splats[k] = nn.Parameter(splat_data[k].to(self.device))
            else:
                print(f"[Warning] Key {k} not found in checkpoint.")

        # CRITICAL: Re-initialize optimizers
        # Old optimizers are tied to the memory addresses of the old parameters
        sh0_lr = self.config.get('lr_color', 2.5e-3)
        shN_lr = self.config.get('lr_color_rest', sh0_lr / 20.0)
        self.optimizers = {
            "means":     Adam([{"params": self.splats["means"]}],     lr=self.config['lr_xyz'],      eps=1e-15),
            "scales":    Adam([{"params": self.splats["scales"]}],    lr=self.config['lr_scaling'],  eps=1e-15),
            "quats":     Adam([{"params": self.splats["quats"]}],     lr=self.config['lr_rotation'], eps=1e-15),
            "opacities": Adam([{"params": self.splats["opacities"]}], lr=self.config['lr_opacity'],  eps=1e-15),
            "sh0":       Adam([{"params": self.splats["sh0"]}],       lr=sh0_lr,                     eps=1e-15),
            "shN":       Adam([{"params": self.splats["shN"]}],       lr=shN_lr,                     eps=1e-15),
        }
        
        # Restore the scheduler state for the new 'means' optimizer
        self.scheduler = ExponentialLR(self.optimizers["means"], gamma=self.config['lr_decay'])
        # Step the scheduler up to the current iteration
        for _ in range(self.iteration):
            self.scheduler.step()
            
        print(f"[Trainer] Resuming from iteration {self.iteration}")


if __name__ == "__main__":
    # Config
    config = {
        # 'lr_xyz': 0.00000,
        'lr_xyz': 0.0002,
        # 'lr_xyz': 0.00001,
        'lr_color': 0.0025,
        'lr_opacity': 0.05,
        'lr_scaling': 0.001,
        'lr_rotation': 0.001,
        'lr_decay': 0.9999,
        
        'checkpoint_interval': 50,
        'init_scale_lidar': 0.10,   # Size in meters for LiDAR points
        # 'init_scale_lidar': 0.10,    # Size in meters for LiDAR points
        'init_scale_sky': 0.5,
        'init_opacity': 0.5,
        
        # ---- Densification window: ~150 cycles within current iter budget ----
        # 200 epochs * ceil(1314/64)=21 batches ≈ 4200 iters; (3850-100)/25 = 150 cycles
        'refine_start_iter': 200,
        'refine_stop_iter': 3800,
        'refine_every':     50,
        'checkpoint_path':  None,
        # 'checkpoint_path': '/mnt/HDD6/miayan/omega/sdc_hw/3dgs_slam/sdc_3dgs_reconstruction/output/Track1/checkpoint_epoch_100.pt',
        # 'checkpoint_path': '/mnt/HDD6/miayan/omega/sdc_hw/3dgs_slam/sdc_3dgs_reconstruction/output/Track2/checkpoint_epoch_150.pt',

        # Densification strategy (gsplat DefaultStrategy params)
        'grow_grad2d': 0.0002,
        'prune_opa':   0.01,

        # ---- Spherical harmonics ----
        'sh_degree':           3,       # 0=DC only (Lambertian); 3 = full 16 SH bands
        'sh_degree_interval':  2000, # iters per warmup band increment

        # Auxiliary losses (set weight > 0 to enable; 0 = disabled)
        'w_ssim':               0.2,    # standard 3DGS: L = 0.8 L1 + 0.2 (1 - SSIM)
        'w_depth':              0.05,   # weight on LiDAR depth L1 (range ~ meters)
        'w_scale_reg':          0.01,   # weight on Gaussian scale hinge
        'scale_reg_threshold':  0.5,    # only penalize axes > this many meters

        # ----- Depth-guided densification (silhouette + DA3 metric depth) -----
        # Disabled when start/end == -1.  Both bounds are 0-indexed epochs (inclusive).
        'start_depth_densify_epoch':       10,
        'end_depth_densify_epoch':         31,
        'depth_densify_interval':          20,     # every N epochs in the range
        'silhouette_threshold':            0.2,    # alpha < this  →  treated as a hole
        'depth_densify_conf_percentile':   30,     # only keep pixels above this DA3-conf percentile (per frame, non-sky)
        'depth_densify_min_depth':         1.0,    # meters; reject very close DA3 depth
        'depth_densify_max_depth':         100.0,  # meters; reject very far DA3 depth
        'depth_densify_init_scale':        0.05,   # match init_scale_lidar
        'depth_densify_init_opacity':      0.1,    # low — let split/clone/prune adjust
        'depth_densify_frames_per_pass':   100,     # subsample frames per pass
        'depth_densify_max_points_per_pass': 50000,
    }
    
    num_epochs = 30
    batch_size = 32
    
    data_dir = "/home/lab605/SLAM_3DGS/Track2"
    output_dir = "output/Track2"
    
    trainer = Trainer(data_dir, output_dir, config)

    ckpt = config.get('checkpoint_path')
    if ckpt and os.path.exists(ckpt):
        trainer.load_checkpoint(ckpt)

    trainer.train(num_epochs=num_epochs, batch_size=batch_size)
    trainer.save_ply("gaussian_reconstruction.ply")
    print("\nTraining complete!")
