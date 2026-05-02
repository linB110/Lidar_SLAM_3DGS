"""
3D Gaussian Splatting Trainer - Rewritten following gsplat official example.
Uses step_pre_backward / step_post_backward pattern for proper densification.
"""

import os
import sys
from pathlib import Path

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
import utils.loss_tracker 
from utils.loss import ssim, depth_smooth_loss

from math import exp

def create_gaussians_with_optimizers(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    init_scales: torch.Tensor,
    init_opacity: float = 0.5,
    sh_degree: int = 3,  # 0 means only DC component (RGB colors)
    means_lr: float = 1.6e-4,
    scale_lr: float = 5e-3,
    opacity_lr: float = 5e-2,
    quat_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    device: str = "cuda",
) -> tuple:
    """
    Initialize Gaussians from point cloud with fixed scale.
    Returns: (splats ParameterDict, optimizers dict)
    """
    # Ensure tensors are on device
    points = points.to(device).float()
    rgbs = rgbs.to(device).float()
    rgbs = torch.clamp(rgbs, 0, 1)
    
    N = points.shape[0]
    
    # Initialize means (positions)
    means = points  # [N, 3]
    
    # Use the per-point scales and convert to log space
    # gsplat expects scales in log space: s_log = log(s_actual)
    scales = torch.log(init_scales.to(device))  # [N, 3]
    
    # Initialize rotations (identity quaternions: [0, 0, 0, 1])
    quats = torch.zeros((N, 4), device=device)
    quats[:, 3] = 1.0  # [N, 4]
    
    # Initialize opacities (logit space)
    init_opacity = np.clip(init_opacity, 0.001, 0.999)
    logit_opacity = np.log(init_opacity / (1 - init_opacity))
    opacities = torch.ones((N, 1), device=device) * logit_opacity  # [N, 1]
    
    # Initialize colors (DC SH component for RGB)
    # Convert RGB to SH DC space
    colors_sh = RGB2SH(rgbs)  # [N, 3]
    sh0 = colors_sh.unsqueeze(1)  # [N, 1, 3]
    
    # Create ParameterDict (mimics gsplat's approach)
    splats = nn.ParameterDict({
        "means": nn.Parameter(means),
        "scales": nn.Parameter(scales),
        "quats": nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "sh0": nn.Parameter(sh0),
    })
    
    # Create optimizers for each parameter with different learning rates
    optimizers = {
        "means": Adam([{"params": splats["means"], "lr": means_lr}], eps=1e-15),
        "scales": Adam([{"params": splats["scales"], "lr": scale_lr}], eps=1e-15),
        "quats": Adam([{"params": splats["quats"], "lr": quat_lr}], eps=1e-15),
        "opacities": Adam([{"params": splats["opacities"], "lr": opacity_lr}], eps=1e-15),
        "sh0": Adam([{"params": splats["sh0"], "lr": sh0_lr}], eps=1e-15),
    }
    
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
        self.splats, self.optimizers = create_gaussians_with_optimizers(
            points=combined_points,
            rgbs=combined_colors,
            init_scales=combined_scales,
            init_opacity=init_opacity,
            means_lr=config['lr_xyz'],
            scale_lr=config['lr_scaling'],
            opacity_lr=config['lr_opacity'],
            quat_lr=config['lr_rotation'],
            device=str(self.device),
        )
        
        print(f"[Trainer] Initialized {len(self.splats['means'])} Gaussians")
        
        # Setup learning rate scheduler (only for means which has schedule)
        self.scheduler = ExponentialLR(self.optimizers["means"], gamma=config['lr_decay'])
        
        # Setup strategy
        self.strategy = DefaultStrategy(
            prune_opa=config.get('prune_opa', 0.01),
            grow_grad2d=config.get('grow_grad2d', 0.0003),
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
    
    def rasterize_splats(self, camtoworld: torch.Tensor, K: torch.Tensor, width: int, height: int):
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)
        
        # 1. Calculate the depth of each Gaussian in camera space
        viewmat = torch.linalg.inv(camtoworld) # [4, 4]
        # Transform means to camera space: P_cam = R*P_world + t
        # We only need the Z-component (depth)
        means_h = torch.cat([means, torch.ones((means.shape[0], 1), device=self.device)], dim=-1)
        p_cam = (viewmat @ means_h.T).T
        gauss_depths = p_cam[:, 2:3] # [N, 1]

        # 2. Combine RGB (from SH) and Depth into a 4-channel 'color' tensor
        # C0 is the standard SH constant (1 / (2 * sqrt(pi)))
        C0 = 0.28209479177387814
        sh0 = self.splats["sh0"].squeeze(1) # [N, 3]
        rgb = sh0 * C0 + 0.5
        render_features = torch.cat([rgb, gauss_depths], dim=-1) # [N, 4]
        
        # 3. Render
        K_batch = K.unsqueeze(0)
        viewmat_batch = viewmat.unsqueeze(0)
        
        render_features, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=render_features, # Pass 4 channels [RGB + Depth]
            viewmats=viewmat_batch,
            Ks=K_batch,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1000.0, # CRITICAL: Increase for sky coverage
        )
        
        self.last_info = info
        
        # 4. Separate RGB and Depth
        # render_features is [1, H, W, 4]
        image_rgb = render_features[0, ..., :3]
        image_depth = render_features[0, ..., 3] # This is your [H, W] rendered depth
        
        # Composite with background
        image_rgb = image_rgb + (1.0 - render_alphas[0]) * 0.0 # Black background
        
        # Return both for the train_step to use
        return image_rgb.permute(2, 0, 1), image_depth
    
    def train_step(self, frame_indices: np.ndarray) -> float:
        """Run one training step."""
        # Zero gradients for all optimizers
        for opt in self.optimizers.values():
            opt.zero_grad()
        
        total_loss = 0.0
        num_frames = len(frame_indices)
        
        # Render all frames in batch
        for frame_idx in frame_indices:
            target_image = self.dataset.get_image(frame_idx)  # [3, H, W]
            target_image = target_image.to(self.device)
            mask = self.dataset.get_mask(frame_idx).to(self.device)
            # Mask: 0 for objects (keep), 255 for sky (ignore)
            loss_mask = (mask == 0).float().unsqueeze(0)
            
            pose_c2w = self.poses_c2w[frame_idx]
            
            # Render
            rendered_rgb, rendered_depth = self.rasterize_splats(
                pose_c2w,
                self.K,
                self.dataset.image_width,
                self.dataset.image_height,
            )

            # Apply mask to images before loss calculation
            masked_render = rendered_rgb * loss_mask
            masked_target = target_image * loss_mask
            
            l1_val = F.l1_loss(masked_render, masked_target)
            
            # ==========================================================================
            # You can implement other loss function (e.g., depth loss) and combine them with weights
            # ==========================================================================

            # SSIM
            ssim_val = ssim(masked_render, masked_target)
            
            # depth smooth loss
            depth_val = depth_smooth_loss(rendered_depth, masked_target)
            
            loss = 0.1 * l1_val + 0.5 * ssim_val + 0.4 * depth_val
            total_loss += loss
        
        total_loss = total_loss / num_frames
        
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
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 5) == 0:
                self.save_checkpoint(epoch + 1)
            
            # LR schedule
            self.scheduler.step()
    
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
        """Export as PLY file."""
        import struct
        
        means = self.splats["means"].detach().cpu().numpy()
        scales = self.splats["scales"].detach().cpu().numpy()
        quats = self.splats["quats"].detach().cpu().numpy()
        opacities = self.splats["opacities"].squeeze(-1).detach().cpu().numpy()
        sh0 = self.splats["sh0"].squeeze(1).detach().cpu().numpy()
        
        num_points = len(means)
        
        ply_path = self.output_dir / filename
        
        with open(ply_path, 'wb') as f:
            # Header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {num_points}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float scale_0\n")
            f.write(b"property float scale_1\n")
            f.write(b"property float scale_2\n")
            f.write(b"property float rot_0\n")
            f.write(b"property float rot_1\n")
            f.write(b"property float rot_2\n")
            f.write(b"property float rot_3\n")
            f.write(b"property float opacity\n")
            f.write(b"property float f_dc_0\n")
            f.write(b"property float f_dc_1\n")
            f.write(b"property float f_dc_2\n")
            f.write(b"end_header\n")
            
            # Data
            for i in range(num_points):
                f.write(struct.pack('f', means[i, 0]))
                f.write(struct.pack('f', means[i, 1]))
                f.write(struct.pack('f', means[i, 2]))
                f.write(struct.pack('f', scales[i, 0]))
                f.write(struct.pack('f', scales[i, 1]))
                f.write(struct.pack('f', scales[i, 2]))
                f.write(struct.pack('f', quats[i, 0]))
                f.write(struct.pack('f', quats[i, 1]))
                f.write(struct.pack('f', quats[i, 2]))
                f.write(struct.pack('f', quats[i, 3]))
                f.write(struct.pack('f', opacities[i]))
                f.write(struct.pack('f', sh0[i, 0]))
                f.write(struct.pack('f', sh0[i, 1]))
                f.write(struct.pack('f', sh0[i, 2]))
        
        print(f"[Trainer] Saved PLY: {ply_path}")

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
        self.optimizers = {
            "means": Adam([{"params": self.splats["means"], "lr": self.config['lr_xyz']}], eps=1e-15),
            "scales": Adam([{"params": self.splats["scales"], "lr": self.config['lr_scaling']}], eps=1e-15),
            "quats": Adam([{"params": self.splats["quats"], "lr": self.config['lr_rotation']}], eps=1e-15),
            "opacities": Adam([{"params": self.splats["opacities"], "lr": self.config['lr_opacity']}], eps=1e-15),
            "sh0": Adam([{"params": self.splats["sh0"], "lr": self.config.get('lr_color', 2.5e-3)}], eps=1e-15),
        }
        
        # Restore the scheduler state for the new 'means' optimizer
        self.scheduler = ExponentialLR(self.optimizers["means"], gamma=self.config['lr_decay'])
        # Step the scheduler up to the current iteration
        for _ in range(self.iteration):
            self.scheduler.step()
            
        print(f"[Trainer] Resuming from iteration {self.iteration}")


if __name__ == "__main__":
    
    # training data length
    num_frames = 245
    batch_size = 8
    num_epochs = 100
    
    iters_per_epoch = (num_frames + batch_size - 1)
    total_iters = iters_per_epoch * num_epochs
    
    refine_start_iter = int(total_iters * 0.02)
    refine_stop_iter = int(total_iters * 0.5)  
    reset_every = int(total_iters * 0.1)
    
    # Config
    config = {
        'lr_xyz': 0.00016,
        'lr_color': 0.0025,
        'lr_opacity': 0.05,
        'lr_scaling': 0.005,
        'lr_rotation': 0.001,
        'lr_decay': 0.9999,
        'checkpoint_interval': 10,
        'init_scale_lidar': 0.15,    # Size in meters for LiDAR points
        'init_scale_sky': 0.5,
        'init_opacity': 0.7,
        'refine_start_iter': refine_start_iter,  # Start densification at iteration x
        'refine_stop_iter': refine_stop_iter,  # Stop densification at iteration x
        'refine_every': reset_every,  # Densify every x iterations
        'checkpoint_path': None, # Set to None if starting fresh
    }
    
    data_dir = "/home/lab605/SLAM_3DGS/Track1/"
    output_dir = "output"
    
    trainer = Trainer(data_dir, output_dir, config)

    ckpt = config.get('checkpoint_path')
    if ckpt and os.path.exists(ckpt):
        trainer.load_checkpoint(ckpt)

    trainer.train(num_epochs, batch_size)
    trainer.save_ply("gaussian_reconstruction.ply")
    print("\nTraining complete!")
