"""
Render test images using a trained 3DGS PLY at the interpolated test poses.

Reads:
  - PLY:           output/<Track>/gaussian_reconstruction.ply
  - Test poses:    test_pose_list_<Track>.txt
  - Test frames:   test_frame_list_<Track>.txt
Writes:
  - test_submission/<Track>/<frame_id>.png   (30 PNGs, ready to zip with submission.csv)
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from gsplat.rendering import rasterization


# Hardcoded intrinsics — kept consistent with scene/dataset_readers.py
INTRINSICS = {
    "Track1": dict(fx=653.143433778113, fy=657.670567367976,
                   cx=299.1738577337179, cy=236.60674857178367,
                   width=640, height=480),
    "Track2": dict(fx=1040.18078, fy=1038.55506,
                   cx=720.04463,  cy=464.33648,
                   width=1440, height=928),
}


class StandaloneRenderer:
    def __init__(self, ply_path, K, device="cuda"):
        self.device = torch.device(device)
        self.K = K.to(self.device)
        self.splats, self.sh_degree = self.load_ply(ply_path)

    def load_ply(self, path):
        """Load standard 3DGS PLY. Variable property count → auto-detect sh_degree.

        Expected property order (matches train.write_3dgs_ply):
          x,y,z, nx,ny,nz, f_dc_0..2, f_rest_0..{3*K_rest-1}, opacity, scale_0..2, rot_0..3
        """
        print(f"Loading splats from {path}...")
        prop_names = []
        num_points = 0
        with open(path, 'rb') as f:
            while True:
                line = f.readline().decode('ascii').strip()
                if line == "end_header":
                    break
                if line.startswith("element vertex"):
                    num_points = int(line.split()[-1])
                elif line.startswith("property float"):
                    prop_names.append(line.split()[-1])
            n_props = len(prop_names)
            data = np.fromfile(f, dtype=np.float32,
                               count=num_points * n_props).reshape(num_points, n_props)

        idx = {name: i for i, name in enumerate(prop_names)}
        def cols(names):
            return data[:, [idx[n] for n in names]]

        means     = cols(["x", "y", "z"])
        scales    = cols(["scale_0", "scale_1", "scale_2"])
        quats     = cols(["rot_0", "rot_1", "rot_2", "rot_3"])
        opacities = cols(["opacity"])
        sh0_dc    = cols(["f_dc_0", "f_dc_1", "f_dc_2"])    # (N, 3)

        rest_count = sum(1 for n in prop_names if n.startswith("f_rest_"))
        if rest_count > 0:
            K_rest = rest_count // 3
            rest_flat = cols([f"f_rest_{i}" for i in range(rest_count)])    # (N, 3*K_rest), transposed
            shN_np = rest_flat.reshape(num_points, 3, K_rest).transpose(0, 2, 1)  # (N, K_rest, 3)
        else:
            shN_np = np.zeros((num_points, 0, 3), dtype=np.float32)

        sh_degree = int(round((1 + (rest_count // 3)) ** 0.5)) - 1
        print(f"  {num_points} Gaussians, {n_props} props/pt, sh_degree={sh_degree}")

        device = self.device
        splats = nn.ParameterDict({
            "means":     nn.Parameter(torch.from_numpy(means).to(device)),
            "scales":    nn.Parameter(torch.from_numpy(scales).to(device)),
            "quats":     nn.Parameter(torch.from_numpy(quats).to(device)),
            "opacities": nn.Parameter(torch.from_numpy(opacities).to(device)),
            "sh0":       nn.Parameter(torch.from_numpy(sh0_dc[:, None, :]).to(device)),
            "shN":       nn.Parameter(torch.from_numpy(shN_np).to(device)),
        })
        return splats, sh_degree

    @torch.no_grad()
    def render_rgb(self, camtoworld, width, height):
        means     = self.splats["means"]
        quats     = self.splats["quats"]
        scales    = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)

        # Full SH coefficients [N, K, 3]; gsplat evaluates view-dependent color
        colors_sh = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)

        viewmat = torch.linalg.inv(camtoworld)

        features, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_sh,
            viewmats=viewmat.unsqueeze(0),
            Ks=self.K.unsqueeze(0),
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1000.0,
            sh_degree=self.sh_degree,
            render_mode="RGB",
        )
        image_rgb = (features[0] + (1.0 - alphas[0]) * 0.0).permute(2, 0, 1)
        return image_rgb


def render_track(track: str):
    repo_root = Path(__file__).resolve().parent

    PLY_FILE        = repo_root / "output" / track / "gaussian_reconstruction.ply"
    track_small = track.lower()
    POSE_LIST_FILE  = repo_root / f"{track_small}/{track_small}_test_pose_list.txt"
    FRAME_LIST_FILE = repo_root / f"{track_small}/{track_small}_test_frame_list.txt"
    if not FRAME_LIST_FILE.exists():
        FRAME_LIST_FILE = repo_root / "test_frame_list.txt"
        print(f"[warn] using generic {FRAME_LIST_FILE.name}")
    OUTPUT_DIR      = repo_root / "test_submission" / track
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Frames + poses
    with open(FRAME_LIST_FILE) as f:
        frame_ids = [line.strip().split('.')[0] for line in f if line.strip()]
    raw_poses = np.loadtxt(POSE_LIST_FILE)
    test_poses = [torch.from_numpy(p.reshape(4, 4)).float() for p in raw_poses]
    assert len(frame_ids) == len(test_poses), \
        f"frame_list ({len(frame_ids)}) != pose_list ({len(test_poses)})"

    # 2. Intrinsics
    p = INTRINSICS[track]
    K = torch.tensor([[p['fx'], 0, p['cx']],
                      [0, p['fy'], p['cy']],
                      [0, 0,        1]]).float()
    width, height = p['width'], p['height']

    # 3. Renderer
    renderer = StandaloneRenderer(str(PLY_FILE), K)

    # 4. Render each test view
    print(f"Rendering {len(frame_ids)} {track} views @ {width}x{height} ...")
    for fid, pose in zip(tqdm(frame_ids), test_poses):
        rgb = renderer.render_rgb(pose.to(renderer.device), width, height)
        rgb_np = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(OUTPUT_DIR / f"{fid}.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

    print(f"\n[done] {track} renders saved to {OUTPUT_DIR}")
    print(f"\nNext step: zip the PNGs + submission.csv:")
    print(f"  cd {repo_root}")
    print(f"  cp submission.csv test_submission/{track}/")
    print(f"  cd test_submission/{track} && zip submission_{track}.zip *.png submission.csv")


if __name__ == "__main__":
    TRACK = sys.argv[1] if len(sys.argv) > 1 else "Track2"
    render_track(TRACK)
