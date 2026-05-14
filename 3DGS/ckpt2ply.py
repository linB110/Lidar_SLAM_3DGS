"""
Convert a saved training checkpoint (.pt) → standard 3DGS PLY.

Useful when training is paused/killed before the final save_ply() call, or when
you want to render submissions from any intermediate checkpoint.

Usage:
    python ckpt2ply.py output/Track1/checkpoint_epoch_250.pt
    python ckpt2ply.py output/Track1/checkpoint_epoch_250.pt -o output/Track1/gaussian_reconstruction.ply
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import write_3dgs_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Path to .pt checkpoint")
    ap.add_argument("-o", "--output", default=None,
                    help="Output .ply path (default: same dir, gaussian_reconstruction.ply)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"[error] checkpoint not found: {ckpt_path}")

    out_path = Path(args.output) if args.output else ckpt_path.parent / "gaussian_reconstruction.ply"

    print(f"[ckpt2ply] Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splat_data = ckpt["splats"]
    print(f"[ckpt2ply] iteration={ckpt.get('iteration', '?')}, "
          f"epoch={ckpt.get('epoch', '?')}, keys={list(splat_data.keys())}")

    splats = {k: nn.Parameter(v) for k, v in splat_data.items()}

    # Backward compat: checkpoints from before sh_degree=3 update have no shN
    if "shN" not in splats:
        n = splats["means"].shape[0]
        splats["shN"] = nn.Parameter(torch.zeros(n, 0, 3))
        print("[ckpt2ply] no shN in checkpoint → assuming sh_degree=0")

    write_3dgs_ply(splats, out_path)


if __name__ == "__main__":
    main()
