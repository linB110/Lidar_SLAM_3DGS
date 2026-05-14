"""
SH utilities for color encoding/decoding
"""

import numpy as np
import torch

# SH DC component basis coefficient
C0 = 0.28209479177387814

def RGB2SH(rgb):
    """
    Convert RGB [0,1] to spherical harmonics DC component.
    
    Args:
        rgb: RGB values in [0, 1] range
        
    Returns:
        SH DC component (can be outside [-1, 1] range after transformation)
    """
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    """
    Convert spherical harmonics DC component back to RGB.
    
    Args:
        sh: SH DC component values
        
    Returns:
        RGB values in [0, 1] range
    """
    return sh * C0 + 0.5
