"""
General utility functions.
"""

import os
import numpy as np
import torch


def mkdir_p(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def safe_state(loss_for_log):
    """Utility function for logging."""
    return {
        'loss': loss_for_log,
    }


def inverse_sigmoid(x):
    """Inverse sigmoid function."""
    return torch.log(x / (1 - x))

