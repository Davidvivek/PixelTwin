"""
seed.py — global random-seed utility.

Call ``set_seed(SEED)`` at the top of every script that uses randomness so
that results are reproducible across runs and machines.

Usage
-----
    from pixeltwin.seed import set_seed, SEED
    set_seed()          # uses the project-wide default
    set_seed(42)        # override
"""

import random
import numpy as np

# Project-wide default seed — never change this value; override per-script if needed.
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python's ``random``, NumPy, and (if available) PyTorch.

    Parameters
    ----------
    seed:
        Integer seed value.  Defaults to the project constant ``SEED = 42``.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Optional: seed PyTorch only if it is importable (not required for M1).
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic CuDNN (may slow training; acceptable for a benchmark).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed — fine for pure-classical pipeline
