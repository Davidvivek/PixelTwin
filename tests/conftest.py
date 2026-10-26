"""
conftest.py — shared pytest fixtures for PixelTwin tests.

Fixtures defined here are auto-discovered by pytest and available to all
test modules without explicit import.
"""

import numpy as np
import pytest

from pixeltwin.seed import set_seed, SEED


@pytest.fixture(autouse=True)
def _seed_each_test():
    """Re-seed all RNGs before every test to ensure isolation and reproducibility."""
    set_seed(SEED)
    yield


@pytest.fixture
def tiny_gray_image() -> np.ndarray:
    """Return a deterministic 64×64 uint8 grayscale image for fast unit tests."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def tiny_color_image() -> np.ndarray:
    """Return a deterministic 64×64×3 uint8 BGR image (OpenCV convention)."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
