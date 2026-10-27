"""
tests/test_transforms.py — comprehensive unit tests for synthetic transformations.

Tests coverage:
  - TransformSpec properties, stringification, filesystem slugs
  - Individual primitives:
      * jpeg_compress (different quality levels, color & grayscale)
      * resize_image (fractional scaling, error cases)
      * center_crop (fractional cropping, coordinate bounds, error cases)
      * rotate_image (angle rotation, reflection border, shape preservation)
      * adjust_brightness (positive & negative shifts, uint8 clipping)
      * adjust_contrast (scaling around mid-gray, uint8 clipping)
      * add_gaussian_noise (seeded determinism, distribution limits)
      * apply_combo (chained sequential transforms, seeding)
  - apply_transform dispatch & error handling
  - ALL_SPECS registry completeness & generate_all_transforms execution
  - In-place immutability: ensuring input arrays are never mutated
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from pixeltwin.seed import SEED
from pixeltwin.transforms import (
    ALL_SPECS,
    TransformSpec,
    add_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
    apply_combo,
    apply_transform,
    center_crop,
    generate_all_transforms,
    jpeg_compress,
    resize_image,
    rotate_image,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def color_img() -> np.ndarray:
    """Deterministic 128x128x3 BGR uint8 image."""
    rng = np.random.default_rng(SEED)
    return rng.integers(20, 235, size=(128, 128, 3), dtype=np.uint8)


@pytest.fixture
def gray_img() -> np.ndarray:
    """Deterministic 128x128 grayscale uint8 image."""
    rng = np.random.default_rng(SEED + 1)
    return rng.integers(20, 235, size=(128, 128), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. TransformSpec tests
# ---------------------------------------------------------------------------

class TestTransformSpec:
    """Test TransformSpec construction, string format, slugs, and serializability."""

    def test_str_and_slug(self) -> None:
        spec = TransformSpec("jpeg", {"quality": 90})
        assert str(spec) == "jpeg(quality=90)"
        assert spec.slug() == "jpeg_quality90"

        spec_float = TransformSpec("resize", {"scale": 0.75})
        assert str(spec_float) == "resize(scale=0.75)"
        assert spec_float.slug() == "resize_scale0p75"

        spec_neg = TransformSpec("brightness", {"shift": -30})
        assert str(spec_neg) == "brightness(shift=-30)"
        assert spec_neg.slug() == "brightness_shiftneg30"

    def test_json_serializable(self) -> None:
        """Every spec in ALL_SPECS must be JSON-serialisable for manifest persistence."""
        for spec in ALL_SPECS:
            dumped = json.dumps(spec.to_dict())
            loaded = json.loads(dumped)
            spec_restored = TransformSpec.from_dict(loaded)
            assert spec_restored == spec


# ---------------------------------------------------------------------------
# 2. Primitive Transform Tests
# ---------------------------------------------------------------------------

class TestPrimitives:
    """Test each transform primitive for shape, dtype, clipping, and invariance."""

    @pytest.mark.parametrize("quality", [90, 70, 50, 30])
    def test_jpeg_compress(self, color_img: np.ndarray, gray_img: np.ndarray, quality: int) -> None:
        out_c = jpeg_compress(color_img, quality=quality)
        assert out_c.shape == color_img.shape
        assert out_c.dtype == np.uint8

        out_g = jpeg_compress(gray_img, quality=quality)
        assert out_g.shape == gray_img.shape
        assert out_g.dtype == np.uint8

    @pytest.mark.parametrize("scale", [0.75, 0.50, 0.25])
    def test_resize_image(self, color_img: np.ndarray, scale: float) -> None:
        out = resize_image(color_img, scale=scale)
        H, W = color_img.shape[:2]
        expected_shape = (round(H * scale), round(W * scale), 3)
        assert out.shape == expected_shape
        assert out.dtype == np.uint8

    def test_resize_image_invalid_scale(self, color_img: np.ndarray) -> None:
        with pytest.raises(ValueError):
            resize_image(color_img, scale=0.0)
        with pytest.raises(ValueError):
            resize_image(color_img, scale=1.5)

    @pytest.mark.parametrize("crop_frac", [0.90, 0.80, 0.70])
    def test_center_crop(self, color_img: np.ndarray, crop_frac: float) -> None:
        out = center_crop(color_img, crop_frac=crop_frac)
        H, W = color_img.shape[:2]
        expected_shape = (int(H * crop_frac), int(W * crop_frac), 3)
        assert out.shape == expected_shape
        assert out.dtype == np.uint8

    def test_center_crop_invalid_frac(self, color_img: np.ndarray) -> None:
        with pytest.raises(ValueError):
            center_crop(color_img, crop_frac=0.0)
        with pytest.raises(ValueError):
            center_crop(color_img, crop_frac=1.2)

    @pytest.mark.parametrize("degrees", [5, 15, 45, 90])
    def test_rotate_image(self, color_img: np.ndarray, degrees: float) -> None:
        out = rotate_image(color_img, degrees=degrees)
        assert out.shape == color_img.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("shift", [30, -30, 60, -60])
    def test_adjust_brightness(self, color_img: np.ndarray, shift: int) -> None:
        out = adjust_brightness(color_img, shift=shift)
        assert out.shape == color_img.shape
        assert out.dtype == np.uint8
        assert out.min() >= 0
        assert out.max() <= 255

    @pytest.mark.parametrize("factor", [1.3, 0.7])
    def test_adjust_contrast(self, color_img: np.ndarray, factor: float) -> None:
        out = adjust_contrast(color_img, factor=factor)
        assert out.shape == color_img.shape
        assert out.dtype == np.uint8
        assert out.min() >= 0
        assert out.max() <= 255

    def test_gaussian_noise_determinism_and_bounds(self, color_img: np.ndarray) -> None:
        out1 = add_gaussian_noise(color_img, sigma=15.0, seed=42)
        out2 = add_gaussian_noise(color_img, sigma=15.0, seed=42)
        out3 = add_gaussian_noise(color_img, sigma=15.0, seed=99)

        # Same seed produces exact same output
        assert np.array_equal(out1, out2)
        # Different seed produces different output
        assert not np.array_equal(out1, out3)
        # Value ranges
        assert out1.dtype == np.uint8
        assert out1.min() >= 0 and out1.max() <= 255

    def test_apply_combo(self, color_img: np.ndarray) -> None:
        steps = [
            ("jpeg", {"quality": 80}),
            ("resize", {"scale": 0.5}),
            ("brightness", {"shift": 20}),
        ]
        out = apply_combo(color_img, steps=steps, seed=123)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# 3. apply_transform & ALL_SPECS registry tests
# ---------------------------------------------------------------------------

class TestDispatchAndRegistry:
    """Test central dispatch, error checking, and ALL_SPECS completeness."""

    def test_apply_transform_unknown_raises(self, color_img: np.ndarray) -> None:
        spec = TransformSpec("unsupported_transform", {})
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_transform(color_img, spec)

    def test_all_specs_valid(self, color_img: np.ndarray) -> None:
        """Every spec in ALL_SPECS must execute without raising errors."""
        for spec in ALL_SPECS:
            out = apply_transform(color_img, spec, seed=SEED)
            assert out.dtype == np.uint8
            assert out.min() >= 0 and out.max() <= 255

    def test_generate_all_transforms(self, color_img: np.ndarray) -> None:
        results = generate_all_transforms(color_img, seed=SEED)
        assert len(results) == len(ALL_SPECS)
        for spec, img in results:
            assert isinstance(spec, TransformSpec)
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint8

    def test_immutability(self, color_img: np.ndarray) -> None:
        """Input image must never be modified in-place."""
        clone = color_img.copy()
        for spec in ALL_SPECS:
            _ = apply_transform(color_img, spec, seed=SEED)
            assert np.array_equal(color_img, clone), f"spec {spec} mutated input in-place!"
