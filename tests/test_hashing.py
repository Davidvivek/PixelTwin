"""
tests/test_hashing.py — invariance and correctness tests for hashing.py.

Test philosophy
---------------
Every threshold used below is backed by a comment explaining *why* it is
set where it is.  We intentionally use generous (but not meaningless)
thresholds so the suite is stable across minor NumPy/SciPy version
differences, while still catching obviously broken implementations.

Test image
----------
We use a 256×256 synthetic image with strong low-frequency structure
(smooth gradient + low-frequency sine wave overlay) rather than random
noise, because perceptual hashes are designed for natural-image statistics.
Random noise would make pHash unreliable by design.
"""

from __future__ import annotations

import io
import numpy as np
import cv2
import pytest

from pixeltwin.hashing import (
    ahash,
    dhash,
    phash,
    hamming_distance,
    hamming_distances,
)
from pixeltwin.seed import SEED, set_seed


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def natural_image() -> np.ndarray:
    """256×256 BGR uint8 image with strong low-frequency, block-like structure.

    Built from 16 flat-colour patches arranged in a 4×4 grid so that the
    dominant energy sits in the lowest DCT coefficients.  This is the regime
    pHash is designed for: JPEG quantisation barely touches the low-frequency
    block, so the hash is stable across quality levels.

    We deliberately avoid a pure smooth-gradient or sine-wave image because
    those place many coefficients near the median threshold, making the hash
    brittle to small perturbations (JPEG ringing, brightness shifts).

    Returned as BGR (3-channel) to exercise the BGR→Gray path inside the
    hash functions.
    """
    H, W = 256, 256
    # 16 flat-colour blocks (4×4 grid), values spread across [40, 220]
    rng = np.random.default_rng(SEED)
    values = rng.integers(40, 220, size=(4, 4), dtype=np.uint8)
    gray = np.zeros((H, W), dtype=np.uint8)
    bh, bw = H // 4, W // 4
    for r in range(4):
        for c in range(4):
            gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw] = values[r, c]
    # Smooth the block edges slightly so JPEG doesn't add severe ringing
    gray = cv2.GaussianBlur(gray, (15, 15), sigmaX=5.0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def natural_image_gray(natural_image: np.ndarray) -> np.ndarray:
    """Grayscale version of the natural test image."""
    return cv2.cvtColor(natural_image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Helper: JPEG round-trip (in-memory, no file I/O)
# ---------------------------------------------------------------------------

def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Encode *img* as JPEG at *quality*, decode and return the same dtype/shape."""
    flag = cv2.IMWRITE_JPEG_QUALITY
    ok, buf = cv2.imencode(".jpg", img, [flag, quality])
    assert ok, f"cv2.imencode failed for quality={quality}"
    decode_flag = cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR
    return cv2.imdecode(buf, decode_flag)


# ---------------------------------------------------------------------------
# 1. Identity: same image → Hamming distance == 0
# ---------------------------------------------------------------------------

class TestHashIdentity:
    """Hashing the exact same image twice must give distance 0."""

    def test_ahash_identity(self, natural_image: np.ndarray) -> None:
        h = ahash(natural_image)
        assert hamming_distance(h, h) == 0

    def test_dhash_identity(self, natural_image: np.ndarray) -> None:
        h = dhash(natural_image)
        assert hamming_distance(h, h) == 0

    def test_phash_identity(self, natural_image: np.ndarray) -> None:
        h = phash(natural_image)
        assert hamming_distance(h, h) == 0

    def test_ahash_gray_vs_bgr(self, natural_image: np.ndarray,
                                natural_image_gray: np.ndarray) -> None:
        """aHash of a BGR image must equal aHash of its grayscale version."""
        assert ahash(natural_image) == ahash(natural_image_gray)

    def test_dhash_gray_vs_bgr(self, natural_image: np.ndarray,
                                natural_image_gray: np.ndarray) -> None:
        assert dhash(natural_image) == dhash(natural_image_gray)

    def test_phash_gray_vs_bgr(self, natural_image: np.ndarray,
                                natural_image_gray: np.ndarray) -> None:
        assert phash(natural_image) == phash(natural_image_gray)


# ---------------------------------------------------------------------------
# 2. JPEG recompression invariance
# ---------------------------------------------------------------------------
# pHash is designed to survive JPEG because the low-frequency DCT block it
# hashes is nearly identical to what JPEG retains.  We allow up to 10 bits
# of change at quality=90 (in practice it's usually 0–2 bits).

class TestJpegInvariance:
    """Small Hamming distance after JPEG encode/decode."""

    # (quality, max_bits_changed)
    # Thresholds set from measured distances on the block-grid test image
    # (see PROGRESS.md C1 section).  A buffer of +4 bits is added above the
    # measured value so the suite is stable without being trivially loose.
    # Measured: q=90→4, q=70→6, q=50→6, q=30→6 bits.
    _CASES_PHASH = [
        (90, 8),
        (70, 10),
        (50, 10),
        (30, 12),
    ]

    @pytest.mark.parametrize("quality,max_dist", _CASES_PHASH)
    def test_phash_jpeg(self, quality: int, max_dist: int,
                        natural_image: np.ndarray) -> None:
        orig = phash(natural_image)
        compressed = phash(_jpeg(natural_image, quality))
        dist = hamming_distance(orig, compressed)
        assert dist <= max_dist, (
            f"pHash JPEG q={quality}: distance {dist} > threshold {max_dist}"
        )

    @pytest.mark.parametrize("quality,max_dist", [(90, 10), (70, 12)])
    def test_ahash_jpeg(self, quality: int, max_dist: int,
                        natural_image: np.ndarray) -> None:
        orig = ahash(natural_image)
        compressed = ahash(_jpeg(natural_image, quality))
        dist = hamming_distance(orig, compressed)
        assert dist <= max_dist, (
            f"aHash JPEG q={quality}: distance {dist} > threshold {max_dist}"
        )

    @pytest.mark.parametrize("quality,max_dist", [(90, 10), (70, 12)])
    def test_dhash_jpeg(self, quality: int, max_dist: int,
                        natural_image: np.ndarray) -> None:
        orig = dhash(natural_image)
        compressed = dhash(_jpeg(natural_image, quality))
        dist = hamming_distance(orig, compressed)
        assert dist <= max_dist, (
            f"dHash JPEG q={quality}: distance {dist} > threshold {max_dist}"
        )


# ---------------------------------------------------------------------------
# 3. Resize invariance
# ---------------------------------------------------------------------------
# pHash resizes to 32×32 internally, so moderate external rescaling of the
# input should produce very similar low-frequency DCT content.

class TestResizeInvariance:
    """pHash stays close after downscaling the input image."""

    # (scale_factor, max_bits_changed)
    # Measured: 0.75→4, 0.5→10, 0.25→8 bits.  Buffer of +4 added.
    _CASES = [
        (0.75, 8),
        (0.50, 14),
        (0.25, 12),
    ]

    @pytest.mark.parametrize("scale,max_dist", _CASES)
    def test_phash_resize(self, scale: float, max_dist: int,
                          natural_image: np.ndarray) -> None:
        H, W = natural_image.shape[:2]
        new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))
        small = cv2.resize(natural_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        orig = phash(natural_image)
        resized = phash(small)
        dist = hamming_distance(orig, resized)
        assert dist <= max_dist, (
            f"pHash resize×{scale}: distance {dist} > threshold {max_dist}"
        )

    @pytest.mark.parametrize("scale,max_dist", [(0.75, 12), (0.50, 15)])
    def test_ahash_resize(self, scale: float, max_dist: int,
                          natural_image: np.ndarray) -> None:
        H, W = natural_image.shape[:2]
        small = cv2.resize(natural_image,
                           (max(1, int(W * scale)), max(1, int(H * scale))),
                           interpolation=cv2.INTER_AREA)
        dist = hamming_distance(ahash(natural_image), ahash(small))
        assert dist <= max_dist, f"aHash resize×{scale}: {dist} > {max_dist}"


# ---------------------------------------------------------------------------
# 4. Brightness / contrast invariance
# ---------------------------------------------------------------------------

class TestBrightnessInvariance:
    """Uniform brightness shifts produce small Hamming distances.

    dHash is explicitly designed for brightness invariance (it hashes
    differences, not absolute values).  pHash is moderately invariant
    because brightness primarily affects the DC coefficient which is
    excluded from the median calculation.  aHash is the least invariant.
    """

    @pytest.mark.parametrize("shift,max_dist", [(30, 5), (60, 8)])
    def test_dhash_brightness(self, shift: int, max_dist: int,
                              natural_image_gray: np.ndarray) -> None:
        """dHash should be nearly invariant to uniform brightness shifts."""
        brightened = np.clip(natural_image_gray.astype(np.int32) + shift, 0, 255).astype(np.uint8)
        dist = hamming_distance(dhash(natural_image_gray), dhash(brightened))
        assert dist <= max_dist, (
            f"dHash brightness +{shift}: distance {dist} > threshold {max_dist}"
        )

    @pytest.mark.parametrize("shift,max_dist", [(30, 10), (50, 12)])
    def test_phash_brightness(self, shift: int, max_dist: int,
                              natural_image: np.ndarray) -> None:
        """pHash DC-exclusion makes it moderately brightness-invariant.

        Measured on the block-grid fixture: +30→0 bits, +50→8 bits.
        Threshold set to measured+4 for stability.
        """
        gray = cv2.cvtColor(natural_image, cv2.COLOR_BGR2GRAY)
        brightened_gray = np.clip(gray.astype(np.int32) + shift, 0, 255).astype(np.uint8)
        brightened = cv2.cvtColor(brightened_gray, cv2.COLOR_GRAY2BGR)
        dist = hamming_distance(phash(natural_image), phash(brightened))
        assert dist <= max_dist, (
            f"pHash brightness +{shift}: distance {dist} > threshold {max_dist}"
        )


# ---------------------------------------------------------------------------
# 5. Hamming distance properties
# ---------------------------------------------------------------------------

class TestHammingDistance:
    """Mathematical properties of hamming_distance (scalar)."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        set_seed(SEED)

    def test_zero_self_distance(self) -> None:
        rng = np.random.default_rng(SEED)
        for h in rng.integers(0, 2**64, size=200, dtype=np.uint64):
            assert hamming_distance(int(h), int(h)) == 0

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(SEED)
        pairs = rng.integers(0, 2**64, size=(500, 2), dtype=np.uint64)
        for a, b in pairs:
            assert hamming_distance(int(a), int(b)) == hamming_distance(int(b), int(a))

    def test_range(self) -> None:
        rng = np.random.default_rng(SEED)
        pairs = rng.integers(0, 2**64, size=(500, 2), dtype=np.uint64)
        for a, b in pairs:
            d = hamming_distance(int(a), int(b))
            assert 0 <= d <= 64, f"Distance {d} out of [0, 64]"

    def test_max_distance(self) -> None:
        """0 XOR (2^64-1) has all 64 bits set."""
        assert hamming_distance(0, 0xFFFF_FFFF_FFFF_FFFF) == 64

    def test_triangle_inequality(self) -> None:
        """d(a, c) ≤ d(a, b) + d(b, c) for random triples."""
        rng = np.random.default_rng(SEED)
        triples = rng.integers(0, 2**64, size=(300, 3), dtype=np.uint64)
        for a, b, c in triples:
            dab = hamming_distance(int(a), int(b))
            dbc = hamming_distance(int(b), int(c))
            dac = hamming_distance(int(a), int(c))
            assert dab + dbc >= dac, (
                f"Triangle violated: d(a,b)={dab} + d(b,c)={dbc} < d(a,c)={dac}"
            )


# ---------------------------------------------------------------------------
# 6. Batch hamming_distances correctness
# ---------------------------------------------------------------------------

class TestHammingDistancesBatch:
    """hamming_distances must match the scalar hamming_distance element-wise."""

    def test_batch_matches_scalar_small(self) -> None:
        rng = np.random.default_rng(SEED)
        hashes = rng.integers(0, 2**64, size=50, dtype=np.uint64)
        q = int(rng.integers(0, 2**64, dtype=np.uint64))
        batch = hamming_distances(hashes, q)
        scalar = np.array([hamming_distance(int(h), q) for h in hashes], dtype=np.int32)
        assert np.array_equal(batch, scalar), (
            f"{(batch != scalar).sum()} / {len(hashes)} mismatches"
        )

    def test_batch_matches_scalar_large(self) -> None:
        rng = np.random.default_rng(SEED + 1)
        hashes = rng.integers(0, 2**64, size=10_000, dtype=np.uint64)
        q = int(rng.integers(0, 2**64, dtype=np.uint64))
        batch = hamming_distances(hashes, q)
        scalar = np.array([hamming_distance(int(h), q) for h in hashes], dtype=np.int32)
        assert np.array_equal(batch, scalar), (
            f"{(batch != scalar).sum()} / {len(hashes)} mismatches"
        )

    def test_batch_zero_for_self(self) -> None:
        """Every hash compared to itself must give distance 0."""
        rng = np.random.default_rng(SEED)
        hashes = rng.integers(0, 2**64, size=100, dtype=np.uint64)
        for h in hashes:
            result = hamming_distances(np.array([h], dtype=np.uint64), int(h))
            assert result[0] == 0

    def test_batch_output_dtype(self) -> None:
        rng = np.random.default_rng(SEED)
        hashes = rng.integers(0, 2**64, size=10, dtype=np.uint64)
        result = hamming_distances(hashes, 0)
        assert result.dtype == np.int32

    def test_batch_range(self) -> None:
        rng = np.random.default_rng(SEED)
        hashes = rng.integers(0, 2**64, size=1000, dtype=np.uint64)
        q = int(rng.integers(0, 2**64, dtype=np.uint64))
        result = hamming_distances(hashes, q)
        assert result.min() >= 0
        assert result.max() <= 64


# ---------------------------------------------------------------------------
# 7. Return-type and range checks for the hash functions
# ---------------------------------------------------------------------------

class TestHashOutputContract:
    """Hash functions must return plain Python ints in [0, 2**64 - 1]."""

    @pytest.mark.parametrize("hash_fn", [ahash, dhash, phash])
    def test_returns_int(self, hash_fn, natural_image: np.ndarray) -> None:
        h = hash_fn(natural_image)
        assert isinstance(h, int), f"{hash_fn.__name__} returned {type(h)}"

    @pytest.mark.parametrize("hash_fn", [ahash, dhash, phash])
    def test_range(self, hash_fn, natural_image: np.ndarray) -> None:
        h = hash_fn(natural_image)
        assert 0 <= h < 2**64, f"{hash_fn.__name__} returned out-of-range value {h}"

    @pytest.mark.parametrize("hash_fn", [ahash, dhash, phash])
    def test_deterministic(self, hash_fn, natural_image: np.ndarray) -> None:
        """Same input must always produce the same hash."""
        assert hash_fn(natural_image) == hash_fn(natural_image)
