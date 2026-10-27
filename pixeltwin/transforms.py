"""
transforms.py — synthetic transformation generator for PixelTwin evaluation.

Purpose
-------
Generates transformed copies of images to build ground-truth data for the
near-duplicate search evaluation pipeline (C3a onward).  Every transform is
deterministic given its parameters; stochastic transforms (Gaussian noise,
combos that include noise) additionally accept a ``seed`` argument.

Transforms implemented
----------------------
jpeg        JPEG encode + decode at a given quality level.
resize      Downscale by a fractional scale factor.
crop        Centre-crop to a fraction of the original dimensions.
rotate      Rotate by a fixed angle (borders filled by reflection).
brightness  Add a constant shift to every pixel (signed, clipped to uint8).
contrast    Scale pixel deviations from mid-grey by a factor (clipped).
noise       Add zero-mean Gaussian noise with a given sigma.
combo       Apply an ordered sequence of the above transforms.

``TransformSpec`` dataclass
---------------------------
A lightweight, JSON-serialisable record that names a transform and stores its
parameters.  Used as the key in ground-truth manifests so every evaluated
pair can be traced back to the exact transform that produced it.

``ALL_SPECS`` registry
----------------------
The canonical list of all (transform, parameter) combinations used in the
benchmark.  Import and iterate this list in C3a / C4 to build the dataset.

``apply_transform(image, spec, seed)``
--------------------------------------
Single dispatch point: maps a TransformSpec to the correct primitive.

``generate_all_transforms(image, seed)``
-----------------------------------------
Convenience wrapper: applies every spec in ALL_SPECS and returns a list of
(TransformSpec, transformed_image) pairs.

All functions
-------------
- Accept uint8 NumPy arrays of shape (H, W) or (H, W, 3) [BGR convention].
- Return uint8 NumPy arrays; the channel count is always preserved.
- Never modify the input in-place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from pixeltwin.seed import SEED

__all__ = [
    "TransformSpec",
    "ALL_SPECS",
    "apply_transform",
    "generate_all_transforms",
    # Individual primitives (also exported for direct use in tests/benchmarks)
    "jpeg_compress",
    "resize_image",
    "center_crop",
    "rotate_image",
    "adjust_brightness",
    "adjust_contrast",
    "add_gaussian_noise",
    "apply_combo",
]


# ---------------------------------------------------------------------------
# TransformSpec — lightweight descriptor
# ---------------------------------------------------------------------------

@dataclass
class TransformSpec:
    """Describes a single image transform and its parameters.

    Attributes
    ----------
    name:
        Transform identifier string — one of ``{"jpeg", "resize", "crop",
        "rotate", "brightness", "contrast", "noise", "combo"}``.
    params:
        Dictionary of keyword arguments forwarded to the transform primitive.
        Must be JSON-serialisable (str/int/float/list/dict only) so that
        ground-truth manifests can be persisted to disk.

    Examples
    --------
    >>> TransformSpec("jpeg", {"quality": 90})
    TransformSpec(name='jpeg', params={'quality': 90})
    >>> str(TransformSpec("resize", {"scale": 0.75}))
    'resize(scale=0.75)'
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable one-liner, e.g. ``jpeg(quality=90)``."""
        if not self.params:
            return self.name
        kv = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({kv})"

    def slug(self) -> str:
        """Filesystem-safe identifier, e.g. ``jpeg_quality90``.

        Removes parentheses, replaces dots with ``p``, commas with ``_``.
        Useful for constructing output filenames in the dataset prep script.
        """
        s = str(self)
        for old, new in [("(", "_"), (")", ""), ("=", ""), (",", "_"), (".", "p"), ("-", "neg")]:
            s = s.replace(old, new)
        return s


# ---------------------------------------------------------------------------
# Primitive transforms
# ---------------------------------------------------------------------------

def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    """JPEG-encode then decode *image* at the given quality level.

    Simulates the lossy compression that happens when an image is saved to
    JPEG and re-opened — a very common real-world near-duplicate scenario.
    The operation is fully deterministic (no RNG involved).

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3) [BGR].
    quality:
        JPEG quality in [1, 100].  Typical benchmark values: 90, 70, 50, 30.

    Returns
    -------
    np.ndarray
        uint8 array of the same shape as *image*.

    Raises
    ------
    RuntimeError
        If OpenCV's ``imencode`` or ``imdecode`` fails.
    """
    flag = cv2.IMWRITE_JPEG_QUALITY
    ok, buf = cv2.imencode(".jpg", image, [flag, int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed (quality={quality})")
    decode_flag = cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR
    decoded = cv2.imdecode(buf, decode_flag)
    if decoded is None:
        raise RuntimeError("cv2.imdecode returned None")
    return decoded


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Downscale *image* by *scale* using area interpolation.

    Area interpolation (``cv2.INTER_AREA``) averages all contributing source
    pixels before downsampling — equivalent to a box filter, which minimises
    aliasing.  This is the correct choice for evaluation because it matches
    how most image viewers and editors resize photos.

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    scale:
        Fractional scale in (0, 1].  E.g. 0.75 shrinks to 75% of each side.

    Returns
    -------
    np.ndarray
        uint8 array, shape (max(1, round(H*scale)), max(1, round(W*scale)))
        with the same number of channels as *image*.
    """
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    H, W = image.shape[:2]
    new_h = max(1, round(H * scale))
    new_w = max(1, round(W * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop(image: np.ndarray, crop_frac: float) -> np.ndarray:
    """Crop the central ``crop_frac`` portion of *image*.

    No resize-back step: the returned image is smaller than the input.  The
    hash functions (pHash etc.) resize internally, so the crop is faithfully
    evaluated without introducing a second resizing artefact.

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    crop_frac:
        Fraction of each dimension to retain, in (0, 1].
        E.g. 0.9 removes 5% from each edge.

    Returns
    -------
    np.ndarray
        uint8 array, shape (max(1, int(H*crop_frac)),
        max(1, int(W*crop_frac))) with the same channel count.
    """
    if not (0.0 < crop_frac <= 1.0):
        raise ValueError(f"crop_frac must be in (0, 1], got {crop_frac}")
    H, W = image.shape[:2]
    ch = max(1, int(H * crop_frac))
    cw = max(1, int(W * crop_frac))
    y0 = (H - ch) // 2
    x0 = (W - cw) // 2
    return image[y0 : y0 + ch, x0 : x0 + cw].copy()


def rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate *image* counter-clockwise by *degrees* around its centre.

    The output has the same spatial dimensions as the input.  Corners that
    fall outside the image after rotation are filled using ``BORDER_REFLECT_101``
    (mirror-reflect without repeating the edge pixel), which avoids the solid
    black borders that would otherwise corrupt a perceptual hash.

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    degrees:
        Rotation angle in degrees.  Positive = counter-clockwise.
        Benchmark values: 5, 15, 45, 90.

    Returns
    -------
    np.ndarray
        uint8 array, same shape as *image*.
    """
    H, W = image.shape[:2]
    centre = (W / 2.0, H / 2.0)
    # scale=1.0 keeps the image size unchanged.
    M = cv2.getRotationMatrix2D(centre, float(degrees), scale=1.0)
    return cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def adjust_brightness(image: np.ndarray, shift: int) -> np.ndarray:
    """Add *shift* to every pixel value, clipping to [0, 255].

    A positive shift brightens; a negative shift darkens.  This is a uniform
    additive offset — the simplest brightness change model.

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    shift:
        Integer offset.  Benchmark values: ±30, ±60.

    Returns
    -------
    np.ndarray
        uint8 array, same shape as *image*.
    """
    # Cast to int32 to safely handle overflow before clipping.
    shifted = image.astype(np.int32) + int(shift)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel deviations from mid-grey (128) by *factor*.

    Pixels brighter than 128 become brighter (factor > 1) or closer to 128
    (factor < 1); pixels darker than 128 move symmetrically.  This preserves
    the average brightness of the image while changing its dynamic range.

    Formula:  output = clip(128 + (input − 128) × factor, 0, 255)

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    factor:
        Contrast multiplier.  > 1 increases contrast; < 1 decreases.
        Benchmark values: 1.3, 0.7.

    Returns
    -------
    np.ndarray
        uint8 array, same shape as *image*.
    """
    img_f = image.astype(np.float32)
    result = 128.0 + (img_f - 128.0) * float(factor)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int = SEED,
) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation *sigma*.

    Noise is generated from a seeded RNG (``numpy.random.default_rng``) so
    results are fully reproducible given the same *seed*.

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    sigma:
        Standard deviation of the noise in pixel units.  Benchmark: 10, 20.
    seed:
        RNG seed for reproducibility.  Default: project-wide ``SEED``.

    Returns
    -------
    np.ndarray
        uint8 array, same shape as *image*.
    """
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=float(sigma), size=image.shape)
    noisy = image.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_combo(
    image: np.ndarray,
    steps: list[tuple[str, dict[str, Any]]],
    seed: int = SEED,
) -> np.ndarray:
    """Apply an ordered sequence of transforms to *image*.

    Each step is a ``(name, params)`` pair, identical to the ``name`` and
    ``params`` fields of a ``TransformSpec``.  Stochastic transforms within
    the combo receive ``seed + step_index`` so different steps do not share
    the same RNG state.

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    steps:
        Ordered list of ``(transform_name, params_dict)`` pairs.
    seed:
        Base seed; step *i* uses ``seed + i``.

    Returns
    -------
    np.ndarray
        uint8 array (shape may differ if resize or crop is included).
    """
    result = image.copy()
    for i, (name, params) in enumerate(steps):
        spec = TransformSpec(name=name, params=params)
        result = apply_transform(result, spec, seed=seed + i)
    return result


# ---------------------------------------------------------------------------
# Central dispatch
# ---------------------------------------------------------------------------

def apply_transform(
    image: np.ndarray,
    spec: TransformSpec,
    seed: int = SEED,
) -> np.ndarray:
    """Apply the transform described by *spec* to *image*.

    This is the single entry point used by the evaluation harness (C4) and
    dataset prep script (C3a).  All callers should use this function rather
    than calling the primitives directly, so that the dispatch logic stays
    in one place.

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    spec:
        A ``TransformSpec`` describing which transform to apply.
    seed:
        Seed forwarded to stochastic transforms (``noise``, ``combo``).

    Returns
    -------
    np.ndarray
        Transformed uint8 array.

    Raises
    ------
    ValueError
        If ``spec.name`` is not a recognised transform.
    """
    p = spec.params
    match spec.name:
        case "jpeg":
            return jpeg_compress(image, quality=p["quality"])
        case "resize":
            return resize_image(image, scale=p["scale"])
        case "crop":
            return center_crop(image, crop_frac=p["frac"])
        case "rotate":
            return rotate_image(image, degrees=p["degrees"])
        case "brightness":
            return adjust_brightness(image, shift=p["shift"])
        case "contrast":
            return adjust_contrast(image, factor=p["factor"])
        case "noise":
            return add_gaussian_noise(image, sigma=p["sigma"], seed=seed)
        case "combo":
            return apply_combo(image, steps=p["steps"], seed=seed)
        case _:
            raise ValueError(
                f"Unknown transform {spec.name!r}.  "
                f"Valid names: jpeg, resize, crop, rotate, brightness, contrast, noise, combo."
            )


# ---------------------------------------------------------------------------
# ALL_SPECS — canonical benchmark registry
# ---------------------------------------------------------------------------
# This list is the single source of truth for which transforms are evaluated.
# C3a iterates it to generate the ground-truth dataset; C4 / C15 reference
# it for ablation grouping.  Keep in sync with the project brief.

ALL_SPECS: list[TransformSpec] = [
    # ── JPEG recompression ──────────────────────────────────────────────────
    TransformSpec("jpeg", {"quality": 90}),
    TransformSpec("jpeg", {"quality": 70}),
    TransformSpec("jpeg", {"quality": 50}),
    TransformSpec("jpeg", {"quality": 30}),

    # ── Resize (downscale) ──────────────────────────────────────────────────
    TransformSpec("resize", {"scale": 0.75}),
    TransformSpec("resize", {"scale": 0.50}),
    TransformSpec("resize", {"scale": 0.25}),

    # ── Centre crop ─────────────────────────────────────────────────────────
    TransformSpec("crop", {"frac": 0.90}),
    TransformSpec("crop", {"frac": 0.80}),
    TransformSpec("crop", {"frac": 0.70}),

    # ── Rotation ────────────────────────────────────────────────────────────
    TransformSpec("rotate", {"degrees": 5}),
    TransformSpec("rotate", {"degrees": 15}),
    TransformSpec("rotate", {"degrees": 45}),
    TransformSpec("rotate", {"degrees": 90}),

    # ── Brightness shift ────────────────────────────────────────────────────
    TransformSpec("brightness", {"shift": 30}),
    TransformSpec("brightness", {"shift": -30}),
    TransformSpec("brightness", {"shift": 60}),
    TransformSpec("brightness", {"shift": -60}),

    # ── Contrast ────────────────────────────────────────────────────────────
    TransformSpec("contrast", {"factor": 1.3}),
    TransformSpec("contrast", {"factor": 0.7}),

    # ── Gaussian noise ──────────────────────────────────────────────────────
    TransformSpec("noise", {"sigma": 10}),
    TransformSpec("noise", {"sigma": 20}),

    # ── Combination transforms ───────────────────────────────────────────────
    # Combo 1: mild JPEG + moderate resize (common real-world pipeline)
    TransformSpec("combo", {"steps": [
        ("jpeg",   {"quality": 90}),
        ("resize", {"scale": 0.75}),
    ]}),
    # Combo 2: aggressive JPEG + slight brightening
    TransformSpec("combo", {"steps": [
        ("jpeg",       {"quality": 70}),
        ("brightness", {"shift": 30}),
    ]}),
    # Combo 3: resize + noise (simulates low-resolution scan)
    TransformSpec("combo", {"steps": [
        ("resize", {"scale": 0.75}),
        ("noise",  {"sigma": 10}),
    ]}),
    # Combo 4: slight rotation + JPEG
    TransformSpec("combo", {"steps": [
        ("rotate", {"degrees": 5}),
        ("jpeg",   {"quality": 90}),
    ]}),
]


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_all_transforms(
    image: np.ndarray,
    seed: int = SEED,
) -> list[tuple[TransformSpec, np.ndarray]]:
    """Apply every spec in ``ALL_SPECS`` to *image* and return the results.

    Parameters
    ----------
    image:
        uint8 array, shape (H, W) or (H, W, 3).
    seed:
        Seed forwarded to stochastic transforms.

    Returns
    -------
    list of (TransformSpec, np.ndarray)
        One pair per entry in ``ALL_SPECS``, in the same order.
        Each array is uint8 and independent (no shared memory with *image*).
    """
    return [
        (spec, apply_transform(image, spec, seed=seed))
        for spec in ALL_SPECS
    ]
