"""
hashing.py — perceptual hash functions for near-duplicate image detection.

Three 64-bit image fingerprints
--------------------------------
aHash  (average hash)
    Threshold each pixel of an 8×8 thumbnail against the global mean.
    Fastest; poor under geometric transforms beyond small scaling.

dHash  (difference hash)
    Compare adjacent pixel pairs in a 9×8 thumbnail row-by-row.
    Fast; naturally invariant to uniform brightness shifts.

pHash  (perceptual / DCT hash)  ← primary fingerprint for PixelTwin
    2-D DCT of a 32×32 thumbnail; threshold the 8×8 low-frequency block
    against the median of the AC coefficients (DC excluded).
    Most robust: survives JPEG recompression, resizing, mild brightness
    and contrast changes.

All functions accept a uint8 NumPy array (H×W grayscale or H×W×C BGR,
matching OpenCV convention) and return a Python ``int`` in [0, 2**64 − 1].

Hamming utilities
-----------------
hamming_distance(a, b)       Scalar pair comparison.
hamming_distances(arr, q)    Vectorised batch comparison (NumPy fallback).

The C++17 accelerated variants (M3) expose the same names when the
extension module is compiled; this file is always the pure-Python fallback.

References
----------
- Zauner, C. (2010). Implementation and Benchmarking of Perceptual Image
  Hash Functions. FH Hagenberg.
- http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.fft import dctn

__all__ = [
    "ahash",
    "dhash",
    "phash",
    "hamming_distance",
    "hamming_distances",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale (H×W) or BGR (H×W×3) uint8 array to grayscale.

    Parameters
    ----------
    image:
        Input image array.

    Returns
    -------
    np.ndarray
        2-D uint8 grayscale array, shape (H, W).

    Raises
    ------
    ValueError
        If the array has an unexpected number of dimensions.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(
        f"Expected a 2-D (H×W) or 3-D (H×W×C) image array, got shape {image.shape}"
    )


def _bits_to_int(bits: np.ndarray) -> int:
    """Pack a 64-element boolean/uint8 array into a 64-bit Python int.

    Packing is row-major, most-significant-bit first, so the first element
    of *bits* becomes the highest-order bit of the returned integer.

    Parameters
    ----------
    bits:
        1-D array of length 64 with values 0 or 1 (or bool).

    Returns
    -------
    int
        Value in [0, 2**64 − 1].
    """
    packed = np.packbits(bits.astype(np.uint8))   # 8 bytes
    return int.from_bytes(packed.tobytes(), byteorder="big")


# Byte-level Hamming-weight lookup table — built once at import time.
# _POPCOUNT_LUT[b] = number of 1-bits in byte b (for b in 0..255).
_POPCOUNT_LUT: np.ndarray = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.int32
)


# ---------------------------------------------------------------------------
# aHash — Average Hash
# ---------------------------------------------------------------------------

def ahash(image: np.ndarray) -> int:
    """Compute the 64-bit Average Hash (aHash) of *image*.

    Algorithm
    ---------
    1. Convert to grayscale.
    2. Resize to 8×8 using area interpolation (anti-aliased downscale).
    3. Compute the mean pixel value of the 64-pixel thumbnail.
    4. Threshold: bit[i] = 1 if pixel[i] > mean, else 0.
    5. Pack 64 bits to a Python int (MSB first).

    Properties
    ----------
    - Time complexity: O(H·W) for the resize, then O(1).
    - Invariant to: mild JPEG recompression, uniform scaling.
    - Sensitive to: gamma, contrast changes, rotation.

    Parameters
    ----------
    image:
        uint8 NumPy array, shape (H, W) or (H, W, 3) [BGR].

    Returns
    -------
    int
        64-bit hash in [0, 2**64 − 1].
    """
    gray = _to_gray(image)
    # Area interpolation averages all contributing source pixels — best
    # choice for downscaling because it acts as a low-pass filter first.
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean_val = float(small.mean())
    bits = (small.flatten() > mean_val).astype(np.uint8)
    return _bits_to_int(bits)


# ---------------------------------------------------------------------------
# dHash — Difference Hash
# ---------------------------------------------------------------------------

def dhash(image: np.ndarray) -> int:
    """Compute the 64-bit Difference Hash (dHash) of *image*.

    Algorithm
    ---------
    1. Convert to grayscale.
    2. Resize to 9 columns × 8 rows (area interpolation).
    3. For each of the 8 rows, compare each pixel to its right neighbour:
       bit = 1 if left_pixel > right_pixel (falling gradient), else 0.
       This yields 8 bits × 8 rows = 64 bits.
    4. Flatten row-major and pack to int (MSB first).

    Why 9×8?  Each row of 9 pixels yields 8 *differences*; 8 rows × 8
    differences = 64 bits exactly.

    Properties
    ----------
    - Time complexity: O(H·W) for the resize, then O(1).
    - Invariant to: uniform brightness shifts (subtracting a constant from
      all pixels does not change the sign of any difference).
    - Sensitive to: rotation, large crops.

    Parameters
    ----------
    image:
        uint8 NumPy array, shape (H, W) or (H, W, 3) [BGR].

    Returns
    -------
    int
        64-bit hash in [0, 2**64 − 1].
    """
    gray = _to_gray(image)
    # 9 columns so we get exactly 8 left-right comparisons per row.
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    # Slice off last column from the left side, first column from the right.
    bits = (small[:, :-1] > small[:, 1:]).astype(np.uint8)  # shape (8, 8)
    return _bits_to_int(bits.flatten())


# ---------------------------------------------------------------------------
# pHash — Perceptual (DCT) Hash
# ---------------------------------------------------------------------------

def phash(
    image: np.ndarray,
    hash_size: int = 8,
    highfreq_factor: int = 4,
) -> int:
    """Compute the 64-bit Perceptual Hash (pHash) of *image*.

    Algorithm (as specified in the PixelTwin project brief)
    -------------------------------------------------------
    1. Convert to grayscale.
    2. Resize to ``img_size × img_size`` where ``img_size = hash_size *
       highfreq_factor`` (default 32×32) using area interpolation.
    3. Cast to float32 and apply the 2-D DCT-II with orthonormal
       normalisation (``scipy.fft.dctn(..., norm='ortho')``).
    4. Retain the top-left ``hash_size × hash_size`` (8×8) block — the
       *low-frequency* portion of the spectrum.
    5. Flatten to 64 coefficients.
    6. Compute the median of coefficients at indices 1..63, **excluding
       the DC component at index 0** (position [0, 0] in the block).
       The DC coefficient encodes mean brightness and has a much larger
       magnitude than the AC terms; including it would bias the threshold.
    7. Threshold: bit[i] = 1 if coeff[i] > median, else 0.
    8. Pack 64 bits to a Python int (MSB first).

    Why DCT?
        JPEG compresses images by discarding high-frequency DCT coefficients.
        Consequently, the *low-frequency* coefficients are almost identical
        across different JPEG quality levels — exactly what pHash exploits.
        Resizing a natural image does not dramatically alter its low-frequency
        content either, making pHash robust to both transforms.

    Properties
    ----------
    - Time complexity: O((hash_size · highfreq_factor)²) for DCT.
    - Invariant to: JPEG recompression (quality ≥ 30), moderate resize
      (down to ~50%), mild brightness/contrast shifts.
    - Sensitive to: rotation beyond ~5°, aggressive cropping (> 30%).

    Parameters
    ----------
    image:
        uint8 NumPy array, shape (H, W) or (H, W, 3) [BGR].
    hash_size:
        Side length of the low-frequency DCT block to retain. The hash
        will contain ``hash_size ** 2`` bits. Default: 8 (64 bits).
    highfreq_factor:
        Upscale factor relative to *hash_size* for the intermediate
        resize. Larger values include more high-frequency information
        before the DCT but increase compute. Default: 4 (→ 32×32).

    Returns
    -------
    int
        64-bit hash in [0, 2**64 − 1] (for default parameters).
    """
    img_size = hash_size * highfreq_factor   # 32 by default

    gray = _to_gray(image)
    small = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # 2-D DCT-II, orthonormal normalisation.
    # dctn returns float64; we compute on float32 for speed (precision is
    # more than sufficient for a 1-bit-per-coefficient threshold).
    dct_full: np.ndarray = dctn(small.astype(np.float32), norm="ortho")

    # Top-left block holds the low-spatial-frequency content.
    dct_low = dct_full[:hash_size, :hash_size]   # shape (8, 8)
    flat = dct_low.flatten()                      # 64 coefficients

    # Median of AC terms (indices 1..63); DC (index 0) excluded.
    median = float(np.median(flat[1:]))

    bits = (flat > median).astype(np.uint8)
    return _bits_to_int(bits)


# ---------------------------------------------------------------------------
# Hamming distance utilities  (NumPy / pure-Python fallback)
# ---------------------------------------------------------------------------

def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Return the Hamming distance between two 64-bit integer hashes.

    The Hamming distance counts the number of bit positions at which the
    two hashes differ.  Range: [0, 64].

    Typical interpretation
    ----------------------
    - 0       : identical images (or near-identical thumbnails).
    - ≤ 10    : likely near-duplicate (default match threshold).
    - > 10    : probably different images.

    Implementation: XOR the two integers, then count set bits with
    Python's built-in ``bin().count('1')``.  Fast for scalar use.
    For batches of N comparisons use ``hamming_distances`` instead.

    Parameters
    ----------
    hash_a, hash_b:
        64-bit integer hashes as returned by ``ahash``, ``dhash``, or
        ``phash``.

    Returns
    -------
    int
        Number of differing bits, in [0, 64].
    """
    return bin(hash_a ^ hash_b).count("1")


def hamming_distances(hashes: np.ndarray, query_hash: int) -> np.ndarray:
    """Compute Hamming distances from *query_hash* to every hash in *hashes*.

    This is the vectorised NumPy fallback used until the C++ extension
    (M3) is compiled.  It uses a 256-entry byte-level popcount lookup
    table so each 64-bit XOR result is split into 8 bytes, looked up, and
    summed — avoiding the uint64 multiplication overflow edge-cases of the
    SWAR approach on some NumPy/platform combinations.

    Algorithm
    ---------
    For each hash h in *hashes*:
        xor  = h XOR query_hash          (uint64)
        dist = popcount(xor)             (byte-LUT lookup, summed over 8 bytes)

    Parameters
    ----------
    hashes:
        1-D array of dtype ``uint64``, shape (N,).  Build it with
        ``np.array(list_of_ints, dtype=np.uint64)``.
    query_hash:
        64-bit integer to compare against (Python int or np.uint64).

    Returns
    -------
    np.ndarray
        1-D int32 array of Hamming distances, shape (N,).
    """
    q = np.uint64(int(query_hash) & 0xFFFF_FFFF_FFFF_FFFF)
    xor = hashes.astype(np.uint64) ^ q    # shape (N,), dtype uint64

    # View each uint64 as 8 consecutive bytes for LUT lookup.
    byte_view = xor.view(np.uint8)                    # shape (8*N,)
    counts = _POPCOUNT_LUT[byte_view]                 # shape (8*N,)
    return counts.reshape(-1, 8).sum(axis=1).astype(np.int32)  # shape (N,)
