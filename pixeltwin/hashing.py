"""
hashing.py — perceptual hash functions.

Implements three 64-bit image fingerprints:
  - aHash  (average hash)
  - dHash  (difference hash)
  - pHash  (perceptual / DCT hash)  ← primary fingerprint for PixelTwin

All functions accept a uint8 NumPy array (H×W or H×W×C) and return a
64-bit integer hash.  A pure-NumPy ``hamming_distance`` is provided as
the fallback when the C++ extension is not available.

See Also
--------
cpp/hamming.cpp : C++17 accelerated Hamming distance (M3).
"""

# Implementation filled in during C1.
raise NotImplementedError("hashing.py will be implemented in chunk C1")
