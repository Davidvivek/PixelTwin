"""
transforms.py — synthetic transformation generator.

Generates transformed copies of images for ground-truth construction:
  - JPEG recompression  (quality 90 / 70 / 50 / 30)
  - Resize              (0.75 / 0.5 / 0.25 of original area)
  - Center crop         (90 / 80 / 70 %)
  - Rotation            (5 / 15 / 45 / 90 degrees)
  - Brightness / contrast shifts
  - Gaussian noise
  - Combination transforms

All operations are seeded via pixeltwin.seed so results are reproducible.
"""

# Implementation filled in during C2.
raise NotImplementedError("transforms.py will be implemented in chunk C2")
