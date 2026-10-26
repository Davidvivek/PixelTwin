"""
PixelTwin — near-duplicate image search engine.

Package layout
--------------
pixeltwin/
    hashing.py      aHash, dHash, pHash (64-bit perceptual hashes)
    index.py        LSH index + brute-force Hamming scan
    verify.py       ORB + RANSAC geometric verification
    embeddings.py   ResNet-50 embedding extractor (learned baseline)
    transforms.py   Synthetic transformation generator for evaluation
    eval.py         Precision, recall, F1 per transform type
    seed.py         Global random seed utility (reproducibility)
    cli.py          CLI entry points: `pixeltwin index` / `pixeltwin query`
"""

__version__ = "0.1.0"
__author__ = "David Vivek"
