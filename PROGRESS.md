# PixelTwin — Progress Log

Tracks milestone completion, measured numbers, and open decisions.
Updated at the end of every chunk.

---

## Milestone 1 — Foundation

### C0 — Repo scaffold ✅
**Committed:** 2026-10-26

Files added:
- `.gitignore` — Python, C++, OS, IDE, data exclusions
- `.gitattributes` — LF line endings enforced for all source files
- `pixeltwin/__init__.py` — package entry, `__version__ = "0.1.0"`
- `pixeltwin/seed.py` — `set_seed(seed=42)` seeds Python `random`, NumPy, and optionally PyTorch
- `pixeltwin/hashing.py` — stub (C1)
- `pixeltwin/index.py` — stub (C4/C5)
- `pixeltwin/verify.py` — stub (C10)
- `pixeltwin/embeddings.py` — stub (C12)
- `pixeltwin/transforms.py` — stub (C2)
- `pixeltwin/eval.py` — stub (C4)
- `pixeltwin/cli.py` — stub (C14)
- `tests/conftest.py` — shared fixtures (`_seed_each_test`, `tiny_gray_image`, `tiny_color_image`)
- `requirements.txt` — all deps pinned (see file)
- `PROGRESS.md` — this file

**Install check:** see `requirements.txt` header for verification command.

---

### C1 — Hashing ⬜ (next)
### C2 — Transforms ⬜
### C3a — Dataset prep ⬜
### C3b — Full dataset download ⬜
### C4 — Metrics + brute force + M1 benchmark ⬜

---

## Milestone 2 — LSH Index
### C5 ⬜  C6 ⬜

## Milestone 3 — C++ Core
### C7 ⬜  C8 ⬜  C9 ⬜

## Milestone 4 — Verification
### C10 ⬜  C11 ⬜

## Milestone 5 — Learned Baseline
### C12 ⬜  C13 ⬜

## Milestone 6 — Ship It
### C14 ⬜  C15 ⬜  C16 ⬜  C17 ⬜  C18 ⬜  C19 ⬜  C20 ⬜  C21 ⬜

---

## Open Decisions
| # | Decision | Status |
|---|----------|--------|
| 1 | Dataset (COCO / Open Images / Kaggle) | ⏳ Deferred to C3b |
| 2 | C++ toolchain upgrade (GCC 6 → 13 via MSYS2) | ⏳ Required before M3 |
