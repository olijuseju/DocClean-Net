# DocClean-Net — Current Status

> **INSTRUCTIONS FOR MAINTENANCE**
> Update this file at the end of each work session.
> Replace [DATE] with the actual date when updating.
> This file is the source of truth for Claude about what exists, what works, and what doesn't.

Last updated: 08/07/2026

---

## Overall progress

```
Phase 0 — Repo & environment     [x] COMPLETE
Phase 1 — Synthetic generator    [x] COMPLETE (50 tests)
Phase 2 — U-Net & training       [x] COMPLETE (59 tests; best val_loss 0.000805)
Phase 3 — Inference & evaluation [x] COMPLETE (137 tests total; benchmark n=11)
Phase 4 — Integration & README   [ ] Not started
```

---

## What is fully done ✓

### Classic pipeline (`classic_pipeline/digitize_notebook.py`)
- Complete 6-step pipeline: synthetic channel → adaptive threshold → residual grid
  inpainting → re-extraction → color ink detection → noise removal by CC analysis
- CLI with full argument support: `--alpha`, `--block`, `--c-offset`, `--grid-kernel`,
  `--inpaint-r`, `--noise-area-small`, `--noise-area-medium`, `--noise-radius`,
  `--noise-density`, `--skip-inpaint`, `--skip-color`, `--skip-denoise`, `--debug`
- Known working parameters for scanned blue-grid notebooks: alpha=4, block=25, c_offset=5

### GUI (`gui/digitize_gui.py`)
- Full tkinter + ttkbootstrap (theme: darkly) application, v4
- Three view modes, paint layer corrections, undo stack, pipeline in separate thread

### Repo infrastructure (Phase 0)
- GitHub repo `olijuseju/DocClean-Net`, folder structure per repo_structure.md
- pyproject.toml, requirements.txt, requirements-dev.txt, CI (pytest+ruff+black),
  .gitignore, tests/conftest.py with shared fixtures

### Synthetic data generation (Phase 1)
- `data/generators/`: paper.py, strokes.py, degradations.py (blue grid, ruled
  lines, watermark); `data/augmentation.py` (synchronized pair augmentation);
  `data/generate_dataset.py` CLI
- Run as `python -m data.generate_dataset` on Windows (module invocation required)

### U-Net & training (Phase 2)
- `model/unet.py` — UNet(1→16→32→64, bottleneck 128), **482,449 parameters**
  (docstring said ~660k until Phase 3; corrected)
- `model/losses.py` — CombinedLoss = MSE*0.7 + (1-SSIM)*0.3
- `model/dataset.py` — DocCleanDataset (256×256 patches, synced augmentation)
- `model/train.py` — AdamW + cosine annealing; saves raw state_dict to best.pt
- Trained on Colab T4: 50 epochs, 10,000 synthetic pairs 512×512, batch 16,
  ~3.86 h total. Best val_loss 0.000805 at epoch 37. `train_log.csv` committed;
  `best.pt` gitignored (distribute via GitHub Releases)

### Inference & evaluation (Phase 3)
- `inference/io_utils.py` — unicode-safe `_imread`/`_imwrite`
  (np.fromfile/cv2.imdecode + cv2.imencode/buf.tofile; handles `Escáner_*.png`)
- `inference/predict.py` — `predict_image()`: sliding window 256×256 stride 128
  (50% overlap), Gaussian blending (peak-normalized window, sigma=patch/4),
  BORDER_REFLECT_101 padding, batched forward passes, arbitrary input resolution.
  Post-processing white-point normalization (default `white_point="auto"`:
  histogram mode − 10, linear stretch to 255; `None` disables, int fixes it).
  CLI: `python -m inference.predict --model checkpoints/best.pt -–input ... --output ...`
  with `--white-point auto|<int>|off`
- `inference/benchmark.py` — DL vs classic on `data/real_test/`; metrics per
  image: SSIM, PSNR (classic output as reference — no ground truth for real
  scans), BRISQUE via `piq` for both methods, ink_coverage_pct (threshold 128),
  inference_time_ms. Outputs `metrics.csv` + `benchmark_metrics.png` +
  `benchmark_summary.png`. Handles empty test dir (header-only CSV).
- `scripts/visualize_results.py` — original | classic | DL side-by-side figure,
  optional `--crop X Y W H` zoom row. Runnable as plain script from any cwd.
- `tests/test_inference.py` — full suite is now **137 fast tests + 3 slow**,
  all passing on Windows (Python 3.11.9 and 3.13.14) and Linux CI

---

## What is in progress ⟳

Nothing currently in progress.

---

## What is not started yet ✗

### Phase 4 — Integration & README
- README.md finalization: architecture overview, quantitative results table,
  before/after demo images (use `scripts/visualize_results.py` output)
- `scripts/download_model.py` — blocked until `best.pt` is uploaded to a
  GitHub Release (needs the real URL to implement and test honestly)
- Upload `best.pt` to GitHub Releases
- "Future work" section: Quick Draw! API stroke enhancement idea
- Rename `test_unet_parameter_count_approximately_660k` → `_482k`
- Optional cleanup: centralize the three private `_imread` copies
  (model/dataset.py, classic_pipeline, inference/io_utils.py) into one module

---

## Known issues and technical decisions made

### Classic pipeline
- **Residual grid detection uses kernel size 120px minimum.** Smaller values
  cause false positives (diagonal strokes classified as grid).
- **Noise removal radius=25px works for dense comic-style drawings.** For sparse
  stippling-style content, increase to 40-50px or disable denoising.
- **Red ink detection threshold: H=[155-180], S>40, V=[40-230], R-G>15, R-B>15.**
- **GUI tooltip bug fixed in v4** (after/after_cancel pattern).

### Architecture decisions (don't revisit without reason)
- U-Net input: raw grayscale (not the synthetic channel).
- Loss: MSE*0.7 + (1-SSIM)*0.3. No perceptual/VGG loss.
- Inference: sliding window with stride=128 (50% overlap). Tested sweet spot.

### Phase 3 findings
- **MSE-trained sigmoid never saturates**: paper comes out at ~233, residual
  grid survives 10–20 levels below → visible faint grid. Root cause is twofold:
  MSE averages plausible outputs (literature: L1 preferred in restoration),
  AND synthetic clean targets have paper at mean 245, never 255. Fixed with
  deterministic white-point post-processing — retraining with L1 was evaluated
  and rejected (cost >> benefit; wouldn't fix the second cause).
- **BRISQUE is NOT in scikit-image** (spec error). Implemented via `piq`
  (pure PyTorch, no libsvm/opencv-contrib). `piq.brisque` raises
  AssertionError on near-constant images (e.g. binary classic output of
  stroke-free paper) → returns NaN, excluded from summary means.
- **BRISQUE values are relative-comparison only**: it's trained on natural
  image statistics; binarized documents score high in absolute terms.
- **Ink coverage gap (8.1% classic vs 4.1% DL) is stroke thickness, not
  content loss**: DL/classic ratio is 0.44–0.55 across all 11 test images
  (systematic bias signature). Classic binarization captures the full
  antialiasing profile; DL restores thinner strokes.
  Verified visually with `scripts/visualize_results.py --crop`. <!-- confirm -->

---

## Benchmark results (real images)

Benchmark run 08/07/2026 on **11 real scans** (~1700×2338 px, blue grid,
pen drawings), never used for training or classic-pipeline calibration.
Hardware: CPU, [FILL IN: CPU model of the benchmark PC].

| Metric | Classic pipeline | DocClean-Net | Notes |
|--------|-----------------|--------------|-------|
| SSIM | (reference) | 0.80 ± 0.02 | DL vs classic reference, n=11 |
| PSNR (dB) | (reference) | 13.6 ± 0.6 | |
| BRISQUE | 151.9 | 87.0 | Lower = better; relative comparison only |
| Ink coverage % | 8.1 | 4.1 | Classic binarization yields thicker strokes |
| Time (ms/page) | 7154 | 4552 | DL ~1.6× faster on CPU; DL time very stable (±0.2 s) |

Raw data: `benchmark_results/metrics.csv` (gitignored; regenerate with
`python -m inference.benchmark --model checkpoints/best.pt --test-dir data/real_test/`).

---

## Test images available

11 real scans in `data/real_test/` (gitignored, on the secondary PC —
sync manually, git does not carry them):
`21.png` … `24.png`, `Escáner_20230219 (25).png` … `(31).png`.

---

## Next immediate action

**Start Phase 4: README finalization.**

First task: generate 2–3 before/after demo figures with
`scripts/visualize_results.py`, upload `best.pt` to a GitHub Release, then
write the README (architecture, results table above, usage, future work).

---

## Template for conversation context block

Copy this at the start of any new conversation in this project,
filling in the last two sections:

```
CURRENT STATUS (from current_status.md):
- Classic pipeline: COMPLETE (digitize_notebook.py v3)
- GUI: COMPLETE (digitize_gui.py v4, ttkbootstrap darkly)
- Repo infrastructure: COMPLETE
- Synthetic generator: COMPLETE
- U-Net model: COMPLETE (482,449 params)
- Training: COMPLETE (best val_loss 0.000805, epoch 37/50, Colab T4)
- Inference: COMPLETE (sliding window + gaussian blending + white point)
- Benchmark: COMPLETE (n=11, see table in current_status.md)

MODULE I AM WORKING ON NOW: [module name and file path]

RELEVANT CODE (what I have so far in this module):
[paste the actual code]

PROBLEM / WHAT I NEED:
[describe the specific problem or what to build next]
```
