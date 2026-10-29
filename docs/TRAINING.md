# Training Guide

1. Organize datasets under `datasets/` with `real/` and `fake/` splits.
2. Generate a manifest:

```bash
python scripts/prepare_dataset.py
```

3. Train the model:

```bash
python scripts/train_model.py
```

The training pipeline includes:

- EfficientNetB0 transfer learning backbone
- Bidirectional LSTM temporal modeling
- Early stopping
- ReduceLROnPlateau
- Model checkpoints
- TensorBoard logging
- Mixed precision policy

Optional export tooling:

```bash
pip install -r backend/requirements-export.txt
```

## Fine-tuning the backbone (identity-disjoint)

The pipeline above only trains the temporal head on top of a **frozen**,
generic ImageNet EfficientNet-B0. An identity-disjoint audit
(`backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/`,
`backend/artifacts/models/torch/failure_audit_20261023/audit_report.md`)
found that this setup cannot actually detect manipulation: once
identity/domain shortcuts are removed, validation/test balanced accuracy
collapses to ~0.50 (random), and external datasets (SDFVD, UADFV) score
~50% too. Frozen generic features do not carry forgery-specific signal.

`scripts/finetune_torch_video.py` fine-tunes the trailing EfficientNet-B0
stages end-to-end on raw frames (not cached features), on the same
identity-disjoint split, so the model has a chance to learn real artifacts
instead of identity/domain shortcuts.

### 1. Get the dataset

Raw Celeb-DF-v2 videos are not included in this repo (only cached, frozen
features were previously retained, and those can't be used for backbone
fine-tuning). Request access from the dataset authors:

1. Open <https://github.com/yuezunli/celeb-deepfakeforensics> and follow the
   request-form link in the README (a Google/Tencent form).
2. Wait for the approval email with the download link (not instant — this
   is a manual review by the dataset authors).
3. Extract it so that these folders exist:

   ```text
   datasets/Celeb-DF-v2/Celeb-real/*.mp4
   datasets/Celeb-DF-v2/Celeb-synthesis/*.mp4
   datasets/Celeb-DF-v2/YouTube-real/*.mp4
   ```

### 2. Fine-tune

```bash
python scripts/finetune_torch_video.py --epochs 15
```

By default this reuses the already-verified identity-disjoint manifest
(zero identity overlap across train/val/test; see
`celebdf_identity_disjoint_20261023/validation_report.md`), unfreezes the
last 3 of EfficientNet-B0's 9 feature stages (`--unfreeze-stages`), and
trains the backbone and temporal head with separate learning rates
(`--backbone-lr 1e-5`, `--head-lr 1e-3`) so the pretrained filters aren't
destroyed by a head-sized learning rate. Horizontal-flip and
brightness/contrast augmentation is applied per-video (same across all its
frames, to avoid faking temporal flicker) and can be disabled with
`--no-augment`.

Resume an interrupted run:

```bash
python scripts/finetune_torch_video.py \
  --resume backend/artifacts/models/torch/celebdf_finetune/latest_checkpoint.pt
```

Decoding raw video on every batch is much slower than reading cached
features — expect this to take hours on CPU/MPS (no CUDA GPU on this
machine). Run a quick smoke test on a handful of videos first with
`--limit 10 --epochs 1` before committing to a full run.

Outputs land in `backend/artifacts/models/torch/celebdf_finetune/`
(`best_checkpoint.pt`, `training_history.csv`, `test_metrics.json`, …),
mirroring the cached-feature pipeline's conventions. The test split is
evaluated exactly once, at a fixed 0.5 threshold, after training finishes —
never used for tuning.

If this still can't beat random on the identity-disjoint test set, the next
documented step is adding a second, domain-diverse dataset (FaceForensics++
or DFDC) rather than tuning further on Celeb-DF-v2 alone — see
`backend/artifacts/models/torch/failure_audit_20261023/audit_report.md`
section 7.
