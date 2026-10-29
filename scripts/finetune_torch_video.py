"""Fine-tune the EfficientNet-B0 backbone end-to-end on raw video frames.

Cached-feature training (train_torch_temporal.py) only ever updates the
temporal head on top of frozen, generic ImageNet features. An identity-
disjoint re-evaluation of that approach (see
backend/artifacts/models/torch/celebdf_identity_disjoint_20261023/) showed
the resulting model collapses to near-random, degenerate predictions once
identity/domain shortcuts are removed: frozen ImageNet features apparently
do not carry the manipulation-specific signal needed for real detection.

This script instead decodes raw video frames and fine-tunes the trailing
stages of the backbone jointly with the temporal head, using the same
identity-disjoint split already produced for that experiment, so the
resulting model can actually learn forgery artifacts instead of only
identity/domain shortcuts.

Usage (once the Celeb-DF-v2 videos are downloaded and placed under
datasets/Celeb-DF-v2/{Celeb-real,Celeb-synthesis,YouTube-real}/):

    python scripts/finetune_torch_video.py --epochs 15

Each manifest row's relative_path may point to either a video file (decoded
with OpenCV) or a directory of already-extracted frame images (e.g. a
WildDeepfake shard prepared by scripts/prepare_wilddeepfake.py) - both are
handled transparently by decode_sequence(). To fine-tune on WildDeepfake
instead:

    python scripts/finetune_torch_video.py \
        --manifest datasets/WildDeepfake/manifest.csv \
        --dataset-root datasets/WildDeepfake \
        --output backend/artifacts/models/torch/wilddeepfake_finetune \
        --epochs 15

Resume an interrupted run:

    python scripts/finetune_torch_video.py \
        --resume backend/artifacts/models/torch/celebdf_finetune/latest_checkpoint.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_model import EfficientNetBiLSTM  # noqa: E402

SEQUENCE_LENGTH = 20
IMAGE_SIZE = 224

DEFAULT_MANIFEST = (
    ROOT
    / "backend"
    / "artifacts"
    / "models"
    / "torch"
    / "celebdf_identity_disjoint_20261023"
    / "celebdf_manifest.csv"
)
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "Celeb-DF-v2"
DEFAULT_OUTPUT = ROOT / "backend" / "artifacts" / "models" / "torch" / "celebdf_finetune"


def decode_video_frames(path: Path) -> np.ndarray:
    """Decode SEQUENCE_LENGTH evenly spaced RGB frames, resized, in [0, 1].

    Returns an array shaped (T, H, W, 3), float32.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise RuntimeError(f"Video has no reported frames: {path}")

        targets = np.linspace(0, count - 1, SEQUENCE_LENGTH, dtype=np.int64)
        targets_by_frame: dict[int, list[int]] = {}
        for output_index, frame_index in enumerate(targets):
            targets_by_frame.setdefault(int(frame_index), []).append(output_index)

        frames: list[np.ndarray | None] = [None] * SEQUENCE_LENGTH
        frame_index = 0
        while frame_index <= int(targets[-1]):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index in targets_by_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                frame = frame.astype(np.float32) / 255.0
                for output_index in targets_by_frame[frame_index]:
                    frames[output_index] = frame
            frame_index += 1

        if any(frame is None for frame in frames):
            decoded = sum(frame is not None for frame in frames)
            raise RuntimeError(
                f"Could not decode all selected frames ({decoded}/{SEQUENCE_LENGTH}): {path}"
            )
        return np.stack(frames)
    finally:
        cap.release()


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def decode_image_sequence_frames(path: Path) -> np.ndarray:
    """Decode SEQUENCE_LENGTH evenly spaced RGB frames from a directory of
    already-extracted, already face-cropped images (e.g. WildDeepfake
    shards), resized, in [0, 1]. Returns an array shaped (T, H, W, 3).

    Sequences shorter than SEQUENCE_LENGTH are padded by repeating the last
    selected frame, matching the padding behavior inference uses
    (backend/app/ml/torch_inference.py:faces_to_tensor) for short videos.
    """
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    count = len(images)
    if count < 1:
        raise RuntimeError(f"No images found in sequence directory: {path}")

    indices = np.linspace(0, count - 1, num=min(count, SEQUENCE_LENGTH), dtype=np.int64)
    frames: list[np.ndarray] = []
    for index in indices:
        image = cv2.imread(str(images[int(index)]))
        if image is None:
            raise RuntimeError(f"Could not decode image: {images[int(index)]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        frames.append(image.astype(np.float32) / 255.0)
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])
    return np.stack(frames)


def decode_sequence(path: Path) -> np.ndarray:
    """Dispatch to video or image-sequence decoding based on path type."""
    if path.is_dir():
        return decode_image_sequence_frames(path)
    return decode_video_frames(path)


class VideoFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Decodes raw video frames on the fly; optional video-level augmentation.

    Augmentation (horizontal flip, brightness/contrast jitter) is sampled
    once per video and applied identically to every frame, so it does not
    introduce its own frame-to-frame flicker that could be mistaken for a
    manipulation artifact.
    """

    def __init__(self, frame: pd.DataFrame, dataset_root: Path, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        path = self.dataset_root / str(row["relative_path"])
        frames = decode_sequence(path)

        if self.augment:
            if random.random() < 0.5:
                frames = np.ascontiguousarray(frames[:, :, ::-1, :])
            contrast = random.uniform(0.85, 1.15)
            brightness = random.uniform(-0.05, 0.05)
            frames = np.clip(frames * contrast + brightness, 0.0, 1.0)

        tensor = torch.from_numpy(frames.astype(np.float32, copy=False))
        tensor = tensor.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        return tensor, torch.tensor(float(row["label"]), dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(labels: list[float], logits: list[float]) -> dict:
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    result = {
        "loss": None,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["roc_auc"] = None
    return result


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[float] = []
    logits: list[float] = []
    for videos, targets in loader:
        videos = videos.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
            output = model(videos)
            loss = criterion(output, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered")
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(videos)
                loss = criterion(output, targets)
        total_loss += loss.item() * len(targets)
        labels.extend(targets.detach().cpu().tolist())
        logits.extend(output.detach().cpu().tolist())
    result = compute_metrics(labels, logits)
    result["loss"] = total_loss / len(loader.dataset)
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_manifest(manifest_path: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    required = {"relative_path", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if limit:
        frame = (
            frame.groupby("split", group_keys=False)
            .apply(lambda g: g.sample(n=min(limit, len(g)), random_state=42))
            .reset_index(drop=True)
        )
    return frame


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)

    manifest = load_manifest(args.manifest, args.limit)
    train_frame = manifest[manifest["split"] == "train"]
    val_frame = manifest[manifest["split"] == "val"]
    test_frame = manifest[manifest["split"] == "test"]
    if train_frame.empty or val_frame.empty:
        raise ValueError("Both train and val rows are required in the manifest")

    missing_videos = [
        row["relative_path"]
        for _, row in manifest.iterrows()
        if not (args.dataset_root / str(row["relative_path"])).exists()
    ][:5]
    if missing_videos:
        raise FileNotFoundError(
            "Dataset videos not found under "
            f"{args.dataset_root}. Example missing paths: {missing_videos}. "
            "Download Celeb-DF-v2 and place Celeb-real/, Celeb-synthesis/, "
            "and YouTube-real/ under that directory (or pass --dataset-root)."
        )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()}
    config.update(
        {
            "device": str(device),
            "train_rows": len(train_frame),
            "val_rows": len(val_frame),
            "test_rows": len(test_frame),
        }
    )
    (output / "training_config.json").write_text(json.dumps(config, indent=2, default=str))

    train_sampler = None
    train_shuffle = True
    if args.balanced_sampler:
        # With severe class imbalance and a small batch size, plain shuffling
        # leaves most batches containing only the majority class - the model
        # never sees enough minority-class gradient signal per step and can
        # settle into a cheap "always predict the majority" optimum instead
        # of learning to discriminate. Oversample the minority class so every
        # batch has a real/fake mix, independent of loss-level pos_weight.
        class_counts = train_frame["label"].value_counts()
        sample_weights = train_frame["label"].map(lambda label: 1.0 / class_counts[label]).to_numpy()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_frame), replacement=True
        )
        train_shuffle = False

    train_loader = DataLoader(
        VideoFrameDataset(train_frame, args.dataset_root, augment=args.augment),
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        VideoFrameDataset(val_frame, args.dataset_root, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = EfficientNetBiLSTM(
        sequence_length=SEQUENCE_LENGTH,
        frame_chunk_size=args.frame_chunk_size,
        pretrained=True,
        unfreeze_stages=args.unfreeze_stages,
    ).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.projection.parameters())
        + list(model.lstm1.parameters())
        + list(model.lstm2.parameters())
        + list(model.classifier.parameters())
    )
    param_groups = [{"params": head_params, "lr": args.head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    trainable = sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head_params)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    positive = float((train_frame["label"] == 1).sum())
    negative = float((train_frame["label"] == 0).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, device=device)
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    start_epoch = 1
    best_val_loss = float("inf")
    latest = output / "latest_checkpoint.pt"
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"Resumed from epoch {start_epoch - 1}: {args.resume}")

    history_path = output / "training_history.csv"
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
        "seconds",
    ]
    if not args.resume or not history_path.exists():
        with history_path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writeheader()

    no_improvement = 0
    print(f"Device: {device}; train={len(train_frame)} val={len(val_frame)} test={len(test_frame)}")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device, None)
        scheduler.step(val_result["loss"])
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "accuracy": val_result["accuracy"],
            "precision": val_result["precision"],
            "recall": val_result["recall"],
            "f1": val_result["f1"],
            "roc_auc": val_result["roc_auc"],
            "confusion_matrix": json.dumps(val_result["confusion_matrix"]),
            "seconds": elapsed,
        }
        with history_path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            save_checkpoint(output / "best_checkpoint.pt", model, optimizer, scheduler, epoch, best_val_loss, config)
            no_improvement = 0
        else:
            no_improvement += 1
        save_checkpoint(latest, model, optimizer, scheduler, epoch, best_val_loss, config)
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} val_balanced_f1={val_result['f1']:.4f} "
            f"val_auc={val_result['roc_auc']} time={elapsed:.1f}s",
            flush=True,
        )
        if no_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    (output / "validation_metrics.json").write_text(json.dumps({"best_val_loss": best_val_loss}, indent=2))

    if not test_frame.empty:
        print("Evaluating once on the held-out test split (fixed 0.5 threshold)...")
        best_checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state"])
        test_loader = DataLoader(
            VideoFrameDataset(test_frame, args.dataset_root, augment=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        test_result = run_epoch(model, test_loader, criterion, device, None)
        (output / "test_metrics.json").write_text(json.dumps(test_result, indent=2))
        print(
            f"TEST accuracy={test_result['accuracy']:.4f} f1={test_result['f1']:.4f} "
            f"auc={test_result['roc_auc']} confusion_matrix={test_result['confusion_matrix']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-chunk-size", type=int, default=4)
    parser.add_argument("--unfreeze-stages", type=int, default=3, help="Trailing EfficientNet-B0 feature stages to fine-tune (0-9).")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument(
        "--no-balanced-sampler",
        dest="balanced_sampler",
        action="store_false",
        help="Disable per-batch class-balanced sampling (on by default) and fall back to plain shuffling.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap rows per split (e.g. for a smoke test).")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
