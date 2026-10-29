"""Download and prepare a subset of WildDeepfake (xingjunm/WildDeepfake on
Hugging Face) as a second, domain-diverse dataset for backbone fine-tuning.

WildDeepfake is internet-collected "in the wild" deepfakes - a different
domain and generation pipeline than Celeb-DF-v2's studio-sourced face swaps.
The identity-disjoint audit (see docs/TRAINING.md "Fine-tuning the backbone"
section) found Celeb-DF-v2 alone insufficient for the model to learn
generalizable manipulation artifacts; mixing in a diverse second dataset is
the documented next step.

Each shard (<id>.tar.gz) is one face sequence: a folder of already
face-cropped, already 224x224 PNG frames, extracted from one source video.
That maps directly onto one "video" sample in the existing finetuning
pipeline, with no MTCNN step required.

Usage:
    python scripts/prepare_wilddeepfake.py --fake-train 60 --real-train 60 \
        --fake-test 20 --real-test 20

Writes extracted sequences under datasets/WildDeepfake/<split>/<label>/<shard_id>/
and a manifest compatible with finetune_torch_video.py's --manifest/--dataset-root
at datasets/WildDeepfake/manifest.csv (columns: relative_path, label, split).
"""

from __future__ import annotations

import argparse
import random
import sys
import tarfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "xingjunm/WildDeepfake"
DEFAULT_OUTPUT = ROOT / "datasets" / "WildDeepfake"

# label: 0 = real, 1 = fake, matching the Celeb-DF manifest convention.
SPLIT_SPECS = [
    ("fake_train", "train", 1),
    ("real_train", "train", 0),
    ("fake_test", "test", 1),
    ("real_test", "test", 0),
]


def list_shards(folder: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    prefix = f"deepfake_in_the_wild/{folder}/"
    return [f for f in files if f.startswith(prefix)]


def download_and_extract(repo_path: str, label: int, split: str, output: Path, seed: int) -> str | None:
    shard_id = Path(repo_path).stem.replace(".tar", "")
    dest_dir = output / split / ("fake" if label == 1 else "real") / shard_id
    if dest_dir.exists() and any(dest_dir.iterdir()):
        return dest_dir.relative_to(output).as_posix()

    local_tar = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=repo_path,
        local_dir=output / "_shards",
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(local_tar) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and not m.name.startswith("..")]
            for member in members:
                member.name = Path(member.name).name  # flatten any nested dirs
            tar.extractall(dest_dir, members=members, filter="data")
    except Exception as exc:
        print(f"FAILED to extract {repo_path}: {exc}")
        return None
    finally:
        Path(local_tar).unlink(missing_ok=True)

    if not any(dest_dir.glob("*.png")) and not any(dest_dir.glob("*.jpg")):
        print(f"WARNING: no images found after extracting {repo_path}")
        return None
    return dest_dir.relative_to(output).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake-train", type=int, default=60)
    parser.add_argument("--real-train", type=int, default=60)
    parser.add_argument("--fake-test", type=int, default=20)
    parser.add_argument("--real-test", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = {
        "fake_train": args.fake_train,
        "real_train": args.real_train,
        "fake_test": args.fake_test,
        "real_test": args.real_test,
    }
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for folder, split, label in SPLIT_SPECS:
        wanted = counts[folder]
        if wanted <= 0:
            continue
        available = list_shards(folder)
        if wanted > len(available):
            print(f"Only {len(available)} shards available in {folder}; requested {wanted}.")
            wanted = len(available)
        chosen = random.sample(available, wanted)
        print(f"{folder}: downloading {len(chosen)}/{len(available)} shards...")
        for i, repo_path in enumerate(chosen, start=1):
            relative = download_and_extract(repo_path, label, split, args.output, args.seed)
            if relative:
                rows.append({"relative_path": relative, "label": label, "split": split})
            if i % 10 == 0 or i == len(chosen):
                print(f"  {folder}: {i}/{len(chosen)}", flush=True)

    if not rows:
        print("No shards downloaded; nothing to write.")
        sys.exit(1)

    manifest = pd.DataFrame(rows)
    # Carve a val split out of train (10%, stratified by label so small
    # subsets can't end up with a val split missing a class), since the
    # source only has train/test.
    train_mask = manifest["split"] == "train"
    val_idx = (
        manifest[train_mask]
        .groupby("label", group_keys=False)[["label"]]
        .apply(lambda g: g.sample(n=max(1, round(len(g) * 0.1)), random_state=args.seed))
        .index
    )
    manifest.loc[val_idx, "split"] = "val"

    manifest_path = args.output / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    shards_dir = args.output / "_shards"
    if shards_dir.exists():
        import shutil

        shutil.rmtree(shards_dir, ignore_errors=True)

    print(f"\nWrote {len(manifest)} rows to {manifest_path}")
    print(manifest.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
