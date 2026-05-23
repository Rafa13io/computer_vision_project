from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.yolo_pose import (
    TOOTH_BOX_CLASSES,
    bbox_norm_to_xyxy,
    estimate_pbl_ratio,
    expand_bbox,
    parse_yolo_pose_line,
)
from src.utils.config import load_config, project_root


def _load_severe_stems(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    stems = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stems.add(Path(line).stem)
    return stems


def _find_image_label_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for images_root in [split_dir / "images", split_dir]:
        if not images_root.is_dir():
            continue
        seen: set[Path] = set()
        for img_path in sorted(images_root.rglob("*")):
            if img_path.suffix.lower() != ".png":
                continue
            if img_path in seen:
                continue
            seen.add(img_path)
            label_path = (
                split_dir / "labels" / f"{img_path.stem}.txt"
                if (split_dir / "labels").is_dir()
                else img_path.with_suffix(".txt")
            )
            if not label_path.is_file():
                alt = split_dir / "labels" / img_path.name.replace(
                    img_path.suffix, ".txt"
                )
                label_path = alt if alt.is_file() else label_path
            if label_path.is_file():
                pairs.append((img_path, label_path))
        if pairs:
            break
    return pairs


def _collect_splits(dataset_root: Path, cfg: dict) -> list[tuple[str, str, Path]]:
    """Return (split_name, fold_id, directory) for train/test under standard_box."""
    exp = dataset_root / cfg["experiment_subdir"]
    fold_name = cfg.get("fold_name", "fold0")
    entries: list[tuple[str, str, Path]] = []

    fold_dir = exp / fold_name
    if fold_dir.is_dir():
        for split in ("train", "test"):
            sd = fold_dir / split
            if sd.is_dir():
                entries.append((split, fold_name, sd))
    else:
        # try fold0, fold1, ... or flat train/test under exp
        folds = sorted(exp.glob("fold*"))
        if folds:
            for fd in folds:
                for split in ("train", "test"):
                    sd = fd / split
                    if sd.is_dir():
                        entries.append((split, fd.name, sd))
        else:
            for split in ("train", "test"):
                sd = exp / split
                if sd.is_dir():
                    entries.append((split, "single", sd))

    holdout = dataset_root / cfg.get("holdout_subdir", "")
    if holdout.is_dir():
        entries.append(("holdout", "holdout", holdout))

    return entries


def build_manifest(cfg: dict, root: Path | None = None) -> Path:
    root = root or project_root()
    dataset_root = root / cfg["dataset_root"]
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}. "
            "Download perio-KPT from https://zenodo.org/records/17272200"
        )

    patches_dir = root / cfg["patches_dir"]
    patches_dir.mkdir(parents=True, exist_ok=True)

    severe_stems = _load_severe_stems(root / cfg["severe_images_file"])
    pbl_thresh = float(cfg.get("pbl_ratio_threshold", 0.30))
    tooth_classes = set(cfg.get("tooth_box_classes", [0, 1, 2]))
    margin = float(cfg.get("bbox_margin", 0.10))
    patch_size = int(cfg["patch_size"])

    rows: list[dict] = []
    patch_idx = 0

    for split_name, fold_id, split_dir in _collect_splits(dataset_root, cfg):
        pairs = _find_image_label_pairs(split_dir)
        for img_path, label_path in tqdm(
            pairs, desc=f"Processing {fold_id}/{split_name}"
        ):
            image = Image.open(img_path).convert("L")
            img_w, img_h = image.size
            stem = img_path.stem
            is_severe_image = stem in severe_stems

            lines = [
                ln
                for ln in label_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            tooth_idx = 0
            for line in lines:
                ann = parse_yolo_pose_line(line)
                if ann.class_id not in tooth_classes:
                    continue

                x1, y1, x2, y2 = bbox_norm_to_xyxy(
                    *ann.bbox_xywh_norm, img_w, img_h
                )
                x1, y1, x2, y2 = expand_bbox(
                    x1, y1, x2, y2, margin, img_w, img_h
                )
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image.crop((x1, y1, x2, y2)).resize(
                    (patch_size, patch_size), Image.BILINEAR
                )
                pbl = estimate_pbl_ratio(ann, img_w, img_h)
                is_rare = is_severe_image or (
                    pbl is not None and pbl >= pbl_thresh
                )

                rel_name = f"{fold_id}_{split_name}_{stem}_t{tooth_idx}.png"
                out_path = patches_dir / rel_name
                crop.save(out_path)

                rows.append(
                    {
                        "patch_id": patch_idx,
                        "patch_path": str(out_path.relative_to(root)),
                        "image_stem": stem,
                        "tooth_idx": tooth_idx,
                        "fold": fold_id,
                        "split": split_name,
                        "root_class": ann.class_id,
                        "is_severe_image": int(is_severe_image),
                        "is_rare": int(is_rare),
                        "pbl_ratio": "" if pbl is None else f"{pbl:.4f}",
                    }
                )
                patch_idx += 1
                tooth_idx += 1

    manifest_path = root / cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "patch_id",
        "patch_path",
        "image_stem",
        "tooth_idx",
        "fold",
        "split",
        "root_class",
        "is_severe_image",
        "is_rare",
        "pbl_ratio",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_rare = sum(int(r["is_rare"]) for r in rows)
    print(f"Wrote {len(rows)} patches ({n_rare} rare) to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patch manifest from perio-KPT")
    parser.add_argument(
        "--config",
        default="configs/baseline_ldm.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()
    root = project_root()
    cfg = load_config(root / args.config)
    build_manifest(cfg, root)


if __name__ == "__main__":
    main()
