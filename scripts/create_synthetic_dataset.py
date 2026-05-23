"""Create a tiny fake perio-KPT layout for pipeline smoke tests (no real clinical data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/perio-KPT/1_Experiment/standard_box/fold0"),
    )
    parser.add_argument("--num-images", type=int, default=8)
    args = parser.parse_args()

    train_img = args.out / "train" / "images"
    train_lbl = args.out / "train" / "labels"
    test_img = args.out / "test" / "images"
    test_lbl = args.out / "test" / "labels"
    for d in (train_img, train_lbl, test_img, test_lbl):
        d.mkdir(parents=True, exist_ok=True)

    def write_split(img_dir: Path, lbl_dir: Path, prefix: str, n: int) -> None:
        for i in range(n):
            name = f"{prefix}{i:02d}"
            img = Image.new("L", (512, 512), color=40)
            draw = ImageDraw.Draw(img)
            draw.ellipse((180, 80, 330, 420), fill=180)
            draw.rectangle((200, 200, 310, 230), fill=90)
            img.save(img_dir / f"{name}.png")

            # class 0 single root + 11 kpts x,y,vis
            line = [0, 0.5, 0.5, 0.35, 0.7]
            kpts = [
                (0.45, 0.35, 2),
                (0.48, 0.42, 2),
                (0.5, 0.75, 2),
                (0.55, 0.35, 2),
                (0.52, 0.45, 2),
                (0.5, 0.78, 2),
                (0.0, 0.0, 0),
                (0.0, 0.0, 0),
                (0.0, 0.0, 0),
                (0.0, 0.0, 0),
                (0.0, 0.0, 0),
            ]
            for x, y, v in kpts:
                line.extend([x, y, float(v)])
            lbl_dir.joinpath(f"{name}.txt").write_text(
                " ".join(str(x) for x in line) + "\n", encoding="utf-8"
            )

    write_split(train_img, train_lbl, "Image", args.num_images)
    write_split(test_img, test_lbl, "ImageT", max(2, args.num_images // 4))
    print(f"Created synthetic dataset under {args.out}")


if __name__ == "__main__":
    main()
