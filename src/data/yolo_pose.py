from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# perio-KPT: 11 keypoints per tooth bbox (YOLO pose)
NUM_KEYPOINTS = 11
KPT_NAMES = [
    "CEJ-m",
    "BL-m",
    "RL-m",
    "CEJ-d",
    "BL-d",
    "RL-d",
    "RL-c",
    "FA",
    "FBL-m",
    "FBL-d",
    "ARR",
]
TOOTH_BOX_CLASSES = {0, 1, 2}


@dataclass
class ToothAnnotation:
    class_id: int
    bbox_xywh_norm: tuple[float, float, float, float]
    keypoints: np.ndarray  # (11, 3) x, y, visibility — normalized if from file

    def visible_keypoints_xy(self, img_w: int, img_h: int) -> list[tuple[float, float]]:
        pts = []
        for i in range(NUM_KEYPOINTS):
            x, y, v = self.keypoints[i]
            if v < 1:
                continue
            if x <= 0 and y <= 0:
                continue
            pts.append((x * img_w, y * img_h))
        return pts


def parse_yolo_pose_line(line: str) -> ToothAnnotation:
    parts = line.strip().split()
    if len(parts) < 5 + NUM_KEYPOINTS * 3:
        raise ValueError(f"Expected {5 + NUM_KEYPOINTS * 3} fields, got {len(parts)}")

    class_id = int(parts[0])
    bbox = tuple(float(x) for x in parts[1:5])
    kpt_flat = [float(x) for x in parts[5 : 5 + NUM_KEYPOINTS * 3]]
    keypoints = np.array(kpt_flat, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
    return ToothAnnotation(class_id=class_id, bbox_xywh_norm=bbox, keypoints=keypoints)


def bbox_norm_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def expand_bbox(
    x1: int, y1: int, x2: int, y2: int, margin: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(img_w, x2 + mx),
        min(img_h, y2 + my),
    )


def estimate_pbl_ratio(ann: ToothAnnotation, img_w: int, img_h: int) -> float | None:
    """Approximate bone loss ratio from BL vs CEJ and RL keypoints (mesial/distal average)."""

    def _xy(idx: int) -> tuple[float, float] | None:
        x, y, v = ann.keypoints[idx]
        if v < 2 or (x <= 0 and y <= 0):
            return None
        return x * img_w, y * img_h

    # indices: CEJ-m=0, BL-m=1, RL-m=2, CEJ-d=3, BL-d=4, RL-d=5, RL-c=6
    ratios = []
    for cej_i, bl_i, rl_i in [(0, 1, 2), (3, 4, 5)]:
        cej, bl, rl = _xy(cej_i), _xy(bl_i), _xy(rl_i)
        if cej is None or bl is None or rl is None:
            continue
        root_len = abs(rl[1] - cej[1])
        if root_len < 1e-3:
            continue
        bone_drop = abs(bl[1] - cej[1])
        ratios.append(bone_drop / root_len)
    if not ratios:
        return None
    return float(np.mean(ratios))
