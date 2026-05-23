"""Colab helper: run via `%run colab_setup.py` from notebooks/ or project root."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    checked: set[Path] = set()
    candidates: list[Path] = []
    p = Path.cwd().resolve()
    for _ in range(8):
        candidates.append(p)
        if p.parent == p:
            break
        p = p.parent
    candidates.extend(
        [
            Path("/content/computer_vision_project"),
            Path("/content/drive/MyDrive/computer_vision_project"),
        ]
    )
    for cand in candidates:
        cand = cand.resolve()
        if cand in checked:
            continue
        checked.add(cand)
        if (cand / "src" / "utils" / "config.py").is_file():
            return cand
    raise FileNotFoundError(
        "Cannot find project root. Clone repo to /content/computer_vision_project"
    )


def setup() -> Path:
    root = find_project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


if __name__ == "__main__":
    r = setup()
    print("ROOT =", r)
