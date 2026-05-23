from __future__ import annotations

import os
from pathlib import Path


def resolve_dataset_root(project_root: Path, configured: str | Path) -> Path:
    """
    Resolve perio-KPT root. Checks env PERIO_KPT_ROOT, config path, then common Colab/Drive locations.
    Must contain 1_Experiment/standard_box/.
    """
    env = os.environ.get("PERIO_KPT_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    p = Path(configured)
    candidates.append(p if p.is_absolute() else project_root / p)

    candidates.extend(
        [
            project_root / "data" / "perio-KPT",
            Path("/content/drive/MyDrive/perio-KPT"),
            Path("/content/drive/MyDrive/datasets/perio-KPT"),
            Path("/content/drive/MyDrive/Colab Notebooks/perio-KPT"),
            Path("/content/perio-KPT"),
        ]
    )

    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if _is_valid_perio_kpt_root(cand):
            return cand

        # One level of nesting (common after unzip)
        if cand.is_dir():
            for child in cand.iterdir():
                if child.is_dir() and _is_valid_perio_kpt_root(child):
                    return child.resolve()

    tried = "\n  ".join(str(c) for c in seen)
    raise FileNotFoundError(
        "perio-KPT dataset not found. Expected a folder containing "
        "`1_Experiment/standard_box/`.\n\n"
        "Colab steps:\n"
        "  1. Request access: https://zenodo.org/records/17272200\n"
        "  2. Download the zip to Google Drive and extract it\n"
        "  3. Run the notebook cell 'Link dataset from Drive' OR set:\n"
        "       os.environ['PERIO_KPT_ROOT'] = '/content/drive/MyDrive/<your-folder>'\n"
        "  4. Or symlink: ln -s /content/drive/MyDrive/<folder> data/perio-KPT\n\n"
        f"Paths checked:\n  {tried}"
    )


def _is_valid_perio_kpt_root(path: Path) -> bool:
    return (path / "1_Experiment" / "standard_box").is_dir()
