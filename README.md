# Project 4 — Baseline LDM (perio-KPT)

Unconditional **Latent Diffusion Model** on rare periodontal lesion patches for the Computer Vision course (Prof. Irene Amerini).

**Dataset:** [Periodontal Keypoint and Object Detection Dataset (perio-KPT)](https://zenodo.org/records/17272200) — cite [Banks et al., 2026](https://doi.org/10.1016/j.compbiomed.2026.111515) if you publish.

## Objective (current phase)

Train a standard 2D LDM on localized 256×256 tooth crops of rare/severe cases, sample unconditionally, and document structural errors in generated radiographs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Colab: upload this repo or clone, install requirements, mount Drive, set `dataset_root` in `configs/baseline_ldm.yaml`.

## Pipeline

```bash
# From project root (computer_vision_project/)
cd /path/to/computer_vision_project

# 1. After downloading perio-KPT to data/perio-KPT (see data/README.md)
python -m src.data.build_manifest --config configs/baseline_ldm.yaml

# 2. Train VAE on all train patches
python -m src.train.train_vae --config configs/baseline_ldm.yaml

# 3. Train unconditional LDM on rare patches only
python -m src.train.train_ldm --config configs/baseline_ldm.yaml

# 4. Sample and save grids (real vs fake)
python -m src.train.sample --config configs/baseline_ldm.yaml
```

Outputs: `outputs/vae/`, `outputs/ldm/`, `outputs/samples/`.

Local smoke test (fake data only): `python scripts/create_synthetic_dataset.py` then use `configs/baseline_ldm_smoke.yaml`.

Failure analysis template: [docs/failure_analysis_template.md](docs/failure_analysis_template.md).

## Project layout

| Path | Role |
|------|------|
| `configs/baseline_ldm.yaml` | Hyperparameters and paths |
| `configs/severe_images.txt` | Optional list of severe radiograph stems |
| `src/data/` | YOLO parsing, manifest builder, dataset |
| `src/models/` | VAE + latent U-Net |
| `src/diffusion/` | Schedule + DDIM sampler |
| `src/train/` | Training and sampling scripts |
| `notebooks/baseline_ldm.ipynb` | Colab-oriented walkthrough |

## Rare patch selection

1. **Preferred:** fill `configs/severe_images.txt` with the 12 severe radiograph stems from the paper.
2. **Fallback:** keypoint-based PBL ratio ≥ `pbl_ratio_threshold` in config.

## Failure analysis (deliverable)

After sampling, document error types in your report (examples in `outputs/samples/`):

- Wrong alveolar bone level
- CEJ blur or missing landmarks
- Incorrect root morphology
- Realistic texture without valid anatomy
- Crown/root boundary artifacts

## License

Code: course project. Dataset: CC BY-NC-SA — non-commercial research only.
