# Anatomically-Constrained LDMs for Rare Periodontitis Case Augmentation

This repo contains the two notebooks for the project: a baseline (unconstrained) Latent Diffusion Model, and the full anatomically-guided pipeline with downstream evaluation.

| Notebook | What it is |
|---|---|
| `baseline.ipynb` | Unconstrained 2D LDM on rare periodontal lesion patches. Establishes the "why we need anatomical guidance" evidence (structural error analysis). |
| `ldm_and_vae.ipynb` | Full pipeline: from-scratch VAE + ControlNet-style mask/PBL-conditioned latent diffusion, sampling ablations, synthetic data generation, and downstream YOLOv8-pose evaluation (real-only vs. real+synthetic). |

Both notebooks are self-contained (imports → data → network → train → evaluation) and were built/run on **Google Colab with a GPU runtime**.

---

## 1. Requirements

- Google Colab (recommended), or a local Jupyter environment with a CUDA GPU
- Python packages used across the notebooks (installed automatically or already present on Colab):
  - `torch`, `numpy`, `pillow`, `scipy`, `scikit-image`, `matplotlib`, `tqdm`, `pandas`
  - `ultralytics` (installed in-notebook via `!pip install ultralytics -q`, used only in `ldm_and_vae.ipynb` for the downstream YOLOv8-pose step)

If running locally instead of Colab, remove/skip the Google Drive mount cell and point the path variables (see below) directly at your local copy of the dataset.

---

## 2. Dataset setup

Both notebooks expect the **Periodontal Bone Loss Keypoint and Detection Dataset (`perio_KPT`)** (https://zenodo.org/records/17272200 link to dataset dataset requires getting access from owner of the dataset but you can get responsy quickly)as a zip file in Google Drive:

```
/content/drive/MyDrive/CV_project/perio_KPT.zip
```

Inside the zip, the dataset is organized into subfolders used by different parts of the pipeline:

- `0_Baseline/`, `1_Experiment/ - `images/` + YOLO-pose style `labels/*.txt` (11 keypoints/tooth). Used by both notebooks for training patches.
- `3_External_Set/` - the held-out external test set (`ldm_and_vae.ipynb` only), kept completely untouched by synthetic data.

**Before running either notebook:**
1. Upload `perio_KPT.zip` to `MyDrive/CV_project/` in the Google account you'll use for Colab.
2. If you use a different path or folder layout, update the `ZIP_PATH` (and, in `ldm_and_vae.ipynb`, `EXTERNAL_ROOT`) variable near the top of the notebook.

---

## 3. Running `baseline.ipynb`

**Purpose:** train an unconstrained VAE + latent diffusion model on rare lesion patches, then show - quantitatively and visually - how it hallucinates anatomy without any guidance.

**Steps:**
1. Open in Colab, select a GPU runtime
2. Run cells top to bottom:
   - **Imports** → **Colab mount** (authenticates and mounts Drive) → **data extraction** (unzips `0_Baseline/` patches)
   - **Globals** (`cfg` dict - seed, patch size, VAE/LDM hyperparameters, all editable here)
   - **Utils / Data** - extracts rare-lesion patches (ARR, PLS, high-PBL, furcation) and builds the train/val split
   - **Network** - defines `PatchVAE`, `LatentUNet`, `GaussianDiffusion`
   - **Train** - trains the VAE (60 epochs) then the latent diffusion model (200 epochs or less 200 epochs is a lilte to uneccesssary) on an augmented rare-patch latent bank
   - **Evaluation** - DDIM-samples new patches, decodes them, and runs the structural error analysis (edge density, gradient energy, SSIM-to-nearest-real, bone-band contrast) comparing generated vs. real patches

**Outputs** (written under `checkpoints/` and `outputs/` relative to the notebook's working directory):
- `checkpoints/vae/best_vae.pt`, `checkpoints/ldm/` - model weights
- `outputs/baseline_ldm/sample_*.png` - generated patches
- `outputs/structural_error_analysis/` - comparison plots and the structural error report

This notebook has no dependency on `ldm_and_vae.ipynb` - it can be run entirely on its own.

---

## 4. Running `ldm_and_vae.ipynb`

**Purpose:** the full anatomically-constrained pipeline - a from-scratch VAE plus a ControlNet-style mask/PBL conditioning branch injected into the latent diffusion U-Net, followed by ablations and a downstream detection benchmark.

**Steps:**
1. Open in Colab with a GPU runtime.
2. Run top to bottom through these sections:
   - **Imports → Colab setup → Globals** - the `cfg` dict here controls everything (image size 256, VAE/latent-diffusion hyperparameters, `guidance_scale`, `ddim_steps`, downstream YOLO settings). Check the sanity-check cell after the data split - it should show the rare Stage-IV images present in **both** train and val (stratified split).
   - **Network** - `VAE` (grayscale, f=8, 256→32×32×4), `ControlConditionEncoder` (full-res mask/PBL tower with zero-conv injection), `LatentUNet`
   - **Train - Stage 1: VAE**, then **Train - Stage 2: latent diffusion**
   - **Evaluation** - samples from real validation masks across severity classes (Healthy/Mild/Moderate/Severe)
   - **Ablation** - sweeps `guidance_scale` (0.5–5.5) and `ddim_steps` (10/25/50/100) using a fixed seed, to pick the operating point (the notebook lands on `guidance_scale=3.5`, `ddim_steps=25`)
   - **Data generation** - creates the synthetic augmentation set: for every rare-tagged training image, generates `samples_per_rare` (default 2) synthetic variants conditioned on its real mask
   - **Downstream (`!pip install ultralytics`)** - trains two YOLOv8-pose models (`real_only` vs. `real_plus_synth`), then evaluates both on the identical, untouched external test set and writes a per-class mAP50 / mAP50-95 comparison CSV

**Outputs:**
- `checkpoints/mask_ldm/` - VAE and latent-diffusion weights
- `outputs/mask_guided_ldm_v3/` - generated samples and comparison figures
- `data/synthetic/rare_augmentation/` - the synthetic image+label pairs used for downstream training
- `outputs/downstream_yolo/{real_only,real_plus_synth}/` - YOLO training runs and weights
- `outputs/downstream_yolo/real_vs_synth_comparison.csv` - the final mAP comparison table


---

## 5. Recommended run order

1. `baseline.ipynb` first - cheapest way to confirm the data pipeline works and to see why unconstrained generation isn't good enough.
2. `ldm_and_vae.ipynb` second - this is the notebook that produces the actual project deliverables (constrained samples, ablations, downstream mAP results).

They don't share checkpoints or state, so either can technically be run independently or in parallel.

---
