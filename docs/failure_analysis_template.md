# Baseline LDM — structural error analysis

After training on **rare/severe** tooth patches and running `python -m src.train.sample`, fill this template for your report.

## Setup

- Dataset: perio-KPT ([Zenodo](https://zenodo.org/records/17272200))
- Rare selection rule: _(severe_images.txt / PBL threshold)_
- N rare train patches: _
- Checkpoints: `outputs/vae/best.pt`, `outputs/ldm/best.pt`

## Error categories (add 1–2 figure pairs each)

### 1. Wrong alveolar bone level
- Description:
- Example: `outputs/samples/...`

### 2. CEJ blur or missing
- Description:
- Example:

### 3. Incorrect root morphology
- Description:
- Example:

### 4. Texture without valid anatomy
- Description:
- Example:

### 5. Crown/root boundary artifacts
- Description:
- Example:

## Summary

What unconstrained LDM gets right vs wrong, and why anatomical guidance (Phase 2) is needed.
