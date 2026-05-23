from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.patch_dataset import PatchDataset
from src.models.vae import VAE
from src.utils.config import load_config, project_root
from src.utils.seed import set_seed


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_l = F.l1_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_l + kl_weight * kl
    return loss, {"recon": recon_l.item(), "kl": kl.item()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_ldm.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = root / cfg["output_dir"] / "vae"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PatchDataset(
        root / cfg["manifest_path"],
        root,
        split_filter="train",
        rare_only=False,
    )
    val_ds = PatchDataset(
        root / cfg["manifest_path"],
        root,
        split_filter="test",
        rare_only=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["vae_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["vae_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
    )

    model = VAE(
        latent_channels=int(cfg["vae_latent_channels"]),
        base_channels=int(cfg["vae_base_channels"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["vae_lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")

    best_val = float("inf")
    epochs = int(cfg["vae_epochs"])
    kl_w = float(cfg["vae_kl_weight"])

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"VAE epoch {epoch+1}/{epochs}"):
            x = batch["image"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                recon, mu, logvar = model(x)
                loss, _ = vae_loss(recon, x, mu, logvar, kl_w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                recon, mu, logvar = model(x)
                loss, _ = vae_loss(recon, x, mu, logvar, kl_w)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        print(f"epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} val_loss={val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best (val_loss={best_val:.4f})")

    print(f"Done. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
