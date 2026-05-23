from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.patch_dataset import PatchDataset
from src.diffusion.schedule import DiffusionSchedule
from src.models.unet_ldm import LatentUNet
from src.models.vae import VAE
from src.utils.config import load_config, project_root
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_ldm.yaml")
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Path to VAE best.pt (default: outputs/vae/best.pt)",
    )
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_path = Path(args.vae_checkpoint or root / cfg["output_dir"] / "vae" / "best.pt")
    if not vae_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}. Run train_vae first.")

    vae = VAE(
        latent_channels=int(cfg["vae_latent_channels"]),
        base_channels=int(cfg["vae_base_channels"]),
    ).to(device)
    ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    latent_ch = int(cfg["vae_latent_channels"])
    train_ds = PatchDataset(
        root / cfg["manifest_path"],
        root,
        split_filter="train",
        rare_only=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["diffusion_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    schedule = DiffusionSchedule(timesteps=int(cfg["diffusion_steps"])).to(device)
    unet = LatentUNet(
        in_channels=latent_ch,
        base_channels=int(cfg["unet_base_channels"]),
        channel_mult=tuple(cfg["unet_channel_mult"]),
    ).to(device)

    opt = torch.optim.AdamW(unet.parameters(), lr=float(cfg["diffusion_lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")

    out_dir = root / cfg["output_dir"] / "ldm"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["diffusion_epochs"])
    T = schedule.timesteps

    for epoch in range(epochs):
        unet.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"LDM epoch {epoch+1}/{epochs}"):
            x = batch["image"].to(device)
            with torch.no_grad():
                z0 = vae.encode_latent(x)

            t = torch.randint(0, T, (z0.shape[0],), device=device)
            zt, noise = schedule.q_sample(z0, t)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = unet(zt, t)
                loss = F.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_loader)
        print(f"epoch {epoch+1}: diffusion_loss={avg:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "unet": unet.state_dict(),
                "vae_path": str(vae_path),
                "config": cfg,
            },
            out_dir / "last.pt",
        )

    torch.save(
        {"epoch": epochs - 1, "unet": unet.state_dict(), "vae_path": str(vae_path), "config": cfg},
        out_dir / "best.pt",
    )
    print(f"Done. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
