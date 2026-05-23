from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.data.patch_dataset import PatchDataset
from src.diffusion.schedule import DiffusionSchedule
from src.diffusion.sampler import ddim_sample, decode_latents
from src.models.latent_shape import latent_spatial_size
from src.models.unet_ldm import LatentUNet
from src.models.vae import VAE
from src.utils.config import load_config, project_root
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_ldm.yaml")
    parser.add_argument("--ldm-checkpoint", default=None)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)
    set_seed(int(cfg.get("sample_seed", cfg["seed"])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = root / cfg["output_dir"] / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_path = Path(
        args.vae_checkpoint or root / cfg["output_dir"] / "vae" / "best.pt"
    )
    ldm_path = Path(
        args.ldm_checkpoint or root / cfg["output_dir"] / "ldm" / "best.pt"
    )

    vae = VAE(
        latent_channels=int(cfg["vae_latent_channels"]),
        base_channels=int(cfg["vae_base_channels"]),
    ).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=False)["model"])
    vae.eval()

    unet = LatentUNet(
        in_channels=int(cfg["vae_latent_channels"]),
        base_channels=int(cfg["unet_base_channels"]),
        channel_mult=tuple(cfg["unet_channel_mult"]),
    ).to(device)
    unet.load_state_dict(torch.load(ldm_path, map_location=device, weights_only=False)["unet"])
    unet.eval()

    schedule = DiffusionSchedule(timesteps=int(cfg["diffusion_steps"])).to(device)
    n = int(args.num_samples or cfg["num_samples"])
    latent_ch = int(cfg["vae_latent_channels"])
    spatial = latent_spatial_size(vae, int(cfg["patch_size"]))
    z = ddim_sample(
        unet,
        schedule,
        (n, latent_ch, spatial, spatial),
        steps=int(cfg["ddim_steps"]),
        device=device,
    )
    fake = decode_latents(vae, z)
    vutils.save_image(
        (fake + 1) / 2,
        out_dir / "unconditional_samples.png",
        nrow=min(8, n),
    )
    print(f"Saved {n} samples to {out_dir / 'unconditional_samples.png'}")

    # Real rare patches for comparison
    try:
        real_ds = PatchDataset(
            root / cfg["manifest_path"],
            root,
            split_filter="train",
            rare_only=True,
        )
        loader = DataLoader(real_ds, batch_size=min(n, len(real_ds)), shuffle=True)
        batch = next(iter(loader))
        real = batch["image"][:n]
        vutils.save_image(
            (real + 1) / 2,
            out_dir / "real_rare_patches.png",
            nrow=min(8, n),
        )
        ncmp = min(4, n, real.shape[0], fake.shape[0])
        combined = torch.cat([real[:ncmp], fake[:ncmp]], dim=0)
        vutils.save_image(
            (combined + 1) / 2,
            out_dir / "real_vs_fake_top4.png",
            nrow=4,
        )
        print(f"Saved comparison grids to {out_dir}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Skipped real patch comparison: {e}")


if __name__ == "__main__":
    main()
