#!/usr/bin/env python3
"""Demonstrate how activations vary as transforms are applied."""
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from timm.utils.model import reparameterize_model
from torchvision.transforms import v2

from radiomana import NanoGRU
from radiomana.datasets import HighwayDataModule
from radiomana.transforms import LogNoise, RandomTimeCrop
from radiomana.visualization import plot9

WEIGHTS = Path(__file__).parent.parent / "nanogru-loss=0.501.pt"

ModelClass = NanoGRU
model = ModelClass()
model.load_state_dict(torch.load(WEIGHTS))
model = reparameterize_model(model)
model.eval()

dmodule = HighwayDataModule(num_workers=1, batch_size=8)
dmodule.setup()

batch = next(iter(dmodule.val_dataloader()))
inputs, labels = batch

sample_rate_hz = dmodule.data_test.sample_rate_hz
class_labels = dmodule.data_test.class_labels
max_channels = max(model.channels)

# calculate stage positions for subplot sizing
stage_positions = np.cumsum(model.num_blocks_per_stage)

# create demo directory if it doesn't exist
demo_dir = Path(__file__).parent.parent / "demo"
demo_dir.mkdir(exist_ok=True)

# create transform used in training for augmentation
train_transform = v2.Compose(
    [
        v2.RandomErasing(p=1, value=-90),
        RandomTimeCrop(crop_width=211),
        v2.RandomChoice(
            [
                v2.Identity(),
                LogNoise(noise_power_db=-110, p=1),
                LogNoise(noise_power_db=-90, p=1),
                LogNoise(noise_power_db=-70, p=1),
            ],
        ),
    ]
)

# generate date prefix for filenames
date_prefix = datetime.now().strftime("%Y%m%d")

# process each sample in batch
for batch_idx in range(inputs.shape[0]):
    # get clean sample first
    clean_sample = inputs[batch_idx : batch_idx + 1]  # keep batch dimension
    label = labels[batch_idx].item()
    label_name = class_labels[label] if label < len(class_labels) else f"class_{label}"
    # if label in [0, 2, 1, 3, 5, 6, 7, 8]:
    #     continue

    # generate 3 transform variants for quick testing (change to 30 for full video)
    for frame_idx in range(15):
        # apply transform to get augmented version
        augmented_sample = train_transform(clean_sample)

        # Take one sample from dataset, and plot how it flows through the network
        # columns are (input, stem output, conv stages..., freq pool, recurrent, logits)
        num_stage_positions = len(stage_positions)
        total_cols = 2 + num_stage_positions + 3  # input + stem + conv_stages + freq_pool + recurrent + logits
        fig, axes = plt.subplots(max_channels, total_cols, figsize=(16, 16))
        # turn off all axes and ticks
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])

        # plot input samples
        ax = axes[0, 0]
        ax.set_title("Input")
        # check actual tensor dimensions - it should be (batch, freq, time) = (1, 512, 243)
        if len(augmented_sample.shape) == 3:  # (batch, freq, time)
            freqs = np.linspace(-sample_rate_hz / 2, sample_rate_hz / 2, augmented_sample.shape[1])
            times_ms = np.linspace(0, augmented_sample.shape[2] / sample_rate_hz, augmented_sample.shape[2]) * 1000
        else:  # fallback for different shape
            freqs = np.linspace(-sample_rate_hz / 2, sample_rate_hz / 2, augmented_sample.shape[-2])
            times_ms = np.linspace(0, augmented_sample.shape[-1] / sample_rate_hz, augmented_sample.shape[-1]) * 1000
        extent = [times_ms[0], times_ms[-1], freqs[0], freqs[-1]]
        ax.imshow(augmented_sample[0].detach().cpu().squeeze(), aspect="auto", vmin=-57, vmax=-16, extent=extent)

        # pass through stem
        x = model.stem(augmented_sample)
        for cdx in range(x.shape[1]):
            ax = axes[cdx, 1]
            if cdx == 0:
                ax.set_title(f"Stem")
            sample = x[0, cdx].detach().cpu()
            ax.imshow(sample, aspect="auto", vmin=-0.1, vmax=8, extent=extent)

        # pass through conv stages
        for sdx, stage in enumerate(model.conv):
            x = stage(x)
            # plot only when sdx matches stage_positions
            if sdx in stage_positions:
                # get position index within stage_positions for column indexing
                stage_pos_idx = np.where(stage_positions == sdx)[0][0]
                for cdx in range(x.shape[1]):
                    ax = axes[cdx, 2 + stage_pos_idx]
                    if cdx == 0:
                        ax.set_title(f"Conv Stage {sdx}")
                    sample = x[0, cdx].detach().cpu()
                    ax.imshow(sample, aspect="auto", vmin=-0.1, vmax=8, extent=extent)
                # turn off rest of graphs in this column
                for rdx in range(x.shape[1], max_channels):
                    axes[rdx, 2 + stage_pos_idx].axis("off")
                    axes[rdx, 2 + stage_pos_idx].set_xticks([])
                    axes[rdx, 2 + stage_pos_idx].set_yticks([])

        # pass through freq pool and recurrent
        x = model.freq_pool(x)
        # plot freq pooled
        freq_pool_col = 2 + len(stage_positions)
        ax = axes[0, freq_pool_col]
        ax.set_title("Freq Pool")
        ax.imshow(x[0].detach().cpu().T, aspect="auto", vmin=0, vmax=16, extent=extent)

        x, _ = model.recurrent(x)
        # plot recurrent output
        recurrent_col = 2 + len(stage_positions) + 1
        ax = axes[0, recurrent_col]
        ax.set_title("Recurrent")
        ax.imshow(x[0].detach().cpu().T, aspect="auto", vmin=-1, vmax=1, extent=extent)

        x = model.time_pool(x)
        x = model.dropout(x)
        x = model.fc(x)
        # plot final output vector as bar graph
        logits_col = 2 + len(stage_positions) + 2
        ax = axes[0, logits_col]
        ax.set_title("logits")
        ax.bar(np.arange(x.shape[1]), x[0].detach().cpu().numpy())
        # allow ticks here
        ax.axis("on")
        ax.set_xticks(np.arange(x.shape[1]))
        ax.set_yticks([])
        # set fixed y-axis limits for consistent visualization
        ax.set_ylim(-5, 5)

        # add label text in lower left of the figure
        plt.figtext(
            0.02,
            0.02,
            f"Label: {label} ({label_name})",
            fontsize=14,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8),
        )

        # save plot with date prefix and frame number
        filename = f"{date_prefix}_sample_{batch_idx:02d}_frame_{frame_idx:03d}_label_{label}_{label_name.replace(' ', '_').replace(',', '')}.png"
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(demo_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()  # close figure to free memory

        print(f"saved {filename}")
