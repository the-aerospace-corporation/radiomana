import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import HighwayDataModule


def compute_average_psd(dset):
    """compute average for entire dataset and plot it 9 times"""
    dset.dataset.transform = None  # disable any transforms
    avg_psd = None
    counts = None
    for sample, label in dset:
        if label not in [0, 1, 2]:
            # only include "noise" classes
            continue
        if avg_psd is None:
            avg_psd = torch.zeros(3, sample.shape[0], sample.shape[1])
            counts = torch.zeros(3)
        counts[label] += 1
        avg_psd[label] += 10 ** (sample / 10)
    avg_psd /= counts.unsqueeze(1).unsqueeze(2)
    avg_psd_freq = avg_psd.mean(dim=2)  # average along time axis

    avg_psd_freq_db = 10 * torch.log10(avg_psd_freq + 1e-12)
    avg_psd_db = 10 * torch.log10(avg_psd + 1e-12)

    # repeat along time axis to make it plottable
    avg_psd_freq_db_expanded = avg_psd_freq_db.unsqueeze(2).repeat(1, 1, sample.shape[1])

    for idx in range(3):
        print(idx, "Mean PSD:", avg_psd_freq_db[idx].numpy())
    plot9(
        [
            avg_psd_db[0],
            avg_psd_db[1],
            avg_psd_db[2],
            avg_psd_freq_db_expanded[0],
            avg_psd_freq_db_expanded[1],
            avg_psd_freq_db_expanded[2],
            avg_psd_freq_db_expanded[0],
            avg_psd_freq_db_expanded[1],
            avg_psd_freq_db_expanded[2],
        ],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        sample_rate_hz=1,
        class_labels=["Average PSD"] * 8,
    )


def plot16(samples, vmin=-4, vmax=4):
    """Given 16 samples, plot them in a 4x4 grid."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for adx in range(16):
        ax = axes[adx]
        sample = samples[adx]
        ax.imshow(
            sample,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{adx}")
    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    today = datetime.date.today().isoformat()
    plt.savefig(f"{today}_radiomana_16spec.png")
    plt.clf()


def plot9(samples, labels, sample_rate_hz=1, class_labels=None, vmin=-57, vmax=-16):
    """Given 9 samples and labels, plot them in a 3x3 grid."""

    sample_duration_s = samples[0].shape[1] / sample_rate_hz
    freqs = np.linspace(-sample_rate_hz / 2 / 1e6, sample_rate_hz / 2 / 1e6, samples[0].shape[0])
    times = np.linspace(0, sample_duration_s, samples[0].shape[1]) * 1000  # Convert to ms

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for adx in range(9):
        ax = axes[adx]
        sample = samples[adx]
        label = labels[adx]
        if hasattr(sample, "numpy"):
            sample = sample.numpy()
        elif isinstance(sample, list):
            sample = np.array(sample)

        ax.imshow(
            sample,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap="viridis",
            # vmin vmax are 1nd & 99th percentile of whole dset (precomputed)
            vmin=vmin,
            vmax=vmax,
        )

        title = f"{class_labels[label]} ({int(label)})"
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")

        # Format x ticks as whole numbers
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    # Save with today's date YYYY-MM-DD
    today = datetime.date.today().isoformat()
    label_str = "".join(str(int(label)) for label in labels)
    # plt.savefig(f"{today}_radiomana_spec_{label_str}.png")
    # plt.show()


if __name__ == "__main__":
    """
    Show 9 random samples from the Highway2 dataset.
    """
    loader = HighwayDataModule(batch_size=9)
    loader.setup()
    dset = loader.data_train

    samples, labels = next(iter(loader.train_dataloader()))
    plot9(
        samples,
        labels,
        sample_rate_hz=loader.data_test.sample_rate_hz,
        class_labels=loader.data_test.class_labels,
    )

    # compute_average_psd(dset)
