import matplotlib.pyplot as plt
import numpy as np
import torch

from .datasets import HighwayDataModule


def plot_per_class_average(dset):
    """compute the average PSD per class, per (freq, time) bin, over the entire dataset and plot each class"""
    dset.dataset.transform = None  # disable any transforms
    class_labels = dset.dataset.class_labels
    num_classes = len(class_labels)
    avg_psd = None
    counts = None
    for sample, label in dset:
        if avg_psd is None:
            avg_psd = torch.zeros(num_classes, sample.shape[0], sample.shape[1])
            counts = torch.zeros(num_classes)
        counts[label] += 1
        avg_psd[label] += 10 ** (sample / 10)
    counts = counts.clamp(min=1)  # avoid div-by-zero for classes with no samples
    avg_psd /= counts.unsqueeze(1).unsqueeze(2)  # per-bin average, elementwise over (freq, time)

    avg_psd_db = 10 * torch.log10(avg_psd + 1e-12)

    # exactly 9 classes, so this maps 1:1 onto the 3x3 grid
    samples = [avg_psd_db[idx] for idx in range(num_classes)]
    labels = list(range(num_classes))
    plot9(
        samples,
        labels,
        sample_rate_hz=1,
        class_labels=class_labels,
    )


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

    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.1)


if __name__ == "__main__":
    """
    Show Mean PSD per class for the Highway2 dataset.
    """
    loader = HighwayDataModule(batch_size=9)
    loader.setup()
    dset = loader.data_train

    plot_per_class_average(dset)
    plt.show()
