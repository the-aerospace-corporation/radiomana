#!/usr/bin/env python3
"""Benchmark model inference memory, latency, flops, and # parameters."""

import time

import torch
from torchinfo import summary

from radiomana import HighwayBaselineModel


def bench(model_class, input_shape=(512, 243), batch_size=128, warmup_runs=10):
    """Benchmark for inference memory, latency, flops, and # parameters."""
    model = model_class().eval()

    # torchinfo's summary already prints memory/flops/param info when verbose=True
    print(f"model: {model.__class__.__name__}")
    summary(model, input_data=torch.randn(1, *input_shape), verbose=True, device="cpu", depth=3)

    inputs = torch.randn(batch_size, *input_shape)
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            print("\nCUDA not available, skipping GPU benchmark.")
            continue

        print(f"\nbenchmark {device}...")
        model = model.to(device)
        inputs = inputs.to(device)

        # cuda benchmarking extras
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # warmup
        for _ in range(warmup_runs):
            _ = model(inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        # actual benchmark
        runs = 1000 if device == "cuda" else 10
        starttime = time.monotonic()
        for _ in range(runs):
            _ = model(inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        endtime = time.monotonic()

        latency = (endtime - starttime) / (runs * batch_size)
        print(f"latency = {latency * 1e3:.3f} ms per item")


if __name__ == "__main__":
    bench(HighwayBaselineModel)
