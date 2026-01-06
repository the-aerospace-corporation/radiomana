#!/usr/bin/env python3
"""Benchmark model inference memory, latency, flops, and # parameters."""
import time

import torch
from timm.utils.model import reparameterize_model
from torchinfo import summary

from radiomana import NanoGRU


def print_info(model, input_shape: tuple, verbose: bool = False):
    results = summary(model, input_data=torch.randn(*input_shape), verbose=verbose, device="cpu", depth=2)
    print(f"model: {model.__class__.__name__}")
    print(f"item forward memory: {results.total_output_bytes /1e6:.1f} MB")
    total_memory = results.total_input + results.total_output_bytes + results.total_param_bytes
    print(f"item total memory: {total_memory/1e6:.1f} MB")
    print(f"mult-adds: {results.total_mult_adds/1e9:.3f} G")
    print(f"params: {results.total_params / 1e3:.2f} K")


def bench(model_class, input_shape=(1, 512, 243)):
    """Benchmark for inference memory, latency, flops, and # parameters."""
    model = model_class()
    print_info(model, input_shape=input_shape, verbose=True)
    print("Reparameterizing model...")
    model = reparameterize_model(model)
    model.eval()
    print_info(model, input_shape=input_shape, verbose=True)

    batch_size = 128
    some = torch.randn(batch_size, *input_shape[1:])
    starttime = time.monotonic()
    for device in ["cpu", "cuda"]:
        if device == "cuda":
            if not torch.cuda.is_available():
                print("CUDA not available, skipping GPU benchmark.")
                continue

        runs = 1000 if device == "cuda" else 200
        model = model.to(device)
        some = some.to(device)

        # warmup
        for _ in range(10):
            _ = model(some)

        # cuda benchmarking extras
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.synchronize()

        # actual benchmark
        print(f"benchmark {device}...")
        starttime = time.monotonic()
        for _ in range(runs):
            _ = model(some)

        if device == "cuda":
            torch.cuda.synchronize()

        endtime = time.monotonic()
        latency = (endtime - starttime) / (runs * batch_size)
        print(f"Latency on {device}: {latency*1e3:.3f} ms per item")


if __name__ == "__main__":
    bench(NanoGRU)
