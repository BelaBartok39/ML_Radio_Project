#!/usr/bin/env python3
"""
Benchmark RFML inference on Jetson or CPU/GPU.
Measures average latency and throughput for a TorchScript or TensorRT model.
"""
import argparse
import time
import torch
import numpy as np
import os
import sys

# add project root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.modulation_jamming_dataset import ModulationJammingDataset


def load_model(args, device):
    if args.trt_engine:
        from torch2trt import TRTModule
        engine = TRTModule()
        engine.load_state_dict(torch.load(args.trt_engine, map_location=device))
        model = engine
    else:
        model = torch.jit.load(args.model, map_location=device)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description='Benchmark RFML inference')
    parser.add_argument('--model', type=str, help='TorchScript model file')
    parser.add_argument('--trt-engine', type=str, help='TensorRT engine file')
    parser.add_argument('--input-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--runs', type=int, default=100, help='Number of inference runs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    print(f"Benchmarking on device: {device}")

    model = load_model(args, device)
    # random input for benchmarking
    input_tensor = torch.randn(args.batch_size, 2, args.input_length, device=device)

    # warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    # timed runs
    times = []
    for _ in range(args.runs):
        start = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    print(f"Average inference time per batch: {avg_time*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} batches/sec")

if __name__ == '__main__':
    main()
