#!/usr/bin/env python3
"""
Jetson-Optimized Inference Script (Stage 4)
Loads a TensorRT engine or TorchScript model for low-latency inference on RF signals.
"""
# Add project root to PYTHONPATH for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import numpy as np
import torch
from pathlib import Path
from src.data.modulation_jamming_dataset import ModulationJammingDataset

def load_trt_engine(engine_path, input_shape, use_fp16=False):
    try:
        from torch2trt import TRTModule
    except ImportError:
        raise ImportError("torch2trt not installed; cannot load TRT engine.")
    engine = TRTModule()
    engine.load_state_dict(torch.load(engine_path))
    return engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with TorchScript or TensorRT on Jetson')
    parser.add_argument('--data', type=str, required=True, help='HDF5 dataset file')
    parser.add_argument('--split', type=str, default='val', choices=['train','val'], help='Dataset split')
    parser.add_argument('--model', type=str, required=True, help='TorchScript model file')
    parser.add_argument('--trt-engine', type=str, help='TensorRT engine state_dict file')
    parser.add_argument('--input-length', type=int, default=1024, help='Samples per signal')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 for TRT')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if args.trt_engine:
        print(f"Loading TensorRT engine from {args.trt_engine}")
        model = load_trt_engine(args.trt_engine, (args.batch_size, 2, args.input_length), args.fp16)
    else:
        print(f"Loading TorchScript model from {args.model}")
        model = torch.jit.load(args.model)
    model.to(device).eval()

    # Load one batch
    ds = ModulationJammingDataset(args.data, split=args.split)
    x, mod, jam, jam_type = ds[0]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        # unpack
        if isinstance(output, tuple) or isinstance(output, list):
            out_mod, out_jam, out_jam_type = output
        else:
            out_mod, out_jam, out_jam_type = output.mod, output.jam, output.jam_type
        pred_mod = out_mod.argmax(dim=1).cpu().item()
        pred_jam = out_jam.argmax(dim=1).cpu().item()
        pred_jam_type = out_jam_type.argmax(dim=1).cpu().item() if pred_jam==1 else None

    print(f"Predicted modulation class index: {pred_mod}")
    print(f"Predicted jam/detection index: {pred_jam}")
    if pred_jam == 1:
        print(f"Predicted jamming type class index: {pred_jam_type}")

    ds.close()
