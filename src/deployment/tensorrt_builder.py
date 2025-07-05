#!/usr/bin/env python3
"""
TensorRT Engine Builder (Stage 4)
Builds a TensorRT-accelerated model using torch2trt.

Requirements:
  pip install torch2trt
"""
import argparse
import torch
try:
    from torch2trt import torch2trt
except ImportError:
    raise ImportError(
        "torch2trt not found. Install with `pip install torch2trt` or build from source."
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build TensorRT engine from TorchScript model')
    parser.add_argument('--model', type=str, required=True, help='Path to TorchScript .pt model')
    parser.add_argument('--engine', type=str, required=True, help='Output path for TRT engine (.pth)')
    parser.add_argument('--batch-size', type=int, default=1, help='Maximum batch size')
    parser.add_argument('--input-channels', type=int, default=2, help='Number of input channels')
    parser.add_argument('--input-length', type=int, default=1024, help='Length of input signal')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    args = parser.parse_args()

    # Load TorchScript model
    model = torch.jit.load(args.model).eval().cuda()
    # Create example input
    example_input = torch.randn(args.batch_size, args.input_channels, args.input_length).cuda()

    # Convert to TensorRT engine
    print(f"Building TensorRT engine with batch size {args.batch_size}, fp16={args.fp16}...")
    model_trt = torch2trt(model, [example_input], max_batch_size=args.batch_size, fp16_mode=args.fp16)

    # Save TRT engine state
    torch.save(model_trt.state_dict(), args.engine)
    print(f"Saved TensorRT engine to {args.engine}")
