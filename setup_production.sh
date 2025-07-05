#!/bin/bash
# Production ML-Radio Setup Script for Big GPU Machine
cd /project/ndrdmond

srun -c 1 --mem=10G --gres=gpu:2 -t 1-00:00:00 -p igpuq --pty bash

module load nvhpc/23.11
module load python/3.10.13/gcc.8.5.0
module load cuda/12.3
module load cudnn/8.9.7.29

set -e  # Exit on any error

echo "🚀 Setting up ML-Radio Production Environment..."

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install CUDA first."
    exit 1
fi

echo "✅ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create conda environment
ENV_NAME="rfml_production"
echo "📦 Creating conda environment: $ENV_NAME"

if conda env list | grep -q $ENV_NAME; then
    echo "Environment $ENV_NAME already exists. Activating..."
    source activate $ENV_NAME
else
    conda create -n $ENV_NAME python=3.10 -y
    source activate $ENV_NAME
fi

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install GNU Radio (system level)
echo "📡 Installing GNU Radio..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y gnuradio gnuradio-dev
elif command -v yum &> /dev/null; then
    sudo yum install -y gnuradio gnuradio-devel
elif command -v conda &> /dev/null; then
    conda install -c conda-forge gnuradio -y
else
    echo "⚠️  Please install GNU Radio manually for your system"
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p datasets experiments models/checkpoints models/best_models models/torchscript logs/tensorboard logs/training_logs logs/evaluation_reports

# Test PyTorch CUDA
echo "🧪 Testing PyTorch CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

# Test basic imports
echo "📊 Testing critical imports..."
python3 -c "
try:
    import torch.nn.functional as F
    import torchvision
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import yaml
    import tensorboard
    print('✅ All critical imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Check GPU memory for training
echo "💾 Checking GPU memory for production training..."
python3 -c "
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    # Test memory allocation for large batch
    try:
        test_tensor = torch.randn(1024, 2, 1024, device=device)  # Simulate batch_size=1024
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f'✅ GPU memory test passed. Peak usage: {memory_gb:.2f} GB')
        print(f'💡 Recommended batch size for your GPU: {int(16 * 1024 / memory_gb)}')
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print('⚠️  GPU memory insufficient for batch_size=1024. Consider smaller batch size.')
else:
    print('❌ CUDA not available')
"

echo "🎯 Quick performance benchmark..."
python3 -c "
import torch
import time
if torch.cuda.is_available():
    device = torch.device('cuda')
    # Simple benchmark
    x = torch.randn(512, 2, 1024, device=device)
    conv = torch.nn.Conv1d(2, 64, 7).to(device)
    
    # Warmup
    for _ in range(10):
        _ = conv(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = conv(x)
    torch.cuda.synchronize()
    end = time.time()
    
    throughput = 512 * 100 / (end - start)
    print(f'🏃 GPU throughput: {throughput:.0f} samples/sec')
    print(f'📈 Estimated training time for 250K samples: {250000 / throughput / 3600:.1f} hours')
"

echo ""
echo "🎉 Production environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: conda activate $ENV_NAME"
echo "2. Generate production dataset: python3 scripts/gnu_dataset_generator.py"
echo "3. Run training: python3 src/training/production_train.py --config configs/production_train.yaml"
echo "4. Monitor with TensorBoard: tensorboard --logdir logs/tensorboard"
echo ""
echo "💡 Pro tips:"
echo "- Use screen/tmux for long training sessions"
echo "- Monitor GPU usage: watch -n 1 nvidia-smi"
echo "- Check logs: tail -f training.log"
