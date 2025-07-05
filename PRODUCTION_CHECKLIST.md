# Production ML-Radio Environment Setup Checklist

## Hardware Requirements
- [x] NVIDIA GPU with >= 24GB VRAM (for batch_size=1024)
- [x] >= 32GB System RAM
- [x] Fast NVMe SSD for dataset storage
- [x] Multi-core CPU (>= 8 cores recommended)

## Software Dependencies
### Core ML Stack
- [x] Python >= 3.9
- [x] PyTorch >= 2.0 (with CUDA support)
- [x] torchvision
- [x] torchaudio

### Production Enhancements
- [ ] tensorboard (for logging)
- [ ] wandb (optional: for experiment tracking)
- [ ] torch_tensorrt (for TensorRT optimization)
- [ ] torch-audio (for signal processing)

### Data & ML Tools
- [x] numpy
- [x] scipy
- [x] scikit-learn
- [x] h5py
- [x] pandas
- [ ] matplotlib
- [ ] seaborn

### GNU Radio Stack
- [x] gnuradio
- [x] gnuradio-dev

### System Tools
- [ ] nvidia-ml-py3 (GPU monitoring)
- [ ] psutil (system monitoring)
- [ ] tqdm (progress bars)
- [ ] pyyaml (config files)

## Pre-Training Checklist

### Dataset Preparation
- [ ] Generate large-scale dataset (100K+ samples)
- [ ] Verify dataset integrity with scripts/verify_dataset.py
- [ ] Test data loading speed with production batch sizes
- [ ] Ensure balanced class distribution
- [ ] Validate SNR/JSR ranges are appropriate

### Model Architecture
- [ ] Test model compilation with torch.compile
- [ ] Verify memory usage with target batch size
- [ ] Profile training speed on target hardware
- [ ] Test mixed precision training compatibility

### Infrastructure
- [ ] Setup TensorBoard logging directory
- [ ] Configure model checkpoint storage
- [ ] Setup experiment tracking (wandb/tensorboard)
- [ ] Test multi-GPU training (if applicable)
- [ ] Verify CUDA memory usage patterns

### Monitoring & Debugging
- [ ] Setup GPU memory monitoring
- [ ] Configure training logs
- [ ] Test early stopping mechanisms
- [ ] Verify gradient clipping effectiveness
- [ ] Test checkpoint saving/loading

## Production Dataset Recommendations

### Size Guidelines
- **Small Production**: 50K samples (10K per mod type)
- **Medium Production**: 250K samples (50K per mod type)  
- **Large Production**: 500K+ samples (100K+ per mod type)

### SNR Distribution
```yaml
snr_range: [-10, 30]  # dB
snr_distribution: "uniform"  # or "gaussian" centered at 10dB
```

### Jamming Parameters
```yaml
jam_probability: 0.3  # 30% of samples jammed
jsr_range: [0, 25]   # dB, stronger jamming than current
jam_types: ['tone', 'multi_tone', 'chirp', 'barrage', 'pulse', 'sweep']
```

### Channel Models (Future Enhancement)
```yaml
channel_models:
  - awgn: 0.4      # 40% AWGN
  - rayleigh: 0.3  # 30% Rayleigh fading
  - rician: 0.2    # 20% Rician fading  
  - multipath: 0.1 # 10% multipath
```

## Training Schedule Recommendation

### Phase 1: Baseline Training
- Dataset: 50K samples
- Batch size: 512
- Epochs: 50
- Goal: Establish baseline performance

### Phase 2: Scale-Up Training  
- Dataset: 250K samples
- Batch size: 1024
- Epochs: 100
- Goal: Production-ready model

### Phase 3: Fine-Tuning
- Dataset: Same as Phase 2
- Batch size: 256 (for stability)
- Epochs: 25
- Lower learning rate
- Goal: Optimize final performance

## Expected Training Times

### On RTX 4090 (24GB)
- **50K samples**: ~2-3 hours
- **250K samples**: ~8-12 hours  
- **500K samples**: ~16-24 hours

### On A100/H100 (40-80GB)
- **50K samples**: ~1-2 hours
- **250K samples**: ~4-6 hours
- **500K samples**: ~8-12 hours

## Performance Targets

### Modulation Classification
- **High SNR (>10dB)**: >95% accuracy
- **Medium SNR (0-10dB)**: >85% accuracy
- **Low SNR (-5 to 0dB)**: >70% accuracy

### Jamming Detection
- **High JSR (>10dB)**: >90% detection rate
- **Medium JSR (5-10dB)**: >80% detection rate
- **Low JSR (0-5dB)**: >60% detection rate

### Jamming Classification
- **When jamming detected**: >80% type accuracy
- **Per jamming type**: >70% individual accuracy

## File Organization

```
ML_Radio_Project/
├── datasets/
│   ├── production_dataset.h5 (main dataset)
│   ├── validation_dataset.h5 (holdout test)
│   └── real_world_test.h5 (real RF data)
├── experiments/
│   ├── baseline/
│   ├── production_v1/
│   └── production_v2/
├── models/
│   ├── checkpoints/
│   ├── best_models/
│   └── torchscript/
└── logs/
    ├── tensorboard/
    ├── training_logs/
    └── evaluation_reports/
```

## Final Checks Before Big Machine Deployment

1. **Test on Small Scale First**
   - Run production_train.py with 1K samples locally
   - Verify all features work correctly
   - Check memory usage patterns

2. **Data Transfer Strategy**
   - Plan dataset upload to big machine
   - Consider using rsync or scp with compression
   - Verify dataset integrity after transfer

3. **Remote Access Setup**
   - SSH keys configured
   - Screen/tmux for persistent sessions  
   - Remote monitoring setup

4. **Backup Strategy**
   - Regular model checkpoint backups
   - Log file preservation
   - Results backup plan
