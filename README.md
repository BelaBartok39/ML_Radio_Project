# RFML Research and Deployment Framework

This repository provides a modular stack for RF Machine Learning (RFML) from dataset generation to live inference.

## Project Structure

- **configs/**: Configuration files (YAML/JSON) for training and evaluation
- **grc/**: GNU Radio Companion flowgraph templates and companion scripts
- **scripts/**: Utility scripts for dataset generation, project scaffolding, and evaluation
- **src/data/**: HDF5 dataset wrapper (`ModulationJammingDataset`) with streaming support
- **src/models/**: Model architecture (e.g., `MultiTaskCNN`)
- **src/training/**: Training entrypoint with CLI and config support
- **src/deployment/**: Inference scripts and TorchScript/TensorRT conversion utilities

## Quickstart

1. **Setup**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Generate Scaffold** (optional)
   ```bash
   scripts/generate_scaffold.py --root .
   ```

3. **Generate Dataset**
   ```bash
   scripts/generate_dataset.py --output dataset.h5 --samples 5000 --length 1024
   ```
   # Optional: use GNURadio flowgraphs for real RF signals
   ```bash
   scripts/generate_dataset.py --output dataset_grc.h5 --samples 5000 --length 1024 --use-grc
   ```
4. **Verify Dataset**
   ```bash
   scripts/verify_dataset.py --data dataset.h5
   ```

5. **Train Model**
   - Using CLI args:
     ```bash
     python3 src/training/train.py --data dataset.h5 --epochs 50 --batch_size 128 --lr 1e-3 \
       --dropout 0.3 --input_length 1024 --output_dir deployments --model_name rfml_cnn
     ```

   - Using a YAML config (`configs/train.yaml`):
     ```yaml
     data: dataset.h5
     epochs: 50
     batch_size: 128
     lr: 0.001
     dropout: 0.3
     input_length: 1024
     output_dir: deployments
     model_name: rfml_cnn
     ```
     ```bash
     python3 src/training/train.py --config configs/train.yaml
     ```

6. **Evaluate Model**
   ```bash
   scripts/evaluate.py --data dataset.h5 --model deployments/rfml_cnn.pt \
     --mod-classes deployments/mod_classes.txt --jam-type-classes deployments/jam_type_classes.txt \
     --tta
   ```

7. **Inference / Deployment**
   - TorchScript inference: `src/deployment/inference.py`
   - TensorRT builder: `src/deployment/tensorrt_builder.py`
   - Build TensorRT engine:
     ```bash
     python3 src/deployment/tensorrt_builder.py \
       --model deployments/rfml_cnn.pt \
       --engine deployments/rfml_cnn_trt.pth \
       --input-length 1024 --batch-size 1 --fp16
     ```
   - Jetson-optimized inference (TorchScript or TRT):
     ```bash
     python3 src/deployment/inference_jetson.py \
       --data dataset.h5 --model deployments/rfml_cnn.pt \
       --trt-engine deployments/rfml_cnn_trt.pth --input-length 1024
     ```

## Stage 5: Live GNU Radio Integration & Benchmarking

### 5.1 Build & Install OOT Block
```bash
# From project root
git clone /path/to/this/repo grc/python_mod_classifier
cd grc/python_mod_classifier
mkdir build && cd build
cmake ..
make
sudo make install
sudo ldconfig
```
- This installs the `modulation_classifier` block into GNU Radio.

### 5.2 Use Block in GRC Flowgraph
1. Open GNU Radio Companion.
2. Add the `Python Module` block or search for `modulation_classifier`.
3. Set `model_path` to your TorchScript model (e.g., `deployments/rfml_cnn.pt`).
4. Run the flowgraph to see live classification prints.

### 5.3 Benchmarking on Jetson
```bash
# Build TensorRT engine if desired
python3 src/deployment/tensorrt_builder.py \
  --model deployments/rfml_cnn.pt \
  --engine deployments/rfml_cnn_trt.pth --input-length 1024 --batch-size 1 --fp16

# Run benchmark
scripts/benchmark_jetson.py --model deployments/rfml_cnn.pt \
  --trt-engine deployments/rfml_cnn_trt.pth --input-length 1024 --batch-size 1 --runs 200 --device cuda
```

## Configuration

- All scripts accept `--help` for detailed usage.
- Config files in `configs/` can be YAML or JSON. Keys map directly to CLI args.

## Production Pipeline (Large-Scale)
For large-scale dataset generation and model training on powerful GPU clusters or multi-GPU machines, we provide a separate production pipeline with optimized configurations.

### Production Parameters
Below are key parameters defined in `configs/production_train.yaml`:
- **data**: Path to the HDF5 dataset (e.g., `gnuradio_jamming_dataset.h5`).
- **val_split**, **test_split**: Names of validation and test splits in the HDF5 file.
- **input_length**: Number of samples per example (default: 1024).
- **dropout**: Dropout probability in the CNN.
- **model_name**: Output model name prefix (e.g., `rfml_production_cnn`).
- **epochs**: Total training epochs (e.g., 100).
- **batch_size**: Large batch size (e.g., 1024).
- **lr**, **min_lr**: Initial and minimum learning rates (e.g., 0.001 and 0.000001).
- **weight_decay**: Optimizer weight decay (e.g., 0.0001).
- **optimizer**: Optimizer type (e.g., `adamw`).
- **lr_scheduler**, **warmup_epochs**: Scheduler type (e.g., `cosine`) and warmup.
- **mod_loss_weight**, **jam_detection_weight**, **jam_type_weight**: Multi-task loss weights.
- **num_workers**, **pin_memory**, **prefetch_factor**, **dataloader_persistent_workers**: DataLoader optimizations.
- **use_amp**, **grad_clip_norm**: Mixed precision and gradient clipping.
- **torch_compile**: Enable PyTorch 2.0 compilation.
- **save_every_n_epochs**, **early_stopping_patience**, **best_metric**: Checkpointing and early stopping.
- **output_dir**, **tensorboard_log_dir**, **log_level**: Output and logging directories.
- **device**: Target device (e.g., `cuda`).

### Production Workflow
1. **Generate large HDF5 dataset**
   ```bash
   python scripts/gnu_dataset_generator.py --config configs/dataset_generation.yaml
   ```
   This uses actual GNU Radio flowgraphs for realistic signals.

2. **Train model**
   ```bash
   python src/training/production_train.py --config configs/production_train.yaml
   ```
   The configuration file includes optimized settings for high-throughput training and mixed precision.

3. **Evaluate and save TorchScript**
   At the end of training, the script will automatically validate on the test split, save TorchScript (`.pt`), PyTorch checkpoints, and class mappings in `output_dir`.

4. **Deploy**
   - **TorchScript inference**: `src/deployment/inference.py`
   - **TensorRT conversion**: `src/deployment/tensorrt_builder.py`

### Local Testing vs Production
- **Local Testing Workflow**: Use smaller dataset sizes and `scripts/generate_dataset.py` with `src/training/train.py` for quick experiments on your laptop.
- **Production Pipeline**: Use `scripts/gnu_dataset_generator.py` and `src/training/production_train.py` with `configs/production_train.yaml` for large-scale runs on GPUs or clusters.

## Next Steps
- TensorRT builder for optimized inference
- Custom GNU Radio block for live inference
- Experiment tracking (e.g., MLflow, WandB)
