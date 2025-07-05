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

4. **Train Model**
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

5. **Evaluate Model**
   ```bash
   scripts/evaluate.py --data dataset.h5 --model deployments/rfml_cnn.pt \
     --mod-classes deployments/mod_classes.txt --jam-type-classes deployments/jam_type_classes.txt \
     --tta
   ```

6. **Inference / Deployment**
   - TorchScript inference: `src/deployment/inference.py`
   - TensorRT conversion: `src/deployment/tensorrt_builder.py`

## Configuration

- All scripts accept `--help` for detailed usage.
- Config files in `configs/` can be YAML or JSON. Keys map directly to CLI args.

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- h5py
- PyYAML
- scikit-learn
- matplotlib
- tqdm

## Next Steps

- TensorRT builder for optimized inference
- Custom GNU Radio block for live inference
- Experiment tracking (e.g., MLflow, WandB)
