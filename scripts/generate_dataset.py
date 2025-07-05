#!/usr/bin/env python3
"""
Signal Generation Toolkit (Stage 3)
Parametrized RF signal dataset generator with optional jamming.
"""
# Add project root to sys.path for module imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import h5py
import numpy as np
import subprocess  # for GNU Radio flowgraph invocation
from tqdm import tqdm

MOD_TEMPLATES = ['BPSK', 'QPSK', '8PSK', '16QAM']
JAM_TYPES = ['tone', 'noise', 'sweep']


def grc_generate_signal(length, mod_type):
    """
    Generate a modulated signal via the GRC-generated Python module.
    Dynamically loads grc/modulation_generator.py by path.
    """
    import importlib.util
    # Determine project root and module path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    module_path = os.path.join(project_root, 'grc', 'modulation_generator.py')
    if not os.path.isfile(module_path):
        raise RuntimeError(f"GRC module not found at {module_path}")
    spec = importlib.util.spec_from_file_location('modulation_generator', module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'generate_signal'):
        raise RuntimeError("generate_signal() not implemented in modulation_generator.py")
    return mod.generate_signal(length, mod_type)

def generate_sample(sample_length, mod_type, jam_prob, jam_type_list, use_grc=False):
    if use_grc:
        # Use real RF modulation from GNURadio flowgraph
        signal = grc_generate_signal(sample_length, mod_type)
    else:
        # Placeholder: random complex noise signal
        signal = np.random.randn(sample_length) + 1j * np.random.randn(sample_length)
    # Assign labels
    mod_label = mod_type
    jammed = np.random.rand() < jam_prob
    jam_label = int(jammed)
    if jammed:
        jam_type = np.random.choice(jam_type_list)
        # Add simple jamming: add stronger noise
        signal += (np.random.randn(sample_length) + 1j * np.random.randn(sample_length)) * 2
    else:
        jam_type = ''
    return signal, mod_label, jam_label, jam_type


def main():
    parser = argparse.ArgumentParser(description="Generate RF dataset with optional jamming.")
    parser.add_argument('--output', type=str, default='dataset.h5', help='Output HDF5 file')
    parser.add_argument('--samples', type=int, default=1000, help='Total number of samples')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train split fraction')
    parser.add_argument('--mod-types', nargs='+', default=MOD_TEMPLATES, help='List of modulation types')
    parser.add_argument('--jam-prob', type=float, default=0.2, help='Probability of jamming per sample')
    parser.add_argument('--jam-types', nargs='+', default=JAM_TYPES, help='List of jamming types')
    parser.add_argument('--length', type=int, default=1024, help='Samples per signal')
    parser.add_argument('--use-grc', action='store_true', help='Use GNURadio flowgraphs for real RF signals')
    args = parser.parse_args()

    # If using GRC, attempt to compile flowgraph to Python module
    if args.use_grc:
        grc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'grc', 'modulation_generator.grc'))
        if os.path.isfile(grc_path):
            try:
                print(f"Compiling GRC flowgraph: {grc_path}")
                subprocess.run(['grcc', '-d', os.path.dirname(grc_path), grc_path], check=True)
            except Exception as e:
                print(f"Warning: failed to compile GRC flowgraph: {e}")
        else:
            print(f"GRC flowgraph not found at {grc_path}, proceeding with stub generator.")

    n_train = int(args.samples * args.train_split)
    n_val = args.samples - n_train

    with h5py.File(args.output, 'w') as f:
        for split, n in [('train', n_train), ('val', n_val)]:
            grp = f.create_group(split)
            signals_ds = grp.create_dataset('signals', shape=(n, args.length), dtype=np.complex64)
            mod_ds = grp.create_dataset('modulation', shape=(n,), dtype='S10')
            jam_ds = grp.create_dataset('jammed', shape=(n,), dtype=np.int8)
            jamt_ds = grp.create_dataset('jamming_type', shape=(n,), dtype='S10')

            for i in tqdm(range(n), desc=f"Generating {split}"):
                mod_type = np.random.choice(args.mod_types)
                signal, mod_label, jam_label, jam_type = generate_sample(
                    args.length, mod_type, args.jam_prob, args.jam_types, args.use_grc
                )
                signals_ds[i] = signal.astype(np.complex64)
                mod_ds[i] = mod_label.encode()
                jam_ds[i] = jam_label
                jamt_ds[i] = jam_type.encode()

    print(f"Dataset written to {args.output}")


if __name__ == '__main__':
    main()
