#!/usr/bin/env python3
"""
Signal Generation Toolkit (Stage 3)
Parametrized RF signal dataset generator with optional jamming.
"""
import argparse
import h5py
import numpy as np
from tqdm import tqdm

MOD_TEMPLATES = ['BPSK', 'QPSK', '8PSK', '16QAM']
JAM_TYPES = ['tone', 'noise', 'sweep']


def generate_sample(sample_length, mod_type, jam_prob, jam_type_list):
    # Placeholder: random complex noise signal
    signal = np.random.randn(sample_length) + 1j * np.random.randn(sample_length)
    # Simulate simple modulation placeholder
    # TODO: integrate real modulation via GNU Radio flowgraph
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
    args = parser.parse_args()

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
                    args.length, mod_type, args.jam_prob, args.jam_types
                )
                signals_ds[i] = signal.astype(np.complex64)
                mod_ds[i] = mod_label.encode()
                jam_ds[i] = jam_label
                jamt_ds[i] = jam_type.encode()

    print(f"Dataset written to {args.output}")


if __name__ == '__main__':
    main()
