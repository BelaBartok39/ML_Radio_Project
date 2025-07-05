#!/usr/bin/env python3
"""
Production GNU Radio RF Dataset Generator
- Uses actual GNU Radio signal chains for modulation
- Configurable via command line and YAML config files
- Realistic jamming simulation with proper JSR
- Balanced dataset generation with progress tracking
- ML-ready HDF5 output format
"""

import numpy as np
import h5py
from gnuradio import gr, analog, digital, blocks
import random
from tqdm import tqdm
import argparse
import yaml
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SAMPLE_RATE = 1e6
SAMPLES_PER_EXAMPLE = 1024
MODS = [
    ('bpsk', digital.constellation_bpsk().base()),
    ('qpsk', digital.constellation_qpsk().base()),
    ('8psk', digital.constellation_8psk().base()),
    ('qam16', digital.constellation_rect(
        constell=[
            complex(i, q)
            for i in [-3, -1, 1, 3]
            for q in [-3, -1, 1, 3]
        ],
        pre_diff_code=[],
        rotational_symmetry=4,
        real_sectors=4,
        imag_sectors=4,
        width_real_sectors=2.0,
        width_imag_sectors=2.0,
        normalization=2  # AMPLITUDE_NORMALIZATION
    ).base()),
    ('qam64', digital.constellation_rect(
        constell=[
            complex(i, q)
            for i in [-7, -5, -3, -1, 1, 3, 5, 7]
            for q in [-7, -5, -3, -1, 1, 3, 5, 7]
        ],
        pre_diff_code=[],
        rotational_symmetry=4,
        real_sectors=8,
        imag_sectors=8,
        width_real_sectors=2.0,
        width_imag_sectors=2.0,
        normalization=2  # AMPLITUDE_NORMALIZATION
    ).base())
]



JAMMING_TYPES = ['tone', 'multi_tone', 'chirp', 'barrage', 'pulse', 'sweep']

class SignalChain(gr.top_block):
    def __init__(self, constellation, snr_db):
        super().__init__()
        self.constellation = constellation
        self.snr_db = snr_db

        self.src = analog.random_uniform_source_b(0, 256, 0)
        self.packed = blocks.packed_to_unpacked_bb(1, gr.GR_LSB_FIRST)
        self.mod = digital.generic_mod(
            constellation=self.constellation,
            differential=True,
            samples_per_symbol=4,
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False
        )
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, int(SAMPLE_RATE), True)
        self.head = blocks.head(gr.sizeof_gr_complex, SAMPLES_PER_EXAMPLE)
        self.vector_sink = blocks.vector_sink_c()

        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, self.noise_amp(snr_db), 0)
        self.adder = blocks.add_cc()

        self.connect(self.src, self.packed, self.mod, self.throttle)
        self.connect((self.throttle, 0), (self.adder, 0))
        self.connect((self.noise, 0), (self.adder, 1))
        self.connect(self.adder, self.head, self.vector_sink)

    def noise_amp(self, snr_db):
        snr_linear = 10**(snr_db / 10)
        return 1.0 / np.sqrt(2 * snr_linear)

    def get_signal(self):
        self.run()
        return np.array(self.vector_sink.data(), dtype=np.complex64)

def generate_jamming(jtype):
    t = np.arange(SAMPLES_PER_EXAMPLE) / SAMPLE_RATE
    if jtype == 'tone':
        freq = np.random.uniform(1000, 10000)
        return np.exp(1j * 2 * np.pi * freq * t)
    elif jtype == 'multi_tone':
        signal = np.zeros_like(t, dtype=complex)
        for _ in range(np.random.randint(2, 6)):
            f = np.random.uniform(1000, 10000)
            signal += np.exp(1j * 2 * np.pi * f * t)
        return signal / np.abs(signal).max()
    elif jtype == 'chirp':
        f0, f1 = 1000, 50000
        k = (f1 - f0) / (SAMPLES_PER_EXAMPLE / SAMPLE_RATE)
        return np.exp(1j * 2 * np.pi * (f0 * t + 0.5 * k * t ** 2))
    elif jtype == 'barrage':
        return (np.random.randn(SAMPLES_PER_EXAMPLE) + 1j * np.random.randn(SAMPLES_PER_EXAMPLE)) / np.sqrt(2)
    elif jtype == 'pulse':
        signal = np.zeros(SAMPLES_PER_EXAMPLE, dtype=complex)
        for _ in range(np.random.randint(1, 4)):
            start = np.random.randint(0, SAMPLES_PER_EXAMPLE - 100)
            signal[start:start + 100] = (np.random.randn(100) + 1j * np.random.randn(100))
        return signal
    elif jtype == 'sweep':
        sweep_rate = 1e7
        return np.exp(1j * 2 * np.pi * sweep_rate * t ** 2)
    else:
        return np.zeros(SAMPLES_PER_EXAMPLE, dtype=complex)

def generate_dataset(config):
    """Generate RF dataset based on configuration dictionary"""
    filename = config.get('output_file', 'gnuradio_jamming_dataset.h5')
    num_samples = config.get('num_samples', 10000)
    val_ratio = config.get('val_ratio', 0.1)
    test_ratio = config.get('test_ratio', 0.1)
    jam_prob = config.get('jam_prob', 0.3)
    snr_range = config.get('snr_range', [0, 30])
    jsr_range = config.get('jsr_range', [0.1, 20])
    
    logging.info(f"Generating dataset: {filename}")
    logging.info(f"Total samples: {num_samples}")
    logging.info(f"Train/Val/Test ratios: {1-val_ratio-test_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}")
    logging.info(f"Jamming probability: {jam_prob:.1f}")
    logging.info(f"SNR range: {snr_range} dB")
    logging.info(f"JSR range: {jsr_range} dB")
    
    # Calculate counts
    n_val = int(num_samples * val_ratio)
    n_test = int(num_samples * test_ratio)
    n_train = num_samples - n_val - n_test

    # Create splits dict for convenience
    splits = {
        'train': n_train,
        'val': n_val,
        'test': n_test,
    }

    with h5py.File(filename, 'w') as f:
        # Store configuration as attributes
        f.attrs['sample_rate'] = SAMPLE_RATE
        f.attrs['samples_per_example'] = SAMPLES_PER_EXAMPLE
        f.attrs['num_samples'] = num_samples
        f.attrs['jam_prob'] = jam_prob
        f.attrs['snr_range'] = snr_range
        f.attrs['jsr_range'] = jsr_range
        f.attrs['modulation_types'] = [m[0] for m in MODS]
        f.attrs['jamming_types'] = JAMMING_TYPES

        for split, count in splits.items():
            if count == 0:
                continue
            grp = f.create_group(split)
            grp.create_dataset('signals', (count, SAMPLES_PER_EXAMPLE), dtype=np.complex64)
            grp.create_dataset('modulation', (count,), dtype='S10')
            grp.create_dataset('jammed', (count,), dtype=bool)
            grp.create_dataset('snr', (count,), dtype=np.float32)
            grp.create_dataset('jamming_type', (count,), dtype='S10')
            grp.create_dataset('jsr', (count,), dtype=np.float32)

        # Distribute samples evenly by modulation
        mod_names = [m[0] for m in MODS]
        samples_per_mod = num_samples // len(MODS)
        remaining_samples = num_samples % len(MODS)

        idx_tracker = {'train': 0, 'val': 0, 'test': 0}

        for mod_idx, (modname, constel) in enumerate(MODS):
            # Add extra samples to first few modulations if num_samples not divisible
            mod_samples = samples_per_mod + (1 if mod_idx < remaining_samples else 0)
            
            logging.info(f"Generating {mod_samples} samples for {modname}")
            
            for sample_idx in tqdm(range(mod_samples), desc=f"Generating {modname}"):
                # Generate SNR within specified range
                snr = np.random.uniform(snr_range[0], snr_range[1])
                tb = SignalChain(constel, snr)
                sig = tb.get_signal()

                # Determine if jammed
                jammed = np.random.rand() < jam_prob
                jtype = b'none'
                jsr = 0.0

                if jammed:
                    jtype_str = random.choice(JAMMING_TYPES)
                    jtype = jtype_str.encode()
                    sig_power = np.mean(np.abs(sig) ** 2)
                    jsr_db = np.random.uniform(jsr_range[0], jsr_range[1])
                    jsr = jsr_db
                    jammer = generate_jamming(jtype_str)
                    jammer_power = sig_power * 10 ** (jsr_db / 10)
                    jammer *= np.sqrt(jammer_power / (np.mean(np.abs(jammer) ** 2) + 1e-10))
                    sig += jammer
                else:
                    jsr = 0.0
                    jtype = b'none'

                # Use RMS normalization to preserve JSR relationships
                # instead of peak normalization which destroys jamming effects
                rms_power = np.sqrt(np.mean(np.abs(sig) ** 2))
                sig /= (rms_power + 1e-12)

                # Assign sample to split using round-robin to maintain balance
                if idx_tracker['train'] < n_train:
                    split = 'train'
                elif idx_tracker['val'] < n_val:
                    split = 'val'
                elif idx_tracker['test'] < n_test:
                    split = 'test'
                else:
                    # Fallback to train if all splits are full (shouldn't happen)
                    split = 'train'

                i = idx_tracker[split]

                # Write to datasets
                f[split]['signals'][i] = sig
                f[split]['modulation'][i] = modname.encode()
                f[split]['jammed'][i] = jammed
                f[split]['snr'][i] = snr
                f[split]['jamming_type'][i] = jtype
                f[split]['jsr'][i] = jsr

                idx_tracker[split] += 1

    logging.info(f"Dataset generation complete!")
    logging.info(f"Final split sizes: Train={idx_tracker['train']}, Val={idx_tracker['val']}, Test={idx_tracker['test']}")
    
    # Verify the dataset
    with h5py.File(filename, 'r') as f:
        total_generated = sum(f[split]['signals'].shape[0] for split in f.keys())
        logging.info(f"Verification: Generated {total_generated} total samples")
        
        # Show jamming distribution
        for split in f.keys():
            jammed_count = np.sum(f[split]['jammed'][:])
            total_count = len(f[split]['jammed'])
            jam_percentage = 100 * jammed_count / total_count if total_count > 0 else 0
            logging.info(f"{split.capitalize()} split: {jammed_count}/{total_count} jammed ({jam_percentage:.1f}%)")

def load_config(config_path):
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            raw_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            raw_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Normalize hyphenated keys to snake_case
    normalized_config = {}
    for key, value in raw_config.items():
        normalized_key = key.replace('-', '_')
        normalized_config[normalized_key] = value
    
    return normalized_config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Production GNU Radio RF Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main options
    parser.add_argument('--output', '-o', type=str, default='gnuradio_jamming_dataset.h5',
                       help='Output HDF5 dataset file')
    parser.add_argument('--config', '-c', type=str,
                       help='YAML/JSON configuration file')
    
    # Dataset size parameters
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Total number of samples to generate')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test split ratio')
    
    # Signal parameters
    parser.add_argument('--sample-rate', type=float, default=1e6,
                       help='Sample rate in Hz')
    parser.add_argument('--samples-per-example', type=int, default=1024,
                       help='Number of samples per signal example')
    
    # SNR parameters
    parser.add_argument('--snr-min', type=float, default=0,
                       help='Minimum SNR in dB')
    parser.add_argument('--snr-max', type=float, default=30,
                       help='Maximum SNR in dB')
    
    # Jamming parameters
    parser.add_argument('--jam-prob', type=float, default=0.3,
                       help='Probability of jamming (0.0 to 1.0)')
    parser.add_argument('--jsr-min', type=float, default=0.1,
                       help='Minimum JSR in dB')
    parser.add_argument('--jsr-max', type=float, default=20,
                       help='Maximum JSR in dB')
    
    # Control options
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce logging output')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Overwrite existing output file')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        logging.info(f"Random seed set to: {args.seed}")
    
    # Load configuration
    config = {}
    if args.config:
        logging.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    
    # Get default values from argument parser for comparison
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default='gnuradio_jamming_dataset.h5')
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--sample-rate', type=float, default=1e6)
    parser.add_argument('--samples-per-example', type=int, default=1024)
    parser.add_argument('--snr-min', type=float, default=0)
    parser.add_argument('--snr-max', type=float, default=30)
    parser.add_argument('--jam-prob', type=float, default=0.3)
    parser.add_argument('--jsr-min', type=float, default=0.1)
    parser.add_argument('--jsr-max', type=float, default=20)
    defaults = parser.parse_args([])
    
    # Only override config with CLI args that differ from defaults
    if args.output != defaults.output:
        config['output_file'] = args.output
    elif 'output' in config:
        config['output_file'] = config.pop('output')
    else:
        config['output_file'] = args.output
        
    if args.num_samples != defaults.num_samples:
        config['num_samples'] = args.num_samples
    elif 'num_samples' not in config:
        config['num_samples'] = args.num_samples
        
    if args.val_ratio != defaults.val_ratio:
        config['val_ratio'] = args.val_ratio
    elif 'val_ratio' not in config:
        config['val_ratio'] = args.val_ratio
        
    if args.test_ratio != defaults.test_ratio:
        config['test_ratio'] = args.test_ratio
    elif 'test_ratio' not in config:
        config['test_ratio'] = args.test_ratio
        
    # Handle SNR range
    if args.snr_min != defaults.snr_min or args.snr_max != defaults.snr_max:
        config['snr_range'] = [args.snr_min, args.snr_max]
    elif 'snr_range' not in config:
        config['snr_range'] = [args.snr_min, args.snr_max]
        
    if args.jam_prob != defaults.jam_prob:
        config['jam_prob'] = args.jam_prob
    elif 'jam_prob' not in config:
        config['jam_prob'] = args.jam_prob
        
    # Handle JSR range
    if args.jsr_min != defaults.jsr_min or args.jsr_max != defaults.jsr_max:
        config['jsr_range'] = [args.jsr_min, args.jsr_max]
    elif 'jsr_range' not in config:
        config['jsr_range'] = [args.jsr_min, args.jsr_max]
    
    # Handle sample rate and samples per example
    if args.sample_rate != defaults.sample_rate:
        config['sample_rate'] = args.sample_rate
    elif 'sample_rate' not in config:
        config['sample_rate'] = args.sample_rate
        
    if args.samples_per_example != defaults.samples_per_example:
        config['samples_per_example'] = args.samples_per_example
    elif 'samples_per_example' not in config:
        config['samples_per_example'] = args.samples_per_example
    
    # Handle seed from config if not provided via CLI
    if args.seed is None and 'random_seed' in config:
        np.random.seed(config['random_seed'])
        random.seed(config['random_seed'])
        logging.info(f"Random seed set from config: {config['random_seed']}")
    
    # Handle force option from config
    force_overwrite = args.force or config.get('force_overwrite', False)
    
    # Update global constants if specified in config
    global SAMPLE_RATE, SAMPLES_PER_EXAMPLE
    SAMPLE_RATE = config['sample_rate']
    SAMPLES_PER_EXAMPLE = config['samples_per_example']
    
    # Check if output file exists
    output_path = Path(config['output_file'])
    if output_path.exists() and not force_overwrite:
        response = input(f"Output file '{output_path}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            logging.info("Generation cancelled.")
            return
    
    # Validate configuration
    if config['val_ratio'] + config['test_ratio'] >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")
    
    if config['jam_prob'] < 0 or config['jam_prob'] > 1:
        raise ValueError("jam_prob must be between 0.0 and 1.0")
    
    if config['snr_range'][0] >= config['snr_range'][1]:
        raise ValueError("SNR min must be less than SNR max")
    
    if config['jsr_range'][0] >= config['jsr_range'][1]:
        raise ValueError("JSR min must be less than JSR max")
    
    # Display configuration summary
    logging.info("=== Dataset Generation Configuration ===")
    logging.info(f"Output file: {config['output_file']}")
    logging.info(f"Total samples: {config['num_samples']:,}")
    logging.info(f"Modulation types: {[m[0] for m in MODS]}")
    logging.info(f"Jamming types: {JAMMING_TYPES}")
    logging.info(f"Sample rate: {SAMPLE_RATE:,.0f} Hz")
    logging.info(f"Samples per example: {SAMPLES_PER_EXAMPLE}")
    logging.info(f"SNR range: {config['snr_range']} dB")
    logging.info(f"Jamming probability: {config['jam_prob']:.1%}")
    logging.info(f"JSR range: {config['jsr_range']} dB")
    logging.info("========================================")
    
    # Generate the dataset
    try:
        generate_dataset(config)
        logging.info(f"Dataset successfully saved to: {config['output_file']}")
    except Exception as e:
        logging.error(f"Dataset generation failed: {e}")
        raise

if __name__ == '__main__':
    main()