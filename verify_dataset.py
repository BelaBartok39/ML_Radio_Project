import h5py
import numpy as np
import matplotlib.pyplot as plt

def verify_h5_dataset(filename):
    """Comprehensive verification of the 5G jamming dataset"""
    
    with h5py.File(filename, 'r') as f:
        print("="*60)
        print("DATASET VERIFICATION REPORT")
        print("="*60)
        
        # 1. Check structure
        print("\n1. FILE STRUCTURE:")
        for split in ['train', 'val', 'test']:
            if split in f:
                print(f"\n{split}:")
                for key in f[split].keys():
                    dataset = f[split][key]
                    print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # 2. Check metadata
        print("\n2. METADATA:")
        for attr, value in f.attrs.items():
            if isinstance(value, np.ndarray):
                print(f"  - {attr}: {len(value)} items")
            else:
                print(f"  - {attr}: {value}")
        
        # 3. Data statistics
        print("\n3. DATA STATISTICS (from training set):")
        if 'train' in f and len(f['train']['signals']) > 0:
            signals = f['train']['signals'][:]
            snr = f['train']['snr'][:]
            jammed = f['train']['jammed'][:]
            jsr = f['train']['jsr'][:]
            
            print(f"  - Number of examples: {len(signals)}")
            print(f"  - Signal power range: [{np.min(np.abs(signals)**2):.4f}, {np.max(np.abs(signals)**2):.4f}]")
            print(f"  - SNR range: [{np.min(snr):.1f}, {np.max(snr):.1f}] dB")
            print(f"  - Percentage jammed: {np.mean(jammed)*100:.1f}%")
            print(f"  - JSR range (when jammed): [{np.min(jsr[jammed]):.1f}, {np.max(jsr[jammed]):.1f}] dB")
        
        # 4. Modulation distribution
        print("\n4. MODULATION DISTRIBUTION:")
        if 'train' in f and len(f['train']['modulation']) > 0:
            modulations = [m.decode() for m in f['train']['modulation'][:]]
            unique_mods, counts = np.unique(modulations, return_counts=True)
            for mod, count in zip(unique_mods, counts):
                print(f"  - {mod}: {count} examples ({count/len(modulations)*100:.1f}%)")
        
        # 5. Jamming type distribution
        print("\n5. JAMMING TYPE DISTRIBUTION:")
        if 'train' in f and len(f['train']['jamming_type']) > 0:
            jamming_types = [j.decode() for j in f['train']['jamming_type'][:]]
            jammed_mask = f['train']['jammed'][:]
            jammed_types = [j for j, is_jammed in zip(jamming_types, jammed_mask) if is_jammed]
            
            if jammed_types:
                unique_jams, counts = np.unique(jammed_types, return_counts=True)
                for jam, count in zip(unique_jams, counts):
                    print(f"  - {jam}: {count} examples")
        
        # 6. Visualize sample signals
        if 'train' in f and len(f['train']['signals']) >= 4:
            print("\n6. CREATING VISUALIZATION...")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            for idx in range(min(4, len(signals))):
                ax = axes[idx//2, idx%2]
                signal = signals[idx]
                mod = f['train']['modulation'][idx].decode()
                snr_val = f['train']['snr'][idx]
                is_jammed = f['train']['jammed'][idx]
                
                # Time domain
                ax.plot(signal.real[:200], 'b-', alpha=0.7, label='I')
                ax.plot(signal.imag[:200], 'r-', alpha=0.7, label='Q')
                
                title = f"{mod}, SNR={snr_val:.1f}dB"
                if is_jammed:
                    jam_type = f['train']['jamming_type'][idx].decode()
                    jsr_val = f['train']['jsr'][idx]
                    title += f"\nJammed ({jam_type}), JSR={jsr_val:.1f}dB"
                
                ax.set_title(title)
                ax.set_xlabel('Sample')
                ax.set_ylabel('Amplitude')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('dataset_samples.png', dpi=150)
            print("  Saved visualization to 'dataset_samples.png'")
        
        # 7. Data integrity checks
        print("\n7. DATA INTEGRITY CHECKS:")
        checks_passed = True
        
        # Check signal normalization
        if 'train' in f and len(signals) > 0:
            max_amplitudes = np.max(np.abs(signals), axis=1)
            if np.all(max_amplitudes <= 1.1):  # Allow small margin
                print("  ✓ Signals are properly normalized")
            else:
                print("  ✗ Some signals exceed normalization bounds")
                checks_passed = False
        
        # Check SNR range
        if 'train' in f and len(snr) > 0:
            if np.all((snr >= -20) & (snr <= 40)):
                print("  ✓ SNR values are in reasonable range")
            else:
                print("  ✗ Some SNR values are out of expected range")
                checks_passed = False
        
        # Check jamming consistency
        if 'train' in f:
            jammed_mask = f['train']['jammed'][:]
            jsr_vals = f['train']['jsr'][:]
            if np.all((jsr_vals > 0) == jammed_mask):
                print("  ✓ JSR values consistent with jamming status")
            else:
                print("  ✗ JSR values inconsistent with jamming status")
                checks_passed = False
        
        print("\n" + "="*60)
        if checks_passed:
            print("VERDICT: Dataset appears to be VALID and well-formed! ✓")
        else:
            print("VERDICT: Some issues detected, please review.")
        print("="*60)

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    verify_h5_dataset("gnuradio_jamming_dataset.h5")
    
    # Quick example of loading for ML
    print("\nExample: Loading a batch for ML training")
    with h5py.File("gnuradio_jamming_dataset.h5", 'r') as f:
        # Load first 4 examples
        batch_size = min(4, len(f['train']['signals']))
        signals = f['train']['signals'][:batch_size]
        labels = f['train']['modulation'][:batch_size]
        
        # Convert to ML format (I/Q separation)
        X = np.stack([signals.real, signals.imag], axis=-1)
        print(f"Input shape for ML: {X.shape}")
        print(f"Labels: {[l.decode() for l in labels]}")