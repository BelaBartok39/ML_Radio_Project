#!/usr/bin/env python3
"""
Streamlit-based RFML Dataset & Model Explorer
"""
import sys
try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed. Install it with: pip install streamlit")
    sys.exit(1)

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import torch
except ImportError:
    torch = None
    st.sidebar.error("PyTorch is not installed. Install it with: pip install torch")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.modulation_jamming_dataset import ModulationJammingDataset
import torch.nn.functional as F

st.set_page_config(page_title="RFML Dashboard", layout="wide")
st.title("RFML Dataset Viewer & Classifier")

# Sidebar inputs
dataset_path = st.sidebar.text_input("Dataset file (HDF5)", "gnuradio_jamming_dataset.h5")
model_path   = st.sidebar.text_input("TorchScript model file", "/home/jackthelion83/ML_Radio_Project/deployments/rfml_cnn_test.pt")
input_length = st.sidebar.number_input("Input length", value=1024, step=1)

# Load dataset
if dataset_path:
    try:
        f = h5py.File(dataset_path, 'r')
    except Exception as e:
        st.error(f"Failed to open dataset: {e}")
        st.stop()

    splits = list(f.keys())
    split = st.sidebar.selectbox("Data split", splits)
    n_samples = f[split]['signals'].shape[0]
    
    if n_samples == 0:
        st.warning(f"No samples found in the '{split}' split of the dataset.")
        f.close()
        st.stop()
    else:
        idx = st.sidebar.slider("Sample index", 0, n_samples - 1, 0)

        # Display true labels
        signal    = f[split]['signals'][idx]
        mod_label = f[split]['modulation'][idx].astype(str)
        jam_label = bool(f[split]['jammed'][idx])
        jam_type_label = f[split]['jamming_type'][idx].astype(str) if 'jamming_type' in f[split] else 'N/A'
        jsr_value = f[split]['jsr'][idx] if 'jsr' in f[split] else 0.0
        snr_value = f[split]['snr'][idx] if 'snr' in f[split] else 0.0
        
        st.subheader(f"Sample {idx} ({split})")
        st.write(f"**True modulation:** {mod_label}")
        st.write(f"**Jammed:** {jam_label}")
        st.write(f"**True jamming type:** {jam_type_label}")
        if jam_label:
            st.write(f"**JSR (Jammer-to-Signal Ratio):** {jsr_value:.1f} dB")
        st.write(f"**SNR:** {snr_value:.1f} dB")

        # Prepare class mappings for prediction labels
        ds = ModulationJammingDataset(dataset_path, split=split)
        mod_classes      = ds.mod_classes
        jam_type_classes = ds.jam_type_classes
        jam_classes      = ['Clean', 'Jammed']

        # Signal diagnostics
        signal_power = np.mean(np.abs(signal)**2)
        signal_rms = np.sqrt(signal_power)
        st.write(f"**Signal RMS:** {signal_rms:.6f}")
        st.write(f"**Signal Power:** {signal_power:.6f}")
        st.write(f"**I range:** [{signal.real.min():.3f}, {signal.real.max():.3f}]")
        st.write(f"**Q range:** [{signal.imag.min():.3f}, {signal.imag.max():.3f}]")

        # Plot I/Q time-domain and spectrum
        t = np.arange(len(signal))
        fig, axs = plt.subplots(2, 1, figsize=(8, 4))
        axs[0].plot(t, signal.real, label='I')
        axs[0].plot(t, signal.imag, label='Q')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Time Domain (I/Q)')
        axs[1].magnitude_spectrum(signal, Fs=1.0)
        axs[1].set_title('Frequency Spectrum')
        st.pyplot(fig)

        # Enhanced Constellation Diagram
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Raw constellation
        ax2.scatter(signal.real, signal.imag, alpha=0.3, s=10)
        ax2.set_title("Raw Constellation")
        ax2.set_ylabel("Quadrature")
        ax2.set_xlabel("In-phase")
        ax2.grid(True)
        ax2.axis('equal')
        
        # Downsampled constellation (every 4th sample to approximate symbol rate)
        # This assumes 4 samples per symbol as used in GNU Radio
        downsampled = signal[::4]  
        ax3.scatter(downsampled.real, downsampled.imag, alpha=0.6, s=15)
        ax3.set_title("Downsampled Constellation (4x)")
        ax3.set_ylabel("Quadrature")
        ax3.set_xlabel("In-phase")
        ax3.grid(True)
        ax3.axis('equal')
        
        st.pyplot(fig2)

        # Run model inference if provided
        if model_path:
            st.subheader("Model Inference")
            if torch is None:
                st.error("Inference disabled: PyTorch not available")
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                try:
                    model = torch.jit.load(model_path, map_location=device)
                    model.eval()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                else:
                    # Prepare input tensor
                    iq = np.stack((signal.real, signal.imag), axis=0)
                    tensor = torch.tensor(iq, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out = model(tensor)
                        # Unpack multi-task outputs
                        if isinstance(out, (tuple, list)) and len(out) == 3:
                            mod_logits, jam_logits, jamtype_logits = out
                        elif isinstance(out, dict):
                            mod_logits     = out.get('mod')
                            jam_logits     = out.get('jam')
                            jamtype_logits = out.get('jam_type')
                        else:
                            mod_logits     = out
                            jam_logits     = None
                            jamtype_logits = None

                        # Compute softmax probabilities
                        mod_probs = F.softmax(mod_logits, dim=1).cpu().numpy().flatten()
                        jam_probs = None
                        jt_probs  = None
                        if jam_logits is not None:
                            jam_probs = F.softmax(jam_logits, dim=1).cpu().numpy().flatten()
                        if jamtype_logits is not None and jam_probs is not None and jam_probs.argmax() == 1:
                            jt_probs = F.softmax(jamtype_logits, dim=1).cpu().numpy().flatten()

                        # Map to human-readable predictions
                        mod_pred     = mod_classes[int(mod_probs.argmax())]
                        jam_pred     = jam_classes[int(jam_probs.argmax())] if jam_probs is not None else 'N/A'
                        jamtype_pred = jam_type_classes[int(jt_probs.argmax())] if jt_probs is not None else 'N/A'

                    # Display results
                    st.write("**Predicted Modulation:**", mod_pred)
                    st.bar_chart(mod_probs)

                    st.write("**Jamming Detected?**", jam_pred)
                    if jam_probs is not None:
                        st.bar_chart(jam_probs)

                    st.write("**Jamming Type:**", jamtype_pred)
                    if jt_probs is not None:
                        st.bar_chart(jt_probs)

# Synthetic data mix explanation
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Note:** Samples are randomly jammed based on `jam_prob` (default 0.2)."
)
st.sidebar.markdown(
    "If you observe all samples jammed in a small run, adjust `--jam-prob` or increase the sample count."
)