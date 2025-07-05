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
try:
    import torch
except ImportError:
    torch = None
    st.sidebar.error("PyTorch is not installed. Install it with: pip install torch")
import matplotlib.pyplot as plt

st.set_page_config(page_title="RFML Dashboard", layout="wide")
st.title("RFML Dataset Viewer & Classifier")

# Sidebar inputs
dataset_path = st.sidebar.text_input("Dataset file (HDF5)", "dataset_big_test.h5")
model_path = st.sidebar.text_input("TorchScript model file", "/home/jackthelion83/ML_Radio_Project/deployment/multitask_cnn.pt")
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
    idx = st.sidebar.slider("Sample index", 0, n_samples-1, 0)

    # Display true labels
    signal = f[split]['signals'][idx]
    mod_label = f[split]['modulation'][idx].astype(str)
    jam_label = bool(f[split]['jammed'][idx])
    st.subheader(f"Sample {idx} ({split})")
    st.write(f"**True modulation:** {mod_label}")
    st.write(f"**Jammed:** {jam_label}")

    # Plot I/Q time-domain
    t = np.arange(len(signal))
    fig, axs = plt.subplots(2, 1, figsize=(8, 4))
    axs[0].plot(t, signal.real, label='I')
    axs[0].plot(t, signal.imag, label='Q')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Time Domain (I/Q)')
    axs[1].magnitude_spectrum(signal, Fs=1.0)
    axs[1].set_title('Frequency Spectrum')
    st.pyplot(fig)

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
                iq = np.stack((signal.real, signal.imag), axis=0)
                tensor = torch.tensor(iq, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(tensor)
                    # unpack mod logits
                    if isinstance(out, (tuple, list)):
                        mod_logits = out[0]
                    elif hasattr(out, 'mod'):
                        mod_logits = out.mod
                    else:
                        mod_logits = out
                    probs = torch.softmax(mod_logits, dim=1).cpu().numpy().flatten()
                st.write("**Prediction probabilities:**")
                st.bar_chart(probs)
                pred = int(np.argmax(probs))
                st.write(f"**Predicted class index:** {pred}")

# Synthetic data mix explanation
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Note:** Samples are randomly jammed based on `jam_prob` (default 0.2)."
)
st.sidebar.markdown(
    "If you observe all samples jammed in a small run, adjust `--jam-prob` or increase the sample count."
)
