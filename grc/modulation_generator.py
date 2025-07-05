#!/usr/bin/env python3
"""
GRC-Generated Signal Generator Stub

This module should be generated from a GNU Radio Companion flowgraph (e.g., `modulation_generator.grc`) using:

    grcc -d . modulation_generator.grc

and then implement `generate_signal(length, mod_type)` to output a numpy array of complex64 samples.

Template usage:
    from grc.modulation_generator import generate_signal
    signal = generate_signal(length, mod_type)
"""

def generate_signal(length: int, mod_type: str):
    """
    Generate a modulated complex baseband signal of given length and modulation type.

    Parameters:
    - length: number of samples to generate
    - mod_type: string identifier of modulation (e.g., 'BPSK', 'QPSK')

    Returns:
    - numpy.ndarray of shape (length,) dtype=np.complex64
    """
    import numpy as np
    # Define constellation symbols for different modulation types
    if mod_type.upper() == 'BPSK':
        symbols = np.array([1+0j, -1+0j])
    elif mod_type.upper() == 'QPSK':
        symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    elif mod_type.upper() == '8PSK':
        angles = np.arange(8) * 2 * np.pi / 8
        symbols = np.exp(1j * angles)
    elif mod_type.upper() == '16QAM':
        re = np.array([-3, -1, 1, 3])
        im = np.array([-3, -1, 1, 3])
        symbols = np.array([x + 1j*y for x in re for y in im]) / np.sqrt(10)
    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")
    # Randomly choose symbols to fill the requested length
    data = np.random.choice(symbols, size=length)
    return data.astype(np.complex64)
