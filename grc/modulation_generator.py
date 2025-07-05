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
    # TODO: Implement using GNU Radio flowgraph
    raise NotImplementedError("Please generate and implement modulation_generator via GRC.")
