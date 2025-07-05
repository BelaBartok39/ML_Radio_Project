#!/usr/bin/env python3
"""
Python-based GNU Radio Out-Of-Tree block for live modulation classification.
"""
import numpy as np
import torch
from gnuradio import gr

class modulation_classifier(gr.sync_block):
    """
    modulation_classifier: a GNU Radio sync_block that performs live modulation
    classification on incoming complex64 samples using a TorchScript model.

    Parameters:
      model_path: Path to the TorchScript model file
      input_length: Number of samples per inference window
    """
    def __init__(self, model_path, input_length=1024):
        gr.sync_block.__init__(self,
            name="modulation_classifier",
            in_sig=[np.complex64],
            out_sig=None)
        self.input_length = int(input_length)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def work(self, input_items, output_items):
        samples = input_items[0]
        # Process in chunks of input_length
        num_samples = len(samples)
        idx = 0
        while idx + self.input_length <= num_samples:
            window = samples[idx:idx + self.input_length]
            # Convert to tensor [1, 2, L]
            iq = np.stack((np.real(window), np.imag(window)), axis=0)
            tensor = torch.tensor(iq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                # Unpack output tuple or NamedTuple
                if isinstance(out, (tuple, list)):
                    mod_logits = out[0]
                else:
                    mod_logits = out.mod
                pred = int(mod_logits.argmax(dim=1).cpu().item())
            print(f"[ModClassifier] Modulation class: {pred}")
            idx += self.input_length
        return num_samples
