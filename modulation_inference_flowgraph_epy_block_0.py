import numpy as np
import torch
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self,
            name="LiveClassifier",
            in_sig=[np.complex64],
            out_sig=None)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load('/home/jackthelion83/ML_Radio_Project/multitask_cnn.pt', map_location=self.device)
        self.model.eval()

    def work(self, input_items, output_items):
        in0 = input_items[0]

        if len(in0) >= 1024:
            # Step 1: Take first 1024 samples
            signal = in0[:1024]

            # Step 2: Convert complex to two real-valued channels (I/Q)
            iq = np.stack((np.real(signal), np.imag(signal)), axis=0)  # shape: [2, 1024]

            # Step 3: Move to tensor
            iq_tensor = torch.tensor(iq, dtype=torch.float32).unsqueeze(0)  # [1, 2, 1024]

            # Step 4: Move input to same device as model
            iq_tensor = iq_tensor.to(self.device)

            # Step 5: Inference
            with torch.no_grad():
                output = self.model(iq_tensor)

                # Assuming output is a NamedTuple or dict with 'mod' key for modulation logits
                if isinstance(output, dict) or hasattr(output, 'mod'):
                    mod_logits = output['mod'] if isinstance(output, dict) else output.mod
                else:
                    # If output is tuple/list, modulation output is first element
                    mod_logits = output[0]

                pred = mod_logits.argmax(dim=1).item()  # get scalar predicted class index

            print(f"🧠 Detected Modulation Class: {pred}")

        return len(in0)

