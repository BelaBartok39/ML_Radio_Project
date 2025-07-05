import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

class MultiTaskOutput(NamedTuple):
    mod: torch.Tensor
    jam: torch.Tensor
    jam_type: torch.Tensor

class MultiTaskCNN(nn.Module):
    """
    CNN with shared backbone and multi-head outputs for:
      - modulation classification
      - binary jamming detection
      - jamming type classification
    """
    def __init__(self, num_mod_classes: int, num_jam_types: int, input_length: int = 1024, dropout: float = 0.3):
        super(MultiTaskCNN, self).__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv1d(2, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        # Compute flattened feature size after pooling layers
        pools = 3
        out_len = input_length // (2 ** pools)
        self.flatten_size = 128 * out_len
        self.fc1 = nn.Linear(self.flatten_size, 256)

        self.classifier_mod = nn.Linear(256, num_mod_classes)
        self.classifier_jam = nn.Linear(256, 2)
        self.classifier_jam_type = nn.Linear(256, num_jam_types)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x: torch.Tensor) -> MultiTaskOutput:
        feat = self.features(x)
        return MultiTaskOutput(
            mod=self.classifier_mod(feat),
            jam=self.classifier_jam(feat),
            jam_type=self.classifier_jam_type(feat)
        )
