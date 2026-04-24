import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridEncoder(nn.Module):
    """
    Hybrid CNN + LSTM Encoder strictly following the original architectural prompt:
    - Conv Block 1: (4 -> 32) + MaxPool(2)
    - Conv Block 2: (32 -> 64) + MaxPool(2)
    - LSTM Block: (64 -> 128)
    """
    def __init__(self, in_channels=4, embed_dim=128):
        super().__init__()
        self._is_first = True
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        # Recurrent Block
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)

    def forward(self, x):
        # Input x: (B, 4, 50)
        
        # Conv Block 1 -> (B, 32, 25)
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        
        # Conv Block 2 -> (B, 64, 12)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.pool2(h)
        
        # Recurrent Block
        # Permute to (B, SequenceLength, Channels) -> (B, 12, 64)
        h = h.permute(0, 2, 1)
        
        out, (hn, cn) = self.lstm(h)
        
        # Extract last hidden state -> Shape: (B, 128)
        # hn shape is (1, B, 128) because it's a 1-layer unidirectional LSTM
        z = hn[-1]
        
        z = F.normalize(z, p=2, dim=1)

        if self._is_first:
            print(f"[{self.__class__.__name__}] Input: {x.shape} -> Output: {z.shape}")
            self._is_first = False

        return z
