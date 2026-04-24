import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeNet(nn.Module):
    def __init__(self, in_channels=4, hidden_size=64, num_layers=2, embed_dim=128):
        super().__init__()
        self._logged = False
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        z = F.normalize(self.proj(pooled), p=2, dim=1)
        if not self._logged:
            print(f"[TypeNet] Input: {x.shape} -> Output: {z.shape}")
            self._logged = True
        return z


class TSFN(nn.Module):
    def __init__(self, in_channels=4, embed_dim=128):
        super().__init__()
        self._logged = False

        self.temp_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.temp_bn1 = nn.BatchNorm1d(32)
        self.temp_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.temp_bn2 = nn.BatchNorm1d(64)

        self.spec_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.spec_bn1 = nn.BatchNorm1d(32)
        self.spec_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.spec_bn2 = nn.BatchNorm1d(64)

        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        t = F.relu(self.temp_bn1(self.temp_conv1(x)))
        t = F.relu(self.temp_bn2(self.temp_conv2(t)))
        t_pool = t.mean(dim=2)

        x_fft = torch.fft.rfft(x, dim=2).abs()
        x_fft = F.pad(x_fft, (0, x.shape[2] - x_fft.shape[2]))
        s = F.relu(self.spec_bn1(self.spec_conv1(x_fft)))
        s = F.relu(self.spec_bn2(self.spec_conv2(s)))
        s_pool = s.mean(dim=2)

        fused = torch.cat([t_pool, s_pool], dim=1)
        z = F.normalize(self.proj(fused), p=2, dim=1)

        if not self._logged:
            print(f"[TSFN] Input: {x.shape} -> Output: {z.shape}")
            self._logged = True
        return z


if __name__ == "__main__":
    x = torch.randn(8, 4, 50)
    for name, Model in [("TypeNet", TypeNet), ("TSFN", TSFN)]:
        model = Model()
        n = sum(p.numel() for p in model.parameters())
        z = model(x)
        print(f"{name}: params={n:,}  out={z.shape}  L2={z.norm(dim=1).mean():.4f}")
