from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

class KeystrokeDynamicsDataset(Dataset):
    def __init__(
        self,
        pt_path: str | Path,
        pseudo_anomaly_ratio: float = 1.0,
        transform=None,
    ):
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.data: torch.Tensor = payload["data"]      # (N, 4, 50)
        self.labels: torch.Tensor = payload["labels"]   # (N,)
        self.transform = transform
        self.pseudo_anomaly_ratio = pseudo_anomaly_ratio

        # Determine if this is a training split (all labels == 1 ⇒ pure human)
        self.is_train = bool((self.labels == 1).all())

        if self.is_train:
            n_pseudo = int(len(self.data) * self.pseudo_anomaly_ratio)
            self._pseudo_indices = torch.randperm(len(self.data))[:n_pseudo]
            print(
                f"INFO: Loaded {len(self.data):,} human samples.  "
                f"Will generate {n_pseudo:,} pseudo-anomalies on the fly."
            )
        else:
            self._pseudo_indices = torch.empty(0, dtype=torch.long)
            n_human = int((self.labels == 1).sum())
            n_bot = int((self.labels == 0).sum())
            print(
                f"INFO: Loaded {len(self.data):,} samples  "
                f"(human={n_human:,}, bot={n_bot:,})."
            )

    def __len__(self) -> int:
        return len(self.data) + len(self._pseudo_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        n_real = len(self.data)

        if idx < n_real:
            x = self.data[idx].clone()
            y = int(self.labels[idx].item())
        else:
            # Generate a pseudo-anomaly from a real human sample
            src_idx = self._pseudo_indices[idx - n_real]
            x = self._corrupt(self.data[src_idx].clone())
            y = 0   # pseudo-anomaly label

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a 'Generative Mixup' corruption to create a hard pseudo-anomaly.
        Strategies:
            0. Generative Mixup (LLM Mimic) - interpolates two distinct humans
            1. Subtle Micro-Jitter
            2. Local Segment Swap
        """
        strategy = torch.randint(0, 3, (1,)).item()
        x = x.clone()

        if strategy == 0:
            # 0. Generative Mixup (LLM Mimic)
            # Sample another random human sequence
            idx_rand = torch.randint(0, len(self.data), (1,)).item()
            x_rand = self.data[idx_rand].clone()
            
            # Interpolate
            alpha = 0.3 + torch.rand(1).item() * 0.4  # random alpha in [0.3, 0.7]
            x = alpha * x + (1.0 - alpha) * x_rand

        elif strategy == 1:
            # 1. Subtle Micro-Jitter
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        else:
            # 2. Local Segment Swap
            seq_len = x.shape[1]
            seg_len = seq_len // 4
            idx_a = torch.randint(0, seq_len - seg_len, (1,)).item()
            idx_b = torch.randint(0, seq_len - seg_len, (1,)).item()
            tmp = x[:, idx_a:idx_a + seg_len].clone()
            x[:, idx_a:idx_a + seg_len] = x[:, idx_b:idx_b + seg_len]
            x[:, idx_b:idx_b + seg_len] = tmp

        return x
