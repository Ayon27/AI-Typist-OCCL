import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import KeystrokeDynamicsDataset
from src.train.shared.tee import Tee


def run_with_logging(main_fn, results_path):
    tee = Tee(results_path)
    sys.stdout = tee
    try:
        main_fn()
    finally:
        tee.close()
    print(f"Results saved to: {results_path}")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_loader(path, batch_size, pseudo_anomaly_ratio=0.0, shuffle=False,
                num_workers=4, drop_last=False):
    ds = KeystrokeDynamicsDataset(path, pseudo_anomaly_ratio=pseudo_anomaly_ratio)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last), len(ds)


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_z, all_y = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            z = model(x_batch.to(device))
            all_z.append(z.cpu().numpy())
            all_y.append(y_batch.numpy())
    return np.concatenate(all_z), np.concatenate(all_y)
