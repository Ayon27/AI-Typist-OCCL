import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture

from src.models.hybrid_encoder import HybridEncoder
from src.models.baselines import TypeNet, TSFN
from src.train.shared.config import RESULTS_DIR, PROCESSED_DIR
from src.train.shared.utils import make_loader, extract_embeddings

MODEL_MAP = {"hybrid": HybridEncoder, "typenet": TypeNet, "tsfn": TSFN}

CHECKPOINTS = [
    RESULTS_DIR / "hybrid_encoder_occl.pt",
    RESULTS_DIR / "typenet_occl.pt",
    RESULTS_DIR / "tsfn_occl.pt",
]


def load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt.get("model_name", "hybrid")
    model = MODEL_MAP[name]()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, name


def fit_classifiers(z_train, model_name):
    n = min(20000, len(z_train))
    z_sub = z_train[np.random.choice(len(z_train), n, replace=False)]

    print("Fitting One-Class SVM (nu=0.05, rbf)...")
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(z_sub)
    path = RESULTS_DIR / f"ocsvm_{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(ocsvm, f)
    print(f"OC-SVM saved to {path}")

    print("Fitting OC-GMM (2 components)...")
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(z_sub)
    path = RESULTS_DIR / f"ocgmm_{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(gmm, f)
    print(f"OC-GMM saved to {path}")


def save_all_embeddings(model, model_name, device):
    emb_dir = RESULTS_DIR / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        loader, _ = make_loader(PROCESSED_DIR / f"{split}.pt", batch_size=512)
        print(f"Extracting {split} embeddings...")
        z, y = extract_embeddings(model, loader, device)
        np.savez(emb_dir / f"{split}_{model_name}.npz", embeddings=z, labels=y)
        print(f"  {split}: {z.shape}  (human={int((y==1).sum()):,}, bot={int((y==0).sum()):,})")


def process_checkpoint(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {ckpt_path}")
    model, name = load_model_from_checkpoint(ckpt_path, device)
    print(f"Model: {name}")

    loader, _ = make_loader(PROCESSED_DIR / "train.pt", batch_size=512)
    z_train, _ = extract_embeddings(model, loader, device)
    print(f"Train embeddings: {z_train.shape}")

    fit_classifiers(z_train, name)
    save_all_embeddings(model, name, device)
    print(f"{'='*60}\nOC-SVM / OC-GMM complete for {name}.\n{'='*60}")


def main():
    for ckpt in CHECKPOINTS:
        if ckpt.exists():
            process_checkpoint(ckpt)
        else:
            print(f"Checkpoint not found, skipping: {ckpt}")
