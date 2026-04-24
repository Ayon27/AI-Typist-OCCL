import torch
from src.models.hybrid_encoder import HybridEncoder
from src.train.shared.config import HYBRID_CONFIG
from src.train.shared.trainer import Trainer
from src.train.shared.utils import run_with_logging

CFG = HYBRID_CONFIG
RESULTS_TXT = CFG.results_dir / CFG.results_filename

def train_hybrid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridEncoder(embed_dim=CFG.embed_dim).to(device)
    Trainer(model, CFG).run()

if __name__ == "__main__":
    run_with_logging(train_hybrid, RESULTS_TXT)
