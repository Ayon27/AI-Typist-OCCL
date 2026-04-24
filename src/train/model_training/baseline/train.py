import torch
from src.models.baselines import TypeNet, TSFN
from src.train.shared.config import TYPENET_CONFIG, TSFN_CONFIG
from src.train.shared.trainer import Trainer
from src.train.shared.utils import run_with_logging

TYPENET_RESULTS = TYPENET_CONFIG.results_dir / TYPENET_CONFIG.results_filename
TSFN_RESULTS = TSFN_CONFIG.results_dir / TSFN_CONFIG.results_filename


def train_typenet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TypeNet(embed_dim=TYPENET_CONFIG.embed_dim).to(device)
    Trainer(model, TYPENET_CONFIG).run()


def train_tsfn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSFN(embed_dim=TSFN_CONFIG.embed_dim).to(device)
    Trainer(model, TSFN_CONFIG).run()


if __name__ == "__main__":
    run_with_logging(train_typenet, TYPENET_RESULTS)
    run_with_logging(train_tsfn, TSFN_RESULTS)
