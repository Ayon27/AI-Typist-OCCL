import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train.shared.config import HYBRID_CONFIG, TYPENET_CONFIG, TSFN_CONFIG
from src.train.shared.utils import run_with_logging
from src.train.model_training.hybrid.train import train_hybrid
from src.train.model_training.baseline.train import train_typenet, train_tsfn
from src.train.model_training.ocsvm import main as fit_classifiers

PIPELINE = [
    ("HybridEncoder", train_hybrid, HYBRID_CONFIG),
    ("TypeNet", train_typenet, TYPENET_CONFIG),
    ("TSFN", train_tsfn, TSFN_CONFIG),
]


if __name__ == "__main__":
    for name, train_fn, cfg in PIPELINE:
        print(f"\n{'='*70}\n Training {name}\n{'='*70}\n")
        run_with_logging(train_fn, cfg.results_dir / cfg.results_filename)

    print(f"\n{'='*70}\n Fitting OC-SVM / OC-GMM\n{'='*70}\n")
    fit_classifiers()
 