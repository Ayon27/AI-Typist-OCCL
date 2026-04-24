from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"


@dataclass
class TrainConfig:
    model_name: str = "hybrid"
    embed_dim: int = 128
    epochs: int = 100
    batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 1e-3
    patience: int = 10
    margin: float = 1.0
    lam: float = 0.1
    pseudo_anomaly_ratio: float = 1.0
    center_ema_momentum: float = 0.9
    num_workers: int = 4
    ckpt_filename: str = "hybrid_encoder_occl.pt"
    loss_filename: str = "loss_history.json"
    results_filename: str = "results_hybrid.txt"
    processed_dir: Path = field(default=PROCESSED_DIR)
    results_dir: Path = field(default=RESULTS_DIR)
    metrics_dir: Path = field(default=METRICS_DIR)


HYBRID_CONFIG = TrainConfig(
    model_name="hybrid",
    ckpt_filename="hybrid_encoder_occl.pt",
    loss_filename="loss_history.json",
    results_filename="results_hybrid.txt",
)


TYPENET_CONFIG = TrainConfig(
    model_name="typenet",
    ckpt_filename="typenet_occl.pt",
    loss_filename="loss_history_typenet.json",
    results_filename="results_typenet.txt",
)

TSFN_CONFIG = TrainConfig(
    model_name="tsfn",
    ckpt_filename="tsfn_occl.pt",
    loss_filename="loss_history_tsfn.json",
    results_filename="results_tsfn.txt",
)
