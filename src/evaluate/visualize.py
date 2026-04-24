import sys
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluate.metrics import compute_eer, get_distance_scores, get_ocsvm_scores, get_gmm_scores

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.figsize": (7, 5),
})
sns.set_style("whitegrid")


_SCORES_CACHE = {}

def _load_scores(z_test, model_name):
    if model_name in _SCORES_CACHE:
        return _SCORES_CACHE[model_name]
        
    scores = {}
    ckpt_name = "hybrid_encoder_occl.pt" if model_name == "hybrid" else f"{model_name}_occl.pt"
    ckpt_path = RESULTS_DIR / ckpt_name
    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scores["distance"] = get_distance_scores(z_test, ckpt["center"].numpy())
    ocsvm_path = RESULTS_DIR / f"ocsvm_{model_name}.pkl"
    if ocsvm_path.exists():
        with open(ocsvm_path, "rb") as f:
            scores["ocsvm"] = get_ocsvm_scores(pickle.load(f), z_test)
    gmm_path = RESULTS_DIR / f"ocgmm_{model_name}.pkl"
    if gmm_path.exists():
        with open(gmm_path, "rb") as f:
            scores["ocgmm"] = get_gmm_scores(pickle.load(f), z_test)
            
    _SCORES_CACHE[model_name] = scores
    return scores


def plot_umap(model_name="hybrid"):
    try:
        import umap
    except ImportError:
        from sklearn.manifold import TSNE
        umap = None

    emb_dir = RESULTS_DIR / "embeddings"
    test_data = np.load(emb_dir / f"test_{model_name}.npz")
    z, y = test_data["embeddings"], test_data["labels"]

    max_points = 10000
    if len(z) > max_points:
        rng = np.random.RandomState(42)
        idx_h = np.where(y == 1)[0]
        idx_b = np.where(y == 0)[0]
        idx = np.concatenate([
            rng.choice(idx_h, min(len(idx_h), max_points // 2), replace=False),
            rng.choice(idx_b, min(len(idx_b), max_points // 2), replace=False),
        ])
        z, y = z[idx], y[idx]

    print(f"UMAP: projecting {len(z):,} points...")
    if umap is not None:
        z2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(z)
    else:
        z2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z)

    fig, ax = plt.subplots(figsize=(8, 6))
    mask_bot, mask_human = y == 0, y == 1
    ax.scatter(z2d[mask_bot, 0], z2d[mask_bot, 1], c="#e74c3c", s=8, alpha=0.4, label="Synthesized Bot", zorder=1)
    ax.scatter(z2d[mask_human, 0], z2d[mask_human, 1], c="#2ecc71", s=8, alpha=0.6, label="Genuine Human", zorder=2)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Latent Space: Human vs Synthesized Bot Embeddings")
    ax.legend(markerscale=3, loc="best")
    out = FIGURES_DIR / f"umap_latent_{model_name}.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def plot_roc(model_name="hybrid"):
    emb_dir = RESULTS_DIR / "embeddings"
    test_data = np.load(emb_dir / f"test_{model_name}.npz")
    z, y = test_data["embeddings"], test_data["labels"]
    scores_dict = _load_scores(z, model_name)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"distance": "#3498db", "ocsvm": "#e67e22", "ocgmm": "#9b59b6"}
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(name, "#333"), linewidth=2,
                label=f"{name.upper()} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set"); ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    out = FIGURES_DIR / f"roc_curve_{model_name}.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def plot_pr(model_name="hybrid"):
    emb_dir = RESULTS_DIR / "embeddings"
    test_data = np.load(emb_dir / f"test_{model_name}.npz")
    z, y = test_data["embeddings"], test_data["labels"]
    scores_dict = _load_scores(z, model_name)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"distance": "#3498db", "ocsvm": "#e67e22", "ocgmm": "#9b59b6"}
    for name, scores in scores_dict.items():
        prec, rec, _ = precision_recall_curve(y, scores)
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=colors.get(name, "#333"), linewidth=2,
                label=f"{name.upper()} (AUPRC={ap:.4f})")
    baseline = y.sum() / len(y)
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.3, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set"); ax.legend(loc="best")
    out = FIGURES_DIR / f"pr_curve_{model_name}.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def plot_confusion(model_name="hybrid"):
    emb_dir = RESULTS_DIR / "embeddings"
    test_data = np.load(emb_dir / f"test_{model_name}.npz")
    z, y = test_data["embeddings"], test_data["labels"]
    scores_dict = _load_scores(z, model_name)

    scores_name = "distance" if "distance" in scores_dict else list(scores_dict.keys())[0]
    scores = scores_dict[scores_name]
    _, eer_thresh = compute_eer(y, scores)
    y_pred = (scores >= eer_thresh).astype(int)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Bot (Pred)", "Human (Pred)"],
                yticklabels=["Bot (True)", "Human (True)"],
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_title(f"Confusion Matrix at EER Threshold ({scores_name.upper()})")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    out = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def plot_loss_curve(model_name="hybrid"):
    loss_path = METRICS_DIR / f"loss_history_{model_name}.json"
    if not loss_path.exists():
        loss_path = METRICS_DIR / "loss_history.json"
    if not loss_path.exists():
        print(f"No loss history found at {loss_path}")
        return

    with open(loss_path) as f:
        loss_data = json.load(f)

    if isinstance(loss_data, dict):
        train_loss = loss_data.get("train", [])
        val_loss = loss_data.get("val", [])
    else:
        train_loss, val_loss = loss_data, []

    epochs = list(range(1, len(train_loss) + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, color="#2c3e50", linewidth=2, marker="o", markersize=3, label="Train Loss")
    if val_loss:
        ax.plot(list(range(1, len(val_loss) + 1)), val_loss, color="#e74c3c", linewidth=2,
                marker="s", markersize=3, label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("OCCL Loss")
    ax.set_title("Training & Validation Loss Curve"); ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    ax.annotate(f"Train: {train_loss[-1]:.4f}", xy=(epochs[-1], train_loss[-1]),
                xytext=(-90, 20), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#2c3e50"), color="#2c3e50")
    out = FIGURES_DIR / f"loss_curve_{model_name}.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def generate_all_figures(model_name="hybrid"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'='*60}\nGenerating figures for model: {model_name}\n{'='*60}")
    plot_loss_curve(model_name)
    plot_umap(model_name)
    plot_roc(model_name)
    plot_pr(model_name)
    plot_confusion(model_name)
    print(f"{'='*60}\nAll figures saved to {FIGURES_DIR}\n{'='*60}")


def plot_model_comparison_roc():
    model_names = ["hybrid", "typenet", "tsfn"]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"hybrid": "#2ecc71", "typenet": "#e74c3c", "tsfn": "#3498db"}
    
    for name in model_names:
        emb_dir = RESULTS_DIR / "embeddings"
        test_data_path = emb_dir / f"test_{name}.npz"
        if not test_data_path.exists():
            continue
            
        test_data = np.load(test_data_path)
        z, y = test_data["embeddings"], test_data["labels"]
        
        # Fast load only the distance score to avoid huge OCGMM evaluation
        import torch
        ckpt_name = "hybrid_encoder_occl.pt" if name == "hybrid" else f"{name}_occl.pt"
        ckpt_path = RESULTS_DIR / ckpt_name
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scores = get_distance_scores(z, ckpt["center"].numpy())
        
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(name, "#333"), linewidth=2,
                label=f"{name.upper()} (AUC={roc_auc:.4f})")
            
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison (Distance Metric)"); ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    out = FIGURES_DIR / "roc_curve_comparison.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def plot_model_comparison_pr():
    model_names = ["hybrid", "typenet", "tsfn"]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"hybrid": "#2ecc71", "typenet": "#e74c3c", "tsfn": "#3498db"}
    
    baseline = None
    for name in model_names:
        emb_dir = RESULTS_DIR / "embeddings"
        test_data_path = emb_dir / f"test_{name}.npz"
        if not test_data_path.exists():
            continue
            
        test_data = np.load(test_data_path)
        z, y = test_data["embeddings"], test_data["labels"]
        
        if baseline is None:
            baseline = y.sum() / len(y)
            
        import torch
        ckpt_name = "hybrid_encoder_occl.pt" if name == "hybrid" else f"{name}_occl.pt"
        ckpt_path = RESULTS_DIR / ckpt_name
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scores = get_distance_scores(z, ckpt["center"].numpy())
        
        prec, rec, _ = precision_recall_curve(y, scores)
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=colors.get(name, "#333"), linewidth=2,
                label=f"{name.upper()} (AUPRC={ap:.4f})")
            
    if baseline is not None:
        ax.axhline(baseline, color="k", linestyle="--", alpha=0.3, label=f"Baseline ({baseline:.3f})")
        
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison (Distance Metric)"); ax.legend(loc="best")
    out = FIGURES_DIR / "pr_curve_comparison.png"
    fig.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def main():
    for model_name in ["hybrid", "typenet", "tsfn"]:
        generate_all_figures(model_name)
    
    print(f"{'='*60}\nGenerating Cross-Model Comparison Plots\n{'='*60}")
    plot_model_comparison_roc()
    plot_model_comparison_pr()
    print("Done.")


if __name__ == "__main__":
    main()
