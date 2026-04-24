import sys
import pickle
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, auc, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"


def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def compute_auprc(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))


def compute_all_metrics(y_true, scores, threshold=None):
    eer, eer_thresh = compute_eer(y_true, scores)
    if threshold is None:
        threshold = eer_thresh
    y_pred = (scores >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        "EER": eer,
        "EER_threshold": threshold,
        "AUPRC": compute_auprc(y_true, scores),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(auc(fpr, tpr)),
    }


def get_distance_scores(embeddings, center):
    dists = np.sqrt(np.sum((embeddings - center) ** 2, axis=1))
    return -dists


def get_ocsvm_scores(ocsvm, embeddings):
    return ocsvm.decision_function(embeddings)


def get_gmm_scores(gmm, embeddings):
    return gmm.score_samples(embeddings)


def _load_scores(z_test, model_name):
    scores = {}

    ckpt_name = "hybrid_encoder_occl.pt" if model_name == "hybrid" else f"{model_name}_occl.pt"
    ckpt_path = RESULTS_DIR / ckpt_name
    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        center = ckpt["center"].numpy()
        scores["distance"] = get_distance_scores(z_test, center)

    ocsvm_path = RESULTS_DIR / f"ocsvm_{model_name}.pkl"
    if ocsvm_path.exists():
        with open(ocsvm_path, "rb") as f:
            scores["ocsvm"] = get_ocsvm_scores(pickle.load(f), z_test)

    gmm_path = RESULTS_DIR / f"ocgmm_{model_name}.pkl"
    if gmm_path.exists():
        with open(gmm_path, "rb") as f:
            scores["ocgmm"] = get_gmm_scores(pickle.load(f), z_test)

    return scores


def evaluate_model(model_name="hybrid"):
    emb_dir = RESULTS_DIR / "embeddings"
    test_path = emb_dir / f"test_{model_name}.npz"
    if not test_path.exists():
        print(f"Test embeddings not found: {test_path}")
        print("Run `python -m src.train.main` first.")
        return {}

    test_data = np.load(test_path)
    z_test, y_test = test_data["embeddings"], test_data["labels"]
    print(f"Test set: {z_test.shape[0]:,} samples "
          f"(human={int((y_test==1).sum()):,}, bot={int((y_test==0).sum()):,})")

    ckpt_name = "hybrid_encoder_occl.pt" if model_name == "hybrid" else f"{model_name}_occl.pt"
    ckpt_path = RESULTS_DIR / ckpt_name
    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_params = ckpt.get("n_params", "N/A")
        print(f"Model params:  {n_params:,}" if isinstance(n_params, int) else f"Model params: {n_params}")

    results = {}
    scores_dict = _load_scores(z_test, model_name)

    for clf_name, scores in scores_dict.items():
        metrics = compute_all_metrics(y_test, scores)
        results[clf_name] = metrics
        print(f"[{clf_name.upper():>8s}]  EER={metrics['EER']:.4f}  "
              f"F1={metrics['F1_Score']:.4f}  AUC={metrics['ROC_AUC']:.4f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["=" * 70, f"EVALUATION REPORT — {model_name.upper()}", "=" * 70,
             f"Test samples: {z_test.shape[0]:,}  "
             f"(human={int((y_test==1).sum()):,}, bot={int((y_test==0).sum()):,})", ""]
    for clf_name, m in results.items():
        lines.append(f"--- {clf_name.upper()} ---")
        for k, v in m.items():
            lines.append(f"  {k:<20s}: {v:.6f}")
        lines.append("")
    lines.append("=" * 70)
    report_text = "\n".join(lines)

    with open(METRICS_DIR / "evaluation_report.txt", "w") as f:
        f.write(report_text)
    print("\n" + report_text)
    print(f"Report saved to {METRICS_DIR / 'evaluation_report.txt'}")

    with open(METRICS_DIR / f"evaluation_{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_all_models():
    model_names = ["hybrid", "typenet", "tsfn"]
    emb_dir = RESULTS_DIR / "embeddings"
    all_results = {}

    for model_name in model_names:
        test_path = emb_dir / f"test_{model_name}.npz"
        if not test_path.exists():
            print(f"Skipping {model_name}: embeddings not found at {test_path}")
            continue
        print(f"\n{'='*60}\nEvaluating: {model_name.upper()}\n{'='*60}")
        results = evaluate_model(model_name)
        if results:
            all_results[model_name] = results

    if len(all_results) < 2:
        print("Need at least 2 models for comparison.")
        return all_results

    metrics_to_show = ["EER", "ROC_AUC", "AUPRC", "F1_Score", "Accuracy"]
    clf_type = "distance"

    header = f"{'Metric':<15s}"
    for m in all_results:
        header += f"  {m.upper():>12s}"
    sep = "-" * len(header)

    lines = ["\n" + "=" * 70, "MODEL COMPARISON (Distance-based scoring)", "=" * 70, header, sep]
    for metric in metrics_to_show:
        row = f"{metric:<15s}"
        for m in all_results:
            val = all_results[m].get(clf_type, {}).get(metric, float("nan"))
            row += f"  {val:>12.4f}"
        lines.append(row)
    lines.append(sep)
    lines.append("=" * 70)

    comparison_text = "\n".join(lines)
    print(comparison_text)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "comparison_report.txt", "w") as f:
        f.write(comparison_text)
    with open(METRICS_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Comparison saved to {METRICS_DIR / 'comparison_report.txt'}")

    return all_results


def main():
    evaluate_all_models()


if __name__ == "__main__":
    main()
