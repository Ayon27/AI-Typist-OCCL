import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluate.metrics import evaluate_all_models
from src.evaluate.visualize import generate_all_figures

def main():
    print("=" * 70)
    print(" Running Full Evaluation Pipeline")
    print("=" * 70)
    
    # Generate numerical metrics and comparisons
    results = evaluate_all_models()
    
    # Generate plots for models that were evaluated
    for model_name in results.keys():
        generate_all_figures(model_name)

if __name__ == "__main__":
    main()
