import argparse
import json
import sys
from pathlib import Path

from skeval.classifier import SentenceClassifier
from skeval.dataset.loader import DatasetLoader
from skeval.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Sentinel AI Sentence Classifier"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help="Directory containing the trained model",
    )
    parser.add_argument(
        "--data", required=True, type=str, help="Path to test data (.csv or .jsonl)"
    )
    parser.add_argument(
        "--text-col", required=True, type=str, help="Column name containing text"
    )
    parser.add_argument(
        "--label-col",
        required=True,
        type=str,
        help="Column name containing ground truth labels",
    )
    parser.add_argument(
        "--output", type=str, help="Path to save evaluation JSON results (optional)"
    )

    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists() or not (model_path / "model.pt").exists():
        print(f"Error: Valid model not found in {model_path}.")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    if data_path.suffix == ".csv":
        sentences, ground_truth = DatasetLoader.load_csv(
            str(data_path), args.text_col, args.label_col
        )
    elif data_path.suffix in [".jsonl", ".json"]:
        sentences, ground_truth = DatasetLoader.load_json(
            str(data_path), args.text_col, args.label_col
        )
    else:
        print(f"Error: Unsupported file format {data_path.suffix}. Use .csv or .jsonl")
        sys.exit(1)

    print(f"Loaded {len(sentences)} test samples.")

    print(f"Loading model from {model_path}...")
    classifier = SentenceClassifier()
    classifier.load(str(model_path))

    print("Generating predictions...")
    predictions = classifier.predict(sentences)

    print("Evaluating...")
    evaluator = Evaluator()
    results = evaluator.evaluate(predictions, ground_truth)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(json.dumps(results, indent=2))
    print("=" * 40)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
