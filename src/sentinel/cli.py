import argparse
import json
import sys
from pathlib import Path


def _train(args):
    from sentinel.classifier import SentenceClassifier
    from sentinel.dataset.loader import DatasetLoader

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    if data_path.suffix == ".csv":
        sentences, labels = DatasetLoader.load_csv(
            str(data_path), args.text_col, args.label_col
        )
    elif data_path.suffix in (".jsonl", ".json"):
        sentences, labels = DatasetLoader.load_json(
            str(data_path), args.text_col, args.label_col
        )
    else:
        print(f"Error: unsupported format '{data_path.suffix}'. Use .csv or .jsonl")
        sys.exit(1)

    print(f"Loaded {len(sentences)} samples.")
    classifier = SentenceClassifier(embed_dim=args.embed_dim)
    classifier.train(
        sentences, labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    classifier.save(args.save_dir)
    print(f"Model saved to {args.save_dir}")


def _evaluate(args):
    from sentinel.classifier import SentenceClassifier
    from sentinel.dataset.loader import DatasetLoader
    from sentinel.evaluator import Evaluator

    model_path = Path(args.model_dir)
    if not (model_path / "model.pt").exists():
        print(f"Error: no trained model found in {model_path}.")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    if data_path.suffix == ".csv":
        sentences, ground_truth = DatasetLoader.load_csv(
            str(data_path), args.text_col, args.label_col
        )
    elif data_path.suffix in (".jsonl", ".json"):
        sentences, ground_truth = DatasetLoader.load_json(
            str(data_path), args.text_col, args.label_col
        )
    else:
        print(f"Error: unsupported format '{data_path.suffix}'. Use .csv or .jsonl")
        sys.exit(1)

    classifier = SentenceClassifier()
    classifier.load(str(model_path))
    predictions = classifier.predict(sentences)

    evaluator = Evaluator()
    results = evaluator.evaluate(predictions, ground_truth)

    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel AI — Semantic Evaluation Layer for LLMs",
    )
    parser.add_argument("--version", action="version", version="sentinel-ai 0.1.1")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # train
    train_parser = subparsers.add_parser("train", help="Train a sentence classifier")
    train_parser.add_argument(
        "--data", required=True, help="Path to .csv or .jsonl training file"
    )
    train_parser.add_argument(
        "--text-col", required=True, dest="text_col", help="Column name for text"
    )
    train_parser.add_argument(
        "--label-col", required=True, dest="label_col", help="Column name for labels"
    )
    train_parser.add_argument(
        "--save-dir", required=True, dest="save_dir", help="Directory to save the model"
    )
    train_parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        dest="embed_dim",
        help="Embedding dimension (default: 64)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Batch size (default: 32)",
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.005, help="Learning rate (default: 0.005)"
    )
    train_parser.set_defaults(func=_train)

    # evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a trained classifier"
    )
    eval_parser.add_argument(
        "--model-dir",
        required=True,
        dest="model_dir",
        help="Directory containing saved model",
    )
    eval_parser.add_argument(
        "--data", required=True, help="Path to .csv or .jsonl test file"
    )
    eval_parser.add_argument(
        "--text-col", required=True, dest="text_col", help="Column name for text"
    )
    eval_parser.add_argument(
        "--label-col", required=True, dest="label_col", help="Column name for labels"
    )
    eval_parser.add_argument(
        "--output", default=None, help="Optional path to save JSON results"
    )
    eval_parser.set_defaults(func=_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
