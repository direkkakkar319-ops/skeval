import argparse
import sys
from pathlib import Path

from sentinel.classifier import SentenceClassifier
from sentinel.dataset.loader import DatasetLoader


def main():
    parser = argparse.ArgumentParser(
        description="Train a Sentinel AI Sentence Classifier"
    )
    parser.add_argument(
        "--data", required=True, type=str, help="Path to training data (.csv or .jsonl)"
    )
    parser.add_argument(
        "--text-col", required=True, type=str, help="Column name containing text"
    )
    parser.add_argument(
        "--label-col", required=True, type=str, help="Column name containing labels"
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        type=str,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=64, help="Embedding dimension (default: 64)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.005, help="Learning rate (default: 0.005)"
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    if data_path.suffix == ".csv":
        sentences, labels = DatasetLoader.load_csv(
            str(data_path), args.text_col, args.label_col
        )
    elif data_path.suffix in [".jsonl", ".json"]:
        sentences, labels = DatasetLoader.load_json(
            str(data_path), args.text_col, args.label_col
        )
    else:
        print(f"Error: Unsupported file format {data_path.suffix}. Use .csv or .jsonl")
        sys.exit(1)

    print(f"Loaded {len(sentences)} training samples.")

    classifier = SentenceClassifier(embed_dim=args.embed_dim)
    classifier.train(
        sentences, labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    print(f"Saving model to {args.save_dir}...")
    classifier.save(args.save_dir)
    print("Training complete! 🎉")


if __name__ == "__main__":
    main()
