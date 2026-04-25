"""Load training and test data from CSV and JSON Lines files.

Expected CSV format:
    text,label
    "Water boils at 100C",fact
    "I feel sad",emotion

Expected JSONL format (one JSON object per line):
    {"text": "Water boils at 100C", "label": "fact"}
    {"text": "I feel sad", "label": "emotion"}

This example creates small temporary files to demonstrate the API
without requiring any external dataset.
"""

import json
import os
import tempfile

import pandas as pd

from skeval.classifier import SentenceClassifier
from skeval.dataset.loader import DatasetLoader
from skeval.evaluator import Evaluator

# --- Build temporary sample files ---

rows = [
    ("Water boils at 100 degrees Celsius", "fact"),
    ("Paris is the capital of France", "fact"),
    ("The moon orbits the Earth", "fact"),
    ("I am feeling very sad today", "emotion"),
    ("This is the worst day of my life", "emotion"),
    ("I feel so excited right now", "emotion"),
    ("I think this movie is amazing", "opinion"),
    ("In my opinion, pizza is the best food", "opinion"),
    ("I believe coffee is better than tea", "opinion"),
    ("Please close the door", "instruction"),
    ("Open the window right now", "instruction"),
    ("Turn off the lights", "instruction"),
]

with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = os.path.join(tmpdir, "train.csv")
    jsonl_path = os.path.join(tmpdir, "test.jsonl")

    pd.DataFrame(rows, columns=["text", "label"]).to_csv(csv_path, index=False)

    test_rows = [
        {"text": "The sky is blue", "label": "fact"},
        {"text": "I am so happy", "label": "emotion"},
        {"text": "I believe dogs are better than cats", "label": "opinion"},
        {"text": "Turn off the lights", "label": "instruction"},
    ]
    with open(jsonl_path, "w") as f:
        for row in test_rows:
            f.write(json.dumps(row) + "\n")

    # --- Load from CSV ---
    train_sentences, train_labels = DatasetLoader.load_csv(csv_path, "text", "label")
    print(f"Loaded {len(train_sentences)} training samples from CSV.")

    # --- Train ---
    classifier = SentenceClassifier(embed_dim=64)
    classifier.train(train_sentences, train_labels, epochs=30, lr=0.01)

    # --- Load from JSONL ---
    test_sentences, test_labels = DatasetLoader.load_json(jsonl_path, "text", "label")
    print(f"Loaded {len(test_sentences)} test samples from JSONL.")

    # --- Predict and evaluate ---
    predictions = classifier.predict(test_sentences)
    evaluator = Evaluator()
    results = evaluator.evaluate(predictions, test_labels)

    print(f"\nAccuracy: {results['accuracy']:.2%}")
    for label, stats in results["per_class"].items():
        print(f"  {label:>12}  F1={stats['f1-score']:.2f}")
