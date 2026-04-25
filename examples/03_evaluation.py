"""Train a classifier then evaluate it with full metrics."""

import json

from sentinel.classifier import SentenceClassifier
from sentinel.evaluator import Evaluator

train_sentences = [
    "Water boils at 100 degrees Celsius",
    "Paris is the capital of France",
    "The moon orbits the Earth",
    "I am feeling very sad today",
    "This is the worst day of my life",
    "I feel so excited right now",
    "I think this movie is amazing",
    "In my opinion, pizza is the best food",
    "I believe coffee is better than tea",
    "Please close the door",
    "Open the window right now",
    "Turn off the lights",
]
train_labels = [
    "fact", "fact", "fact",
    "emotion", "emotion", "emotion",
    "opinion", "opinion", "opinion",
    "instruction", "instruction", "instruction",
]

classifier = SentenceClassifier(embed_dim=64)
classifier.train(train_sentences, train_labels, epochs=40, lr=0.01)

test_sentences = [
    "The sky is blue",
    "I am so happy",
    "I believe dogs are better than cats",
    "Turn off the lights",
]
test_labels = ["fact", "emotion", "opinion", "instruction"]

predictions = classifier.predict(test_sentences)
print("Predictions: ", predictions)
print("Ground truth:", test_labels)

evaluator = Evaluator()
results = evaluator.evaluate(predictions, test_labels)

print("\n--- Metrics ---")
print(f"Accuracy: {results['accuracy']:.2%}")
print("\nPer-class:")
for label, stats in results["per_class"].items():
    print(
        f"  {label:>12}  "
        f"P={stats['precision']:.2f}  "
        f"R={stats['recall']:.2f}  "
        f"F1={stats['f1-score']:.2f}"
    )
print("\nConfusion matrix (rows=true, cols=predicted):")
print("Labels:", results["labels"])
for row in results["confusion_matrix"]:
    print(" ", row)

print("\nFull results:")
print(json.dumps(results, indent=2))
