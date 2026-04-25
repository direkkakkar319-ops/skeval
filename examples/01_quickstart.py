"""Quickstart: train a classifier and make predictions in under 20 lines.

Note: with only 8 training sentences the model may not generalize perfectly
to unseen phrasing. See 03_evaluation.py for a more robust example.
"""

from sentinel.classifier import SentenceClassifier

sentences = [
    "Water boils at 100 degrees Celsius",
    "Paris is the capital of France",
    "I am feeling very sad today",
    "This is the worst day of my life",
    "I think this movie is amazing",
    "In my opinion, pizza is the best food",
    "Please close the door",
    "Open the window right now",
]
labels = [
    "fact",
    "fact",
    "emotion",
    "emotion",
    "opinion",
    "opinion",
    "instruction",
    "instruction",
]

classifier = SentenceClassifier(embed_dim=64)
classifier.train(sentences, labels, epochs=30, lr=0.01)

test = [
    "The sky is blue",
    "I am so happy",
    "I believe dogs are better than cats",
    "Turn off the lights",
]
predictions = classifier.predict(test)

for sentence, label in zip(test, predictions):
    print(f"[{label:>12}]  {sentence}")
