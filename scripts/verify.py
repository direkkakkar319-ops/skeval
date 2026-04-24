import json

from sentinel.classifier import SentenceClassifier
from sentinel.evaluator import Evaluator


def main():
    classifier = SentenceClassifier(embed_dim=16)

    train_sentences = [
        "Water boils at 100 degrees Celsius",
        "Paris is the capital of France",
        "I am feeling very sad today",
        "This is the worst day of my life",
        "I think this movie is amazing",
        "In my opinion, pizza is the best food",
        "Please close the door",
        "Open the window right now",
    ]
    train_labels = [
        "fact",
        "fact",
        "emotion",
        "emotion",
        "opinion",
        "opinion",
        "instruction",
        "instruction",
    ]

    print("Training PyTorch model...")
    classifier.train(train_sentences, train_labels, epochs=50, lr=0.05)

    test_sentences = [
        "The sky is blue",
        "I am so happy",
        "I believe dogs are better than cats",
        "Turn off the lights",
    ]
    test_labels = ["fact", "emotion", "opinion", "instruction"]

    print("\nPredicting test data...")
    predictions = classifier.predict(test_sentences)
    print("Predictions:", predictions)

    print("\nEvaluating...")
    evaluator = Evaluator()
    results = evaluator.evaluate(predictions, test_labels)

    print("\nMetrics:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
