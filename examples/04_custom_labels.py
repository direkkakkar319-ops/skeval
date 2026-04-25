"""Use a custom label taxonomy instead of the four defaults.

Sentinel AI imposes no fixed label set — you define the categories
by what you put in your training data.
"""

from sentinel.classifier import SentenceClassifier
from sentinel.evaluator import Evaluator

# Domain: customer support ticket classification
sentences = [
    "My order hasn't arrived after two weeks",
    "I still haven't received my package",
    "The delivery is very late",
    "How do I reset my password?",
    "Where can I find my invoice?",
    "What is your return policy?",
    "I love this product, it works perfectly",
    "Best purchase I've made this year",
    "Absolutely happy with the service",
    "I want a refund immediately",
    "Please cancel my subscription",
    "I demand to speak to a manager",
]
labels = [
    "shipping_issue",
    "shipping_issue",
    "shipping_issue",
    "account_help",
    "account_help",
    "account_help",
    "positive_feedback",
    "positive_feedback",
    "positive_feedback",
    "complaint",
    "complaint",
    "complaint",
]

classifier = SentenceClassifier(embed_dim=32)
classifier.train(sentences, labels, epochs=40, lr=0.01)

test_sentences = [
    "My parcel is lost",
    "How do I update my email address?",
    "This is exactly what I needed, thank you!",
    "I want my money back now",
]
test_labels = ["shipping_issue", "account_help", "positive_feedback", "complaint"]

predictions = classifier.predict(test_sentences)
for sentence, pred, truth in zip(test_sentences, predictions, test_labels):
    match = "OK" if pred == truth else "MISS"
    print(f"[{match}] predicted={pred:>18}  true={truth:>18}  | {sentence}")

evaluator = Evaluator()
results = evaluator.evaluate(predictions, test_labels)
print(f"\nAccuracy: {results['accuracy']:.2%}")
