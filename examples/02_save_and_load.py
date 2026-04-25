"""Save a trained model to disk and reload it in a new session."""

from skeval.classifier import SentenceClassifier

sentences = [
    "Water boils at 100 degrees Celsius",
    "I feel happy today",
    "I think cats are better than dogs",
    "Please open the door",
]
labels = ["fact", "emotion", "opinion", "instruction"]

classifier = SentenceClassifier(embed_dim=64)
classifier.train(sentences, labels, epochs=20)
classifier.save("saved_model/")
print("Model saved to saved_model/")

# --- Simulate a new session ---

new_classifier = SentenceClassifier()
new_classifier.load("saved_model/")
print("Model loaded from saved_model/")

predictions = new_classifier.predict(["The earth orbits the sun", "I am furious"])
print("Predictions:", predictions)
