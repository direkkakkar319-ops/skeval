"""Hyperparameter tuning with GridSearchCV.

SentenceClassifier implements get_params() / set_params() so it works
directly with sklearn's GridSearchCV.  The search picks the best
combination of embed_dim, epochs, and lr on cross-validated accuracy.
"""

from sklearn.model_selection import GridSearchCV

from skeval.classifier import SentenceClassifier

sentences = [
    "Water boils at 100 degrees Celsius",
    "Paris is the capital of France",
    "The moon orbits the Earth",
    "Light travels at 300,000 km per second",
    "I am feeling very sad today",
    "This is the worst day of my life",
    "I feel so excited right now",
    "She was overwhelmed with joy",
    "I think this movie is amazing",
    "In my opinion, pizza is the best food",
    "I believe coffee is better than tea",
    "Personally, I prefer winter over summer",
    "Please close the door",
    "Open the window right now",
    "Turn off the lights",
    "Send me the report by Monday",
]
labels = [
    "fact", "fact", "fact", "fact",
    "emotion", "emotion", "emotion", "emotion",
    "opinion", "opinion", "opinion", "opinion",
    "instruction", "instruction", "instruction", "instruction",
]

param_grid = {
    "embed_dim": [32, 64],
    "epochs": [20, 40],
    "lr": [0.005, 0.01],
}

clf = SentenceClassifier(random_state=42)
search = GridSearchCV(clf, param_grid, cv=2, scoring="accuracy", n_jobs=1, verbose=1)
search.fit(sentences, labels)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV acc : {search.best_score_:.2%}")

test = [
    "The sky is blue",
    "I am so happy",
    "I believe dogs are better than cats",
    "Turn off the lights",
]
preds = search.predict(test)
for sentence, label in zip(test, preds):
    print(f"[{label:>12}]  {sentence}")
