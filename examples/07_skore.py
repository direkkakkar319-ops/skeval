"""Tracking experiments with skore.

skore is an open-source ML experiment tracker that works with any
sklearn-compatible estimator.  This example shows how to log a
GridSearchCV run and inspect results in the skore UI.

Install skore first:
    pip install skore

Then launch the UI in a separate terminal:
    skore launch my_project
"""

import skore
from sklearn.model_selection import GridSearchCV, cross_val_score

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

# --- open (or create) a skore project ---
project = skore.open("my_project", overwrite=True)

# --- cross-validated baseline ---
clf = SentenceClassifier(embed_dim=64, epochs=40, lr=0.01, random_state=42)
cv_scores = cross_val_score(clf, sentences, labels, cv=2, scoring="accuracy")
project.put("baseline_cv_accuracy", cv_scores.mean())
print(f"Baseline CV accuracy: {cv_scores.mean():.2%}")

# --- grid search ---
param_grid = {
    "embed_dim": [32, 64],
    "epochs": [20, 40],
    "lr": [0.005, 0.01],
}
search = GridSearchCV(
    SentenceClassifier(random_state=42),
    param_grid,
    cv=2,
    scoring="accuracy",
    n_jobs=1,
    verbose=0,
)
search.fit(sentences, labels)

project.put("best_params", search.best_params_)
project.put("best_cv_accuracy", search.best_score_)

print(f"Best params  : {search.best_params_}")
print(f"Best CV acc  : {search.best_score_:.2%}")
print("\nOpen the skore UI to explore results:")
print("  skore launch my_project")
