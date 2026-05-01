import json
import os
import random
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from skeval.utils.helpers import LabelEncoder, VocabBuilder


def _validate_input(X, y=None):
    if not isinstance(X, (list, tuple)) or len(X) == 0:
        raise ValueError("X must be a non-empty list of strings.")
    if not all(isinstance(s, str) for s in X):
        raise ValueError("All elements of X must be strings.")
    if y is not None:
        if not isinstance(y, (list, tuple)) or len(y) == 0:
            raise ValueError("y must be a non-empty list of strings.")
        if not all(isinstance(s, str) for s in y):
            raise ValueError("All elements of y must be strings.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length, got {len(X)} and {len(y)}."
            )


class BasicTextClassifier(nn.Module):
    """A minimal PyTorch model for text classification.

    Architecture:
        1. EmbeddingBag — looks up a learned vector for each word in the
           sentence and averages them into a single fixed-size vector,
           regardless of how long the sentence is.
        2. Linear — maps that averaged vector to one score per class.
           The class with the highest score becomes the prediction.

    Args:
        vocab_size: Total number of unique words in the vocabulary.
        embed_dim: Size of each word's embedding vector (e.g. 64).
        num_classes: Number of output categories (e.g. 4 for fact /
            emotion / opinion / instruction).
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        """Initialise weights with small uniform values.

        Uniform initialisation in [-0.5, 0.5] gives all words a similar
        starting point and avoids vanishing/exploding gradients early in
        training. Biases start at zero so no class is favoured initially.
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor, offsets: torch.Tensor):
        """Run a forward pass through the network.

        Args:
            text: 1-D tensor of word indices for the whole batch,
                concatenated end-to-end.
            offsets: 1-D tensor marking where each sentence starts
                inside ``text``. EmbeddingBag uses this to average
                each sentence separately.

        Returns:
            Tensor of shape ``(batch_size, num_classes)`` with one
            raw score (logit) per class for each sentence.
        """
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class SentenceClassifier:
    """High-level classifier for labelling the semantic type of sentences.

    Wraps ``BasicTextClassifier`` with vocabulary building, label encoding,
    training loop, and persistence. Follows the sklearn estimator interface
    so it works inside ``sklearn.pipeline.Pipeline`` and ``GridSearchCV``.

    Args:
        embed_dim: Size of word embedding vectors. Larger values can capture
            more meaning but train more slowly. Default: 64.
        epochs: Number of full passes over the training data. Default: 5.
        batch_size: Number of sentences processed together per gradient
            update. Larger batches are faster but need more memory. Default: 32.
        lr: Learning rate for the Adam optimiser. Controls how large each
            weight update step is. Default: 0.005.
        random_state: Seed for ``random``, ``numpy``, and ``torch`` to make
            training reproducible on CPU. Default: None (non-deterministic).

    Example:
        >>> clf = SentenceClassifier(embed_dim=64, epochs=10, random_state=42)
        >>> clf.fit(["The sky is blue", "I feel sad"], ["fact", "emotion"])
        >>> clf.predict(["Water boils at 100C"])
        ['fact']
        >>> clf.score(["I feel happy"], ["emotion"])
        1.0
    """

    def __init__(
        self,
        embed_dim: int = 64,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.005,
        random_state: Optional[int] = None,
    ):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.model = None
        self.vocab = VocabBuilder()
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _seed(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

    # ------------------------------------------------------------------
    # sklearn estimator interface
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the classifier's hyperparameters as a dictionary.

        Required by the sklearn estimator interface so that tools like
        ``GridSearchCV`` can read and vary the hyperparameters automatically.

        Args:
            deep: Ignored — ``SentenceClassifier`` has no nested estimators.

        Returns:
            Dictionary mapping parameter names to their current values.

        Example:
            >>> clf = SentenceClassifier(embed_dim=128, epochs=20)
            >>> clf.get_params()
            {'embed_dim': 128, 'epochs': 20, 'batch_size': 32, 'lr': 0.005, 'random_state': None}
        """
        del deep  # no nested estimators
        return {
            "embed_dim": self.embed_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "SentenceClassifier":
        """Update one or more hyperparameters in place.

        Required by the sklearn estimator interface so that tools like
        ``GridSearchCV`` can set hyperparameters before each trial.

        Args:
            **params: Keyword arguments matching the names returned by
                ``get_params()``. Any unknown key raises ``ValueError``.

        Returns:
            The classifier itself, so calls can be chained.

        Raises:
            ValueError: If any key in ``params`` is not a valid parameter.

        Example:
            >>> clf = SentenceClassifier()
            >>> clf.set_params(embed_dim=128, epochs=20)
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for SentenceClassifier.")
            setattr(self, key, value)
        return self

    def fit(
        self,
        X: List[str],
        y: List[str],
    ) -> "SentenceClassifier":
        """Train the classifier on a labelled dataset.

        Builds the vocabulary and label encoding from the training data,
        constructs the neural network, then runs the training loop for
        ``self.epochs`` epochs using Adam optimisation and cross-entropy loss.

        Args:
            X: List of raw sentences to train on.
            y: Corresponding list of string labels (e.g. ``"fact"``,
               ``"emotion"``, ``"opinion"``, ``"instruction"``). Must be
               the same length as ``X``.

        Returns:
            The classifier itself so calls can be chained, e.g.
            ``clf.fit(X, y).predict(X_test)``.

        Example:
            >>> clf = SentenceClassifier(epochs=10)
            >>> clf.fit(
            ...     ["Water boils at 100C", "I feel happy"],
            ...     ["fact", "emotion"],
            ... )
        """
        _validate_input(X, y)
        self._seed()
        tqdm.write(f"Building vocab on {len(X)} sentences...")
        self.vocab.build(X)
        self.label_encoder.build(y)

        self.model = BasicTextClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            num_classes=self.label_encoder.num_classes,
        ).to(self.device)

        from skeval.dataset.loader import DatasetLoader

        dataloader = DatasetLoader.create_dataloader(
            X, y, self.vocab, self.label_encoder, batch_size=self.batch_size
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        tqdm.write(f"Training on device: {self.device}")
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training", unit="epoch"):
            self.model.train()
            total_loss = 0.0
            total_acc = 0

            for texts, targets, offsets in dataloader:
                texts, targets, offsets = (
                    texts.to(self.device),
                    targets.to(self.device),
                    offsets.to(self.device),
                )

                optimizer.zero_grad()
                output = self.model(texts, offsets)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += (output.argmax(1) == targets).sum().item()

            tqdm.write(
                f"Epoch {epoch}/{self.epochs} | "
                f"Loss: {total_loss / len(dataloader):.4f} | "
                f"Acc: {total_acc / len(X):.4f}"
            )

        return self

    def predict(self, X: List[str]) -> List[str]:
        """Predict the semantic label for each sentence in X.

        Encodes each sentence using the trained vocabulary, runs it through
        the model, and converts the highest-scoring output index back to a
        string label.

        Args:
            X: List of raw sentences to classify.

        Returns:
            List of predicted string labels, one per input sentence,
            in the same order as ``X``.

        Raises:
            RuntimeError: If called before ``fit()`` or ``load()``.

        Example:
            >>> clf.predict(["The sky is blue", "I am so happy"])
            ['fact', 'emotion']
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        _validate_input(X)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for sentence in X:
                encoded = torch.tensor(self.vocab.encode(sentence), dtype=torch.long)
                if len(encoded) == 0:
                    encoded = torch.tensor([0], dtype=torch.long)

                text = encoded.to(self.device)
                offset = torch.tensor([0], dtype=torch.long).to(self.device)

                output = self.model(text, offset)
                predicted_idx = output.argmax(1).item()
                predictions.append(self.label_encoder.decode(predicted_idx))

        return predictions

    def score(self, X: List[str], y: List[str]) -> float:
        """Return the accuracy of the classifier on the given data.

        Convenience method so ``SentenceClassifier`` can be used directly
        with sklearn tools that call ``estimator.score(X, y)``.

        Args:
            X: List of raw sentences to evaluate on.
            y: Correct string labels for each sentence in ``X``.

        Returns:
            Fraction of sentences classified correctly, between 0.0 and 1.0.

        Example:
            >>> clf.score(["The sky is blue"], ["fact"])
            1.0
        """
        predictions = self.predict(X)
        return sum(p == t for p, t in zip(predictions, y)) / len(y)

    def train(
        self,
        sentences: List[str],
        labels: List[str],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
    ):
        """Deprecated — use ``fit()`` instead.

        This method is kept for backwards compatibility only and will be
        removed in v0.3.0. Switch all callers to ``fit(X, y)``.

        Args:
            sentences: List of training sentences.
            labels: Corresponding list of string labels.
            epochs: Overrides ``self.epochs`` for this run if provided.
            batch_size: Overrides ``self.batch_size`` for this run if provided.
            lr: Overrides ``self.lr`` for this run if provided.
        """
        warnings.warn(
            "train() is deprecated and will be removed in v0.3.0. Use fit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if lr is not None:
            self.lr = lr
        return self.fit(sentences, labels)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str):
        """Save the trained model weights and metadata to disk.

        Writes two files into ``save_dir``:

        * ``model.pt`` — the PyTorch ``state_dict`` (weights only).
        * ``metadata.json`` — hyperparameters, vocabulary, and label
          mappings needed to reconstruct the model on load.

        Args:
            save_dir: Path to the directory to write files into.
                Created automatically if it does not exist.

        Raises:
            RuntimeError: If called before ``fit()``.

        Example:
            >>> clf.fit(X, y)
            >>> clf.save("saved_model/")
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))

        metadata = {
            "embed_dim": self.embed_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "vocab": {
                "word2idx": self.vocab.word2idx,
                "idx2word": self.vocab.idx2word,
                "is_built": self.vocab.is_built,
            },
            "labels": {
                "label2idx": self.label_encoder.label2idx,
                "idx2label": self.label_encoder.idx2label,
                "is_built": self.label_encoder.is_built,
            },
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load(self, save_dir: str):
        """Load a previously saved model from disk.

        Reads ``metadata.json`` to reconstruct the vocabulary, label
        encoder, and model architecture, then loads the saved weights
        from ``model.pt``. After this call the classifier is ready to
        call ``predict()`` without any training.

        Args:
            save_dir: Path to the directory produced by ``save()``.

        Example:
            >>> clf = SentenceClassifier()
            >>> clf.load("saved_model/")
            >>> clf.predict(["The sky is blue"])
            ['fact']
        """
        with open(os.path.join(save_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.embed_dim = metadata["embed_dim"]
        self.epochs = metadata.get("epochs", self.epochs)
        self.batch_size = metadata.get("batch_size", self.batch_size)
        self.lr = metadata.get("lr", self.lr)

        self.vocab.word2idx = metadata["vocab"]["word2idx"]
        self.vocab.idx2word = {
            int(k): v for k, v in metadata["vocab"]["idx2word"].items()
        }
        self.vocab.is_built = metadata["vocab"]["is_built"]

        self.label_encoder.label2idx = metadata["labels"]["label2idx"]
        self.label_encoder.idx2label = {
            int(k): v for k, v in metadata["labels"]["idx2label"].items()
        }
        self.label_encoder.is_built = metadata["labels"]["is_built"]

        self.model = BasicTextClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            num_classes=self.label_encoder.num_classes,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(save_dir, "model.pt"),
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model.eval()
