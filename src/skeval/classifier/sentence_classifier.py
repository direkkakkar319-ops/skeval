import json
import os
import random
import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from skeval.utils.helpers import LabelEncoder, VocabBuilder


def _validate_input(X, y=None):
    """Raise ``ValueError`` if ``X`` or ``y`` are malformed.

    Args:
        X: Candidate input sentences.
        y: Candidate labels aligned with ``X``. Pass ``None`` to skip label
            validation (e.g. when calling ``predict``).

    Raises:
        ValueError: If ``X`` or ``y`` are empty, contain non-strings, or have
            mismatched lengths.
    """
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
    """EmbeddingBag + Linear text classifier.

    A lightweight bag-of-words model: token indices are averaged by
    ``EmbeddingBag`` and then projected to class logits by a single linear
    layer.

    Attributes:
        embedding: ``nn.EmbeddingBag`` that averages token embeddings.
        fc: Linear layer that projects the averaged embedding to class logits.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        """Build the embedding and linear layers and initialise weights.

        Args:
            vocab_size: Total number of tokens in the vocabulary (including
                ``<PAD>`` and ``<UNK>``).
            embed_dim: Dimensionality of each token embedding.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        """Initialise embedding and linear weights with a uniform distribution.

        Weights are drawn from ``Uniform(-0.5, 0.5)`` and biases are set to
        zero. This gives a balanced starting point that avoids saturation.
        """
        r = 0.5
        self.embedding.weight.data.uniform_(-r, r)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """Compute class logits for a batch of sentences.

        Args:
            text: Flat 1-D ``LongTensor`` of concatenated token indices.
            offsets: 1-D ``LongTensor`` of sentence start positions within
                ``text``, as produced by ``collate_fn``.

        Returns:
            ``FloatTensor`` of shape ``(batch_size, num_classes)`` containing
            raw (pre-softmax) class scores.
        """
        return self.fc(self.embedding(text, offsets))


class SentenceClassifier:
    """sklearn-compatible sentence classifier backed by a bag-of-words neural network.

    Implements the full sklearn estimator interface (``fit``, ``predict``,
    ``score``, ``get_params``, ``set_params``) so it works directly with
    ``GridSearchCV``, ``cross_val_score``, and similar utilities.

    Attributes:
        embed_dim: Embedding dimensionality.
        epochs: Number of training epochs.
        batch_size: Mini-batch size used during training.
        lr: Adam learning rate.
        random_state: Seed for reproducibility, or ``None`` for non-deterministic runs.
        model: The underlying ``BasicTextClassifier``, or ``None`` before fitting.
        vocab: ``VocabBuilder`` instance populated during ``fit``.
        label_encoder: ``LabelEncoder`` instance populated during ``fit``.
        device: Torch device (``cuda`` if available, else ``cpu``).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.005,
        random_state: Optional[int] = None,
    ):
        """Initialise the classifier with training hyper-parameters.

        Args:
            embed_dim: Size of each token embedding vector.
            epochs: Number of full passes over the training data.
            batch_size: Number of samples per gradient update.
            lr: Learning rate for the Adam optimiser.
            random_state: Integer seed passed to Python, NumPy, and PyTorch
                random generators. Set to an integer for reproducible results.
        """
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

    def get_params(self, deep=True):
        """Return hyper-parameter names and values (sklearn estimator protocol).

        Args:
            deep: Ignored — included for sklearn API compatibility.

        Returns:
            Dictionary mapping parameter names to their current values.
        """
        del deep
        return {
            "embed_dim": self.embed_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set hyper-parameters by name (sklearn estimator protocol).

        Args:
            **params: Keyword arguments where each key is a valid parameter
                name and the value is the new setting.

        Returns:
            The classifier instance (``self``), enabling method chaining.

        Raises:
            ValueError: If any key is not a recognised parameter name.
        """
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter '{k}' for SentenceClassifier.")
            setattr(self, k, v)
        return self

    def fit(self, X: List[str], y: List[str]):
        """Build the vocabulary and train the model on labelled sentences.

        Args:
            X: Training sentences.
            y: Corresponding class labels aligned with ``X``.

        Returns:
            The fitted classifier instance (``self``).

        Raises:
            ValueError: If ``X`` or ``y`` fail input validation.
        """
        _validate_input(X, y)
        self._seed()

        self.vocab.build(X)
        self.label_encoder.build(y)

        self.model = BasicTextClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            num_classes=self.label_encoder.num_classes,
        ).to(self.device)

        from skeval.dataset.loader import DatasetLoader

        loader = DatasetLoader.create_dataloader(
            X, y, self.vocab, self.label_encoder, batch_size=self.batch_size
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in tqdm(range(1, self.epochs + 1), desc="Training", unit="ep"):
            self.model.train()
            total_loss, total_acc = 0.0, 0

            for texts, targets, offsets in loader:
                texts = texts.to(self.device)
                targets = targets.to(self.device)
                offsets = offsets.to(self.device)

                optimizer.zero_grad()
                out = self.model(texts, offsets)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += (out.argmax(1) == targets).sum().item()

            tqdm.write(
                f"Epoch {epoch}/{self.epochs} | "
                f"loss={total_loss / len(loader):.4f} "
                f"acc={total_acc / len(X):.4f}"
            )

        return self

    def predict(self, X: List[str]) -> List[str]:
        """Predict the class label for each sentence in ``X``.

        Args:
            X: Sentences to classify.

        Returns:
            List of predicted label strings in the same order as ``X``.

        Raises:
            RuntimeError: If called before ``fit()`` or ``load()``.
            ValueError: If ``X`` fails input validation.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        _validate_input(X)

        self.model.eval()
        out = []
        with torch.no_grad():
            for s in X:
                ids = self.vocab.encode(s)
                if not ids:
                    ids = [0]
                text = torch.tensor(ids, dtype=torch.long, device=self.device)
                offset = torch.zeros(1, dtype=torch.long, device=self.device)
                logits = self.model(text, offset)
                out.append(self.label_encoder.decode(logits.argmax(1).item()))
        return out

    def score(self, X: List[str], y: List[str]) -> float:
        """Return mean accuracy over the provided samples.

        Args:
            X: Sentences to classify.
            y: True labels aligned with ``X``.

        Returns:
            Fraction of correctly classified samples (0.0 – 1.0).
        """
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / len(y)

    def train(self, sentences, labels, epochs=None, batch_size=None, lr=None):
        """Train the classifier (deprecated — use ``fit()`` instead).

        Args:
            sentences: Training sentences.
            labels: Corresponding class labels.
            epochs: Override the instance ``epochs`` value for this run.
            batch_size: Override the instance ``batch_size`` value for this run.
            lr: Override the instance ``lr`` value for this run.

        Returns:
            The fitted classifier instance (``self``).

        .. deprecated:: 0.2.0
            Use :meth:`fit` instead. ``train()`` will be removed in v0.3.0.
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

    def save(self, save_dir: str):
        """Persist the trained model and vocabulary metadata to disk.

        Writes two files into ``save_dir``:

        - ``model.pt`` — PyTorch state dict.
        - ``metadata.json`` — vocab, label mapping, and hyper-parameters.

        Args:
            save_dir: Directory path to write artefacts into (created if absent).

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))

        meta = {
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
            json.dump(meta, f)

    def load(self, save_dir: str):
        """Restore a previously saved model from disk.

        Reads ``model.pt`` and ``metadata.json`` from ``save_dir`` and
        reconstructs the classifier so it is ready for inference.

        Args:
            save_dir: Directory that was passed to a previous ``save()`` call.
        """
        with open(os.path.join(save_dir, "metadata.json"), "r") as f:
            meta = json.load(f)

        self.embed_dim = meta["embed_dim"]
        self.epochs = meta.get("epochs", self.epochs)
        self.batch_size = meta.get("batch_size", self.batch_size)
        self.lr = meta.get("lr", self.lr)

        self.vocab.word2idx = meta["vocab"]["word2idx"]
        self.vocab.idx2word = {int(k): v for k, v in meta["vocab"]["idx2word"].items()}
        self.vocab.is_built = meta["vocab"]["is_built"]

        self.label_encoder.label2idx = meta["labels"]["label2idx"]
        self.label_encoder.idx2label = {
            int(k): v for k, v in meta["labels"]["idx2label"].items()
        }
        self.label_encoder.is_built = meta["labels"]["is_built"]

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
