import json
import os
from typing import List

import torch
import torch.nn as nn

from skeval.utils.helpers import LabelEncoder, VocabBuilder


class BasicTextClassifier(nn.Module):
    """A fundamental PyTorch deep learning architecture for text classification.
    Uses an EmbeddingBag to average word vectors, followed by a Linear layer.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor, offsets: torch.Tensor):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class SentenceClassifier:
    """High-level Sentinel API to train and predict semantic sentence categories."""

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.model = None
        self.vocab = VocabBuilder()
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        sentences: List[str],
        labels: List[str],
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.005,
    ):
        """Train the classifier from the core."""
        print(f"Building vocab and label encoder on {len(sentences)} sentences...")
        self.vocab.build(sentences)
        self.label_encoder.build(labels)

        self.model = BasicTextClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            num_classes=self.label_encoder.num_classes,
        ).to(self.device)

        from skeval.dataset.loader import DatasetLoader

        dataloader = DatasetLoader.create_dataloader(
            sentences, labels, self.vocab, self.label_encoder, batch_size=batch_size
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print(f"Training on device: {self.device}")
        for epoch in range(1, epochs + 1):
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

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {total_loss / len(dataloader):.4f} | "
                f"Acc: {total_acc / len(sentences):.4f}"
            )

    def predict(self, sentences: List[str]) -> List[str]:
        """Predict labels for a list of sentences."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() or load() first.")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for sentence in sentences:
                encoded = torch.tensor(self.vocab.encode(sentence), dtype=torch.long)
                if len(encoded) == 0:
                    encoded = torch.tensor(
                        [0], dtype=torch.long
                    )  # Handle empty/unknown sentences

                text = encoded.to(self.device)
                offset = torch.tensor([0], dtype=torch.long).to(self.device)

                output = self.model(text, offset)
                predicted_idx = output.argmax(1).item()
                predictions.append(self.label_encoder.decode(predicted_idx))

        return predictions

    def save(self, save_dir: str):
        """Save the trained model and tokenizers."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))

        metadata = {
            "embed_dim": self.embed_dim,
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
        """Load a trained model and tokenizers."""
        with open(os.path.join(save_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.embed_dim = metadata["embed_dim"]

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
            torch.load(os.path.join(save_dir, "model.pt"), map_location=self.device)
        )
        self.model.eval()
