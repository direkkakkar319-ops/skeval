import json
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from skeval.utils.helpers import LabelEncoder, VocabBuilder


class SentenceDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset for sentences."""

    def __init__(
        self,
        sentences: List[str],
        labels: List[str],
        vocab: VocabBuilder,
        label_encoder: LabelEncoder,
    ) -> None:
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoded_sentence = self.vocab.encode(sentence)
        encoded_label = self.label_encoder.encode(label)

        return torch.tensor(encoded_sentence, dtype=torch.long), torch.tensor(
            encoded_label, dtype=torch.long
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function to pad variable-length sentences in a batch."""
    labels = []
    sentences = []
    offsets = [0]

    for text_tensor, label_tensor in batch:
        labels.append(label_tensor)
        sentences.append(text_tensor)
        offsets.append(text_tensor.size(0))

    labels_t = torch.tensor(labels, dtype=torch.long)
    offsets_t = torch.tensor(offsets[:-1]).cumsum(dim=0)
    sentences_t = torch.cat(sentences)

    return sentences_t, labels_t, offsets_t


class DatasetLoader:
    """Utility to load raw data from CSV or JSON into PyTorch DataLoaders."""

    @staticmethod
    def load_csv(
        filepath: str, text_col: str, label_col: str
    ) -> Tuple[List[str], List[str]]:
        """Load sentences and labels from a CSV file."""
        df = pd.read_csv(filepath)
        return df[text_col].tolist(), df[label_col].tolist()

    @staticmethod
    def load_json(
        filepath: str, text_key: str, label_key: str
    ) -> Tuple[List[str], List[str]]:
        """Load sentences and labels from a JSON lines file."""
        sentences: List[str] = []
        labels: List[str] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                sentences.append(data[text_key])
                labels.append(data[label_key])
        return sentences, labels

    @staticmethod
    def create_dataloader(
        sentences: List[str],
        labels: List[str],
        vocab: VocabBuilder,
        label_encoder: LabelEncoder,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:  # type: ignore[type-arg]
        """Create a PyTorch DataLoader from raw text and labels."""
        dataset = SentenceDataset(sentences, labels, vocab, label_encoder)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )
