import json
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from skeval.utils.helpers import LabelEncoder, VocabBuilder


class SentenceDataset(Dataset):  # type: ignore[misc]
    """PyTorch ``Dataset`` that tokenises sentences on the fly.

    Attributes:
        sentences: Raw sentence strings.
        labels: Corresponding label strings.
        vocab: Fitted ``VocabBuilder`` used to encode sentences.
        label_encoder: Fitted ``LabelEncoder`` used to encode labels.
    """

    def __init__(
        self,
        sentences: List[str],
        labels: List[str],
        vocab: VocabBuilder,
        label_encoder: LabelEncoder,
    ) -> None:
        """Wrap sentence and label lists for use with a PyTorch DataLoader.

        Args:
            sentences: List of raw sentence strings.
            labels: List of label strings aligned with ``sentences``.
            vocab: Fitted vocabulary used to convert tokens to indices.
            label_encoder: Fitted encoder used to convert labels to indices.
        """
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the encoded sentence and label at position ``idx``.

        Args:
            idx: Integer index into the dataset.

        Returns:
            A 2-tuple ``(text_tensor, label_tensor)`` where ``text_tensor``
            is a 1-D ``LongTensor`` of token indices and ``label_tensor`` is
            a scalar ``LongTensor`` holding the class index.
        """
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
    """Collate variable-length sentences into a single batch for EmbeddingBag.

    EmbeddingBag expects a flat 1-D token tensor together with an offsets
    tensor that marks where each sentence starts.

    Args:
        batch: List of ``(text_tensor, label_tensor)`` pairs returned by
            ``SentenceDataset.__getitem__``.

    Returns:
        A 3-tuple ``(sentences, labels, offsets)`` where ``sentences`` is the
        concatenated flat token tensor, ``labels`` is a 1-D label tensor, and
        ``offsets`` is a 1-D tensor of sentence start positions.
    """
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
    """Utility for loading raw data from CSV or JSONL into PyTorch DataLoaders."""

    @staticmethod
    def load_csv(
        filepath: str, text_col: str, label_col: str
    ) -> Tuple[List[str], List[str]]:
        """Load sentences and labels from a CSV file.

        Args:
            filepath: Path to the CSV file.
            text_col: Name of the column containing sentence text.
            label_col: Name of the column containing class labels.

        Returns:
            A 2-tuple ``(sentences, labels)`` of equal-length string lists.
        """
        df = pd.read_csv(filepath)
        return df[text_col].tolist(), df[label_col].tolist()

    @staticmethod
    def load_json(
        filepath: str, text_key: str, label_key: str
    ) -> Tuple[List[str], List[str]]:
        """Load sentences and labels from a JSON lines file.

        Each line must be a JSON object with at least the two specified keys.

        Args:
            filepath: Path to the ``.jsonl`` file.
            text_key: Key whose value is the sentence text.
            label_key: Key whose value is the class label.

        Returns:
            A 2-tuple ``(sentences, labels)`` of equal-length string lists.
        """
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
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        """Wrap sentences and labels in a PyTorch DataLoader.

        Args:
            sentences: Raw sentence strings.
            labels: Corresponding label strings.
            vocab: Fitted ``VocabBuilder``.
            label_encoder: Fitted ``LabelEncoder``.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data each epoch.
            num_workers: Number of subprocesses for data loading. ``0`` means
                data is loaded in the main process.
            pin_memory: If ``True``, the DataLoader copies tensors to CUDA
                pinned memory before returning them. Only useful when training
                on a GPU.

        Returns:
            A ``DataLoader`` that yields ``(sentences, labels, offsets)``
            batches compatible with ``BasicTextClassifier``.
        """
        dataset = SentenceDataset(sentences, labels, vocab, label_encoder)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
