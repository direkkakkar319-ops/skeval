import re
from collections import Counter
from typing import Dict, List


def normalize_text(text: str) -> str:
    """Normalize text by lowering case and removing punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


class VocabBuilder:
    """Builds a vocabulary from text corpus and encodes sentences into indices."""

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        # Reserve 0 for padding, 1 for unknown words
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.is_built = False

    def build(self, sentences: List[str]):
        """Build vocabulary from a list of sentences."""
        counter = Counter(
            word
            for sentence in sentences
            for word in normalize_text(sentence).split()
        )
        for idx, word in enumerate(
            (w for w, c in counter.items() if c >= self.min_freq),
            start=len(self.word2idx),
        ):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.is_built = True

    def encode(self, sentence: str) -> List[int]:
        """Convert a sentence into a list of vocabulary indices."""
        if not self.is_built:
            raise ValueError("Vocabulary has not been built yet. Call build() first.")
        words = normalize_text(sentence).split()
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]

    def __len__(self):
        return len(self.word2idx)


class LabelEncoder:
    """Encodes string labels to integers and vice-versa."""

    def __init__(self):
        self.label2idx: Dict[str, int] = {}
        self.idx2label: Dict[int, str] = {}
        self.is_built = False

    def build(self, labels: List[str]):
        """Build label encoding from a list of labels."""
        unique_labels = sorted(list(set(labels)))
        for idx, label in enumerate(unique_labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label
        self.is_built = True

    def encode(self, label: str) -> int:
        if not self.is_built:
            raise ValueError("LabelEncoder not built.")
        if label not in self.label2idx:
            raise ValueError(f"Unknown label: {label}")
        return self.label2idx[label]

    def decode(self, idx: int) -> str:
        if not self.is_built:
            raise ValueError("LabelEncoder not built.")
        return self.idx2label[idx]

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)
