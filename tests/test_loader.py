import json

import pandas as pd

from sentinel.dataset.loader import DatasetLoader
from sentinel.utils.helpers import LabelEncoder, VocabBuilder


def test_vocab_builder():
    vocab = VocabBuilder(min_freq=1)
    sentences = ["Hello world", "Hello there"]
    vocab.build(sentences)

    assert vocab.is_built is True
    # <PAD>, <UNK>, hello, world, there = 5
    assert len(vocab) == 5

    encoded = vocab.encode("Hello unknown")
    # hello should be found, unknown should be <UNK> (1)
    assert encoded[1] == 1


def test_label_encoder():
    encoder = LabelEncoder()
    labels = ["fact", "emotion", "fact"]
    encoder.build(labels)

    assert encoder.num_classes == 2
    assert encoder.encode("fact") in [0, 1]

    decoded = encoder.decode(encoder.encode("emotion"))
    assert decoded == "emotion"


def test_dataset_loader_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"text": ["s1", "s2"], "label": ["fact", "emotion"]})
    df.to_csv(csv_file, index=False)

    texts, labels = DatasetLoader.load_csv(str(csv_file), "text", "label")
    assert texts == ["s1", "s2"]
    assert labels == ["fact", "emotion"]


def test_dataset_loader_json(tmp_path):
    json_file = tmp_path / "data.jsonl"
    with open(json_file, "w") as f:
        f.write(json.dumps({"text": "s1", "label": "fact"}) + "\n")
        f.write(json.dumps({"text": "s2", "label": "emotion"}) + "\n")

    texts, labels = DatasetLoader.load_json(str(json_file), "text", "label")
    assert texts == ["s1", "s2"]
    assert labels == ["fact", "emotion"]
