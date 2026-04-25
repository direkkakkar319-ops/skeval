import os

from skeval.classifier import SentenceClassifier


def test_classifier_initialization():
    classifier = SentenceClassifier(embed_dim=32)
    assert classifier.embed_dim == 32
    assert classifier.model is None


def test_classifier_training_and_prediction():
    classifier = SentenceClassifier(embed_dim=16)

    sentences = ["This is a fact", "I am happy", "I think so", "Do this now"]
    labels = ["fact", "emotion", "opinion", "instruction"]

    classifier.train(sentences, labels, epochs=2)

    assert classifier.model is not None
    assert classifier.vocab.is_built is True
    assert classifier.label_encoder.is_built is True

    predictions = classifier.predict(["I am sad"])
    assert len(predictions) == 1
    assert predictions[0] in labels


def test_classifier_save_and_load(tmp_path):
    classifier = SentenceClassifier(embed_dim=16)

    sentences = ["Water boils at 100C", "I love you"]
    labels = ["fact", "emotion"]

    classifier.train(sentences, labels, epochs=1)

    save_dir = tmp_path / "model_out"
    classifier.save(str(save_dir))

    assert os.path.exists(save_dir / "model.pt")
    assert os.path.exists(save_dir / "metadata.json")

    new_classifier = SentenceClassifier()
    new_classifier.load(str(save_dir))

    assert new_classifier.embed_dim == 16
    assert new_classifier.vocab.is_built is True
    assert new_classifier.label_encoder.is_built is True
    assert new_classifier.model is not None
