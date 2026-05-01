# skeval

**Semantic Evaluation Layer for LLMs**

skeval is a lightweight library designed to evaluate how well Large Language Models (LLMs) understand and generate different types of sentences—such as facts, emotions, opinions, and instructions.

---

## 🚀 Motivation

Most LLM evaluation focuses on:

* Accuracy
* BLEU / ROUGE scores
* Reasoning benchmarks

But real-world language understanding also requires:

* Distinguishing facts from opinions
* Detecting emotions
* Identifying intent and instruction

skeval fills this gap by providing a **semantic classification and evaluation layer**.

---

## 🧠 What It Does

* Classifies sentences into categories:

  * Fact
  * Emotion
  * Opinion
  * Instruction
  * (extendable)

* Evaluates LLM outputs based on:

  * Classification accuracy
  * Confusion between categories
  * Per-class metrics

* Works with:

  * LLM outputs
  * Custom datasets
  * Benchmark pipelines

---

## 📦 Features

* Modular architecture (classifier, evaluator, metrics)
* Custom evaluation metrics for semantic types
* Compatible with LLM pipelines
* Extensible label taxonomy
* Clean CLI support (planned)

---

## 🏗️ Project Structure

```
skeval/
│
├── src/skeval/
│   ├── classifier/
│   ├── evaluator/
│   ├── metrics/
│   └── dataset/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
├── scripts/
├── docs/
└── notebooks/
```

---

## ⚙️ Installation

```bash
git clone github.com/direkkakkar319-ops/skeval
cd skeval
pip install -e .
```

---

## 🧪 Example Usage

```python
from skeval.classifier import SentenceClassifier
from skeval.evaluator import Evaluator

sentences = [
    "Water boils at 100 degrees Celsius",
    "I feel sad today",
    "I think this movie is amazing",
    "Please close the door",
]
labels = ["fact", "emotion", "opinion", "instruction"]

classifier = SentenceClassifier(embed_dim=64)
classifier.train(sentences, labels, epochs=20)

predictions = classifier.predict([
    "The sky is blue",
    "I am so happy",
    "I believe dogs are better than cats",
    "Turn off the lights",
])

evaluator = Evaluator()
results = evaluator.evaluate(predictions, ["fact", "emotion", "opinion", "instruction"])
print(results)
```

---

## 📊 Example Output

```
{
  "accuracy": 0.75,
  "per_class": {"fact": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, ...}, ...},
  "macro_avg": {"precision": ..., "recall": ..., "f1-score": ...},
  "weighted_avg": {"precision": ..., "recall": ..., "f1-score": ...},
  "confusion_matrix": [[...], ...],
  "labels": ["emotion", "fact", "instruction", "opinion"]
}
```

---

## 📚 Documentation

Full documentation (Sphinx-based) is available in the `docs/` directory.

To build locally:

```bash
cd docs
make html
```

---

## 🧠 Future Roadmap

* Multi-label classification (mixed sentences)
* Sarcasm detection
* Benchmark dataset release
* Integration with LLM evaluation tools
* CLI interface

---

## 🤝 Contributing

Contributions are welcome!

Please read `CONTRIBUTING.md` before submitting a PR.

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

This project is for research and educational purposes.
It does not guarantee perfect semantic understanding and should not be used for critical decision-making systems without validation.

---

## ⭐ Acknowledgments

Inspired by the need for better semantic evaluation in modern LLM systems.

---

## 🔥 Tagline

> *“Not just what the model says—but what it means.”*
