# Sentinel AI

**Semantic Evaluation Layer for LLMs**

Sentinel AI is a lightweight library designed to evaluate how well Large Language Models (LLMs) understand and generate different types of sentences—such as facts, emotions, opinions, and instructions.

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

Sentinel AI fills this gap by providing a **semantic classification and evaluation layer**.

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
sentinel-ai/
│
├── src/sentinel/
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
git clone https://github.com/your-username/sentinel-ai.git
cd sentinel-ai
pip install -e .
```

---

## 🧪 Example Usage

```python
from sentinel.classifier import SentenceClassifier
from sentinel.evaluator import Evaluator

classifier = SentenceClassifier()
evaluator = Evaluator()

sentences = [
    "Water boils at 100 degrees Celsius",
    "I feel sad today",
    "This movie is amazing",
]

predictions = [classifier.predict(s) for s in sentences]

# Example ground truth
labels = ["fact", "emotion", "opinion"]

results = evaluator.evaluate(predictions, labels)
print(results)
```

---

## 📊 Example Output

```
{
  "accuracy": 0.66,
  "confusion_matrix": [...],
  "per_class_f1": {...}
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
