# Exercise 1 — Sentiment Analysis with DistilBERT

This exercise studies how a pretrained Transformer can be used and adapted for binary sentiment classification on the **Cornell Rotten Tomatoes** dataset.

The work is organized into three progressive stages:

1. **Exercise 1.1 — Dataset loading and exploration:** inspect the official splits, schema, labels, integrity, class balance, and sentence lengths.
2. **Exercise 1.2 — Tokenizer and pretrained DistilBERT:** tokenize real samples, inspect model inputs and outputs, and identify the sentence representation produced by the Transformer.
3. **Exercise 1.3 — Stable baseline:** use frozen DistilBERT representations as features for a classical classifier and evaluate the resulting sentiment predictions.

Only Exercise 1.1 has been implemented and verified at the current stage.

---

## Repository structure

```text
Exercise1/
├── main.py
├── data.py
├── eda.py
├── README.md
│
├── assets/
│   ├── eda_class_distribution.png
│   └── eda_text_length_distribution.png
│
└── outputs/
    └── exercise_1_1/
        ├── figures/
        │   ├── class_distribution.png
        │   └── text_length_distribution.png
        └── results/
            ├── class_distribution.csv
            ├── integrity_checks.csv
            ├── split_overlap.csv
            └── text_length_summary.csv
```

The `outputs/` directory contains generated artifacts. Selected figures intended for documentation are copied to `assets/`.

---

# Exercise 1.1 — Dataset Loading and Exploration

## Objective

The first task is to load the Cornell Rotten Tomatoes dataset with the Hugging Face `datasets` library and understand:

- which official splits are available;
- how examples and columns are accessed;
- how the dataset is structured;
- how sentiment labels are encoded;
- whether the data contains obvious integrity problems;
- whether the classes are balanced;
- how long the input sentences are.

No tokenizer or Transformer model is used in this exercise.

---

## Dataset

The dataset is loaded from:

```text
cornell-movie-review-data/rotten_tomatoes
```

Each example contains:

| Field | Type | Meaning |
|---|---|---|
| `text` | string | Processed sentence from a movie review |
| `label` | `ClassLabel` | Binary sentiment label |

The verified label mapping is:

| Label ID | Class |
|---:|---|
| `0` | negative (`neg`) |
| `1` | positive (`pos`) |

---

## Official splits

| Split | Examples | Percentage of total |
|---|---:|---:|
| Train | 8,530 | 80.00% |
| Validation | 1,066 | 10.00% |
| Test | 1,066 | 10.00% |
| **Total** | **10,662** | **100.00%** |

The official validation split is already available, so no additional train-validation split is created.

The intended use is:

- **train:** fit the classifier or fine-tune the model;
- **validation:** make development and model-selection decisions;
- **test:** evaluate the final selected configuration.

---

## Dataset access patterns

The inspection confirmed the main Hugging Face access patterns:

```text
dataset["train"][0]
```

returns one example as a Python dictionary.

```text
dataset["train"]["label"]
```

returns one complete column.

```text
dataset["train"][:3]
```

returns a dictionary of lists representing a raw batch.

```text
dataset["train"].select(range(3))
```

returns a new Hugging Face `Dataset` containing the selected rows.

At this stage, texts are still ordinary strings and labels are scalar integers. Tensor shapes will only appear after tokenization in Exercise 1.2.

---

## Integrity checks

The complete dataset was checked for missing values, invalid labels, empty strings, exact duplicates, and exact overlaps between splits.

| Split | Missing texts | Non-string texts | Empty texts | Invalid labels | Exact duplicates within split |
|---|---:|---:|---:|---:|---:|
| Train | 0 | 0 | 0 | 0 | 0 |
| Validation | 0 | 0 | 0 | 0 | 0 |
| Test | 0 | 0 | 0 | 0 | 0 |

Exact text overlap between different splits:

| Split pair | Exact overlap |
|---|---:|
| Train–Validation | 0 |
| Train–Test | 0 |
| Validation–Test | 0 |

These checks show that no obvious cleaning step is required before tokenization.

The overlap analysis only detects exact string matches; it does not attempt semantic or fuzzy duplicate detection.

---

## Class distribution

Every official split is perfectly balanced.

| Split | Negative | Positive | Negative share | Positive share |
|---|---:|---:|---:|---:|
| Train | 4,265 | 4,265 | 50.00% | 50.00% |
| Validation | 533 | 533 | 50.00% | 50.00% |
| Test | 533 | 533 | 50.00% | 50.00% |

<p align="center">
  <img src="assets/eda_class_distribution.png"
       alt="Negative and positive class distribution in the Rotten Tomatoes splits"
       width="780">
</p>

<p align="center">
  <em>Negative and positive examples are equally represented in train, validation, and test.</em>
</p>

### Consequence

The baseline does not initially require:

- class weighting;
- oversampling;
- undersampling;
- a balanced sampler.

A classifier that always predicts one class would obtain approximately 50% accuracy, so this provides a simple reference level.

---

## Sentence-length analysis

Sentence length was measured in characters and whitespace-separated words. Token counts are deliberately postponed until the DistilBERT tokenizer is introduced.

### Length in words

| Split | Mean | Standard deviation | Minimum | Median | 75th percentile | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 20.99 | 9.37 | 1 | 20 | 27.00 | 37.00 | 59 |
| Validation | 21.00 | 9.64 | 1 | 21 | 27.00 | 38.00 | 54 |
| Test | 21.22 | 9.51 | 3 | 20 | 27.75 | 38.00 | 52 |

### Length in characters

| Split | Mean | Standard deviation | Minimum | Median | 75th percentile | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 113.97 | 51.05 | 4 | 111 | 149 | 204 | 267 |
| Validation | 114.30 | 52.62 | 6 | 112 | 148 | 211 | 263 |
| Test | 115.52 | 50.96 | 14 | 113 | 151 | 206 | 261 |

<p align="center">
  <img src="assets/eda_text_length_distribution.png"
       alt="Sentence-length distributions in words"
       width="820">
</p>

<p align="center">
  <em>The three splits have very similar sentence-length distributions.</em>
</p>

### Interpretation

The dataset contains short review sentences rather than full-length reviews:

- the median sentence contains about 20 words;
- 95% of the training sentences contain at most 37 words;
- the longest training sentence contains 59 words;
- train, validation, and test have very similar length distributions.

This suggests that tokenization and batching should be computationally manageable. However, word counts cannot be directly interpreted as DistilBERT token counts because one word may be split into multiple subword tokens and special tokens are added by the tokenizer.

The effective token-length distribution and any truncation decision will therefore be studied in Exercise 1.2.

---

## Main conclusions

Exercise 1.1 establishes that:

1. the dataset loads correctly as a Hugging Face `DatasetDict`;
2. the official train, validation, and test splits are already available;
3. each example contains a text string and a binary sentiment label;
4. the mapping is `0 = negative` and `1 = positive`;
5. all three splits are perfectly balanced;
6. no missing texts, empty strings, invalid labels, or exact duplicates were found;
7. no exact text overlap was found between the official splits;
8. the sentences are generally short;
9. the three splits have comparable sentence-length distributions.

No dataset cleaning, resplitting, or class-balancing strategy is required for the initial baseline.

---

## Running Exercise 1.1

Activate the project environment:

```powershell
conda activate DLA2026-transformers
```

From the `DLA_LAB2` root directory, verify the Python files:

```powershell
python -m compileall -q Exercise1
```

Run the exploratory analysis:

```powershell
python Exercise1/main.py eda
```

The command downloads the public dataset on the first execution, stores it in the Hugging Face cache, prints the analysis to the console, and generates figures and CSV summaries under:

```text
Exercise1/outputs/exercise_1_1/
```

---

## Verified execution status

The complete Exercise 1.1 pipeline has been executed successfully in the Conda environment:

```text
DLA2026-transformers
```

The values reported in this README come from the generated console output and CSV artifacts, not from assumed or fabricated results.

---

## Next step

Exercise 1.2 will introduce:

- `AutoTokenizer`;
- pretrained DistilBERT through `AutoModel`;
- `input_ids`;
- `attention_mask`;
- padding and batching;
- model output inspection;
- `last_hidden_state`;
- extraction of the first-token (`[CLS]`) representation.
