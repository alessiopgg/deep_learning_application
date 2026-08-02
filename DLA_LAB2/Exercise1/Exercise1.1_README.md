# Exercise 1.1 — Dataset Loading and Exploration

## Deep Learning Applications — Laboratory 2

This exercise loads and explores the **Cornell Rotten Tomatoes** sentiment dataset through the Hugging Face `datasets` library.

The current implementation is intentionally compact. It focuses on the information required to understand the dataset before introducing the tokenizer and DistilBERT:

- official dataset splits;
- available columns;
- label mapping;
- class distribution;
- sentence lengths in characters and whitespace-separated words;
- generation of two summary figures and two CSV files.

The dataset identifier is:

```text
cornell-movie-review-data/rotten_tomatoes
```

The local Conda environment used for the project is:

```text
DLA2026-transformers
```

---

## Relevant files

```text
Exercise1/
├── main.py
├── data.py
├── eda.py
└── outputs/
    └── exercise_1_1/
        ├── figures/
        │   ├── class_distribution.png
        │   └── text_length_distribution.png
        └── results/
            ├── class_distribution.csv
            └── text_length_summary.csv
```

### Responsibilities

- `data.py` loads the official Hugging Face `DatasetDict`.
- `eda.py` computes the compact exploratory summaries and generates the plots.
- `main.py` exposes the `eda` command and coordinates the execution.

The refactored version no longer includes the previous duplicate, missing-value, invalid-label, or cross-split overlap checks. Those checks were useful during development, but they were removed from the final compact implementation because the objective of this exercise is dataset understanding rather than a complete data-quality audit.

---

## Dataset structure

Each example contains:

| Field | Type | Meaning |
|---|---|---|
| `text` | string | Sentence extracted from a movie review |
| `label` | `ClassLabel` | Binary sentiment label |

The verified label mapping is:

```text
0 → neg
1 → pos
```

The command also prints the first three training examples to show how individual rows are represented.

---

## Official splits

| Split | Examples |
|---|---:|
| Train | 8,530 |
| Validation | 1,066 |
| Test | 1,066 |
| **Total** | **10,662** |

The official validation split is used directly. No additional split is created.

The intended protocol is:

- **train:** fit the downstream classifier;
- **validation:** select the classifier configuration;
- **test:** evaluate only the selected configuration.

---

## Class distribution

Every split is perfectly balanced.

| Split | Negative | Positive | Negative share | Positive share |
|---|---:|---:|---:|---:|
| Train | 4,265 | 4,265 | 50.00% | 50.00% |
| Validation | 533 | 533 | 50.00% | 50.00% |
| Test | 533 | 533 | 50.00% | 50.00% |

The generated figure is saved as:

```text
Exercise1/outputs/exercise_1_1/figures/class_distribution.png
```

### Consequence for the baseline

The initial baseline does not require:

- class weighting;
- oversampling;
- undersampling;
- a balanced sampler.

A constant one-class prediction would obtain approximately 50% accuracy, which provides a simple reference level.

---

## Sentence-length analysis

The compact EDA measures each sentence in:

- characters;
- whitespace-separated words.

Token counts are not computed here because they depend on the DistilBERT tokenizer introduced in Exercise 1.2.

### Length in words

| Split | Mean | Standard deviation | Minimum | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Train | 20.99 | 9.37 | 1 | 20 | 37 | 59 |
| Validation | 21.00 | 9.64 | 1 | 21 | 38 | 54 |
| Test | 21.22 | 9.51 | 3 | 20 | 38 | 52 |

### Length in characters

| Split | Mean | Standard deviation | Minimum | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Train | 113.97 | 51.05 | 4 | 111 | 204 | 267 |
| Validation | 114.30 | 52.62 | 6 | 112 | 211 | 263 |
| Test | 115.52 | 50.96 | 14 | 113 | 206 | 261 |

The generated distribution figure is saved as:

```text
Exercise1/outputs/exercise_1_1/figures/text_length_distribution.png
```

### Interpretation

The dataset contains short review sentences rather than full reviews. The three official splits have very similar length distributions, which makes them comparable and keeps batching computationally manageable.

Word counts cannot be interpreted as Transformer sequence lengths: punctuation, special tokens, and WordPiece subwords can increase the number of tokens.

---

## Running the experiment

Activate the environment and move to the `DLA_LAB2` root directory:

```powershell
conda activate DLA2026-transformers
cd C:\Users\Alessio\pythonProject\DLA_LAB2
```

Optional syntax check:

```powershell
python -m compileall -q Exercise1
```

Run Exercise 1.1:

```powershell
python Exercise1/main.py eda
```

The command prints the dataset overview and saves the CSV summaries and figures under:

```text
Exercise1/outputs/exercise_1_1/
```

---

## Main conclusions

Exercise 1.1 establishes that:

1. the dataset loads correctly as a Hugging Face `DatasetDict`;
2. official train, validation, and test splits are already available;
3. each row contains a text string and a binary sentiment label;
4. the label mapping is `0 = neg` and `1 = pos`;
5. all three splits are perfectly balanced;
6. the sentence-length distributions are similar across splits;
7. the inputs are short enough to make Transformer batching practical.

No resplitting or class-balancing strategy is required for the initial baseline.

---

## Status

```text
Exercise 1.1: implemented, executed, and verified
```

---

## References and external resources

- Hugging Face Datasets documentation.
- Cornell Rotten Tomatoes dataset repository on the Hugging Face Hub.

## AI assistance disclosure

AI assistance was used to review the code organization and update this documentation. Dataset values and experimental results reported here originate from the project executions and generated artifacts.
