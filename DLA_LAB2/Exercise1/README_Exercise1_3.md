# Exercise 1.3 — Stable Baseline with DistilBERT and LinearSVC

## Deep Learning Applications — Laboratory 2

Exercise 1.3 builds a stable binary sentiment-classification baseline for the **Cornell Rotten Tomatoes** dataset.

DistilBERT is not fine-tuned. It remains frozen and is used as a feature extractor. Each sentence is transformed into the representation of the first token from the final Transformer layer:

```python
last_hidden_state[:, 0, :]
```

This produces one vector of `768` values per sentence. The vectors are then standardized and classified with `LinearSVC`.

The complete pipeline is:

```text
text
→ AutoTokenizer
→ input_ids + attention_mask
→ frozen DistilBERT
→ last_hidden_state[:, 0, :]
→ 768-dimensional sentence vector
→ StandardScaler
→ LinearSVC
→ negative or positive sentiment
```

The checkpoint is:

```text
distilbert/distilbert-base-uncased
```

The local Conda environment is:

```text
DLA2026-transformers
```

---

## Compact refactoring

The final implementation was simplified to retain only the components required for a correct and reproducible experiment.

The following development-only commands and code paths were removed:

- token-length preflight;
- feature-extraction smoke test;
- separate preliminary baseline with `C = 1`;
- repeated internal validations and detailed diagnostics;
- duplicated saving helpers and atomic temporary-file logic.

The final public commands are:

```text
extract-features
select-baseline
evaluate-test
```

The scientific protocol is unchanged:

1. extract frozen DistilBERT features for all official splits;
2. select `LinearSVC.C` using only the validation split;
3. evaluate the already selected pipeline once on the test split.

---

## Dataset and protocol

Dataset identifier:

```text
cornell-movie-review-data/rotten_tomatoes
```

| Split | Examples | Negative | Positive |
|---|---:|---:|---:|
| Train | 8,530 | 4,265 | 4,265 |
| Validation | 1,066 | 533 | 533 |
| Test | 1,066 | 533 | 533 |
| **Total** | **10,662** | **5,331** | **5,331** |

Label mapping:

```text
0 → neg
1 → pos
```

The official splits are used as follows:

- **train:** fit `StandardScaler` and `LinearSVC`;
- **validation:** select the value of `C`;
- **test:** evaluate the selected saved pipeline without refitting.

The test split is not loaded by the model-selection command.

---

## Relevant files

```text
Exercise1/
├── main.py
├── data.py
├── transformer_inspection.py
├── feature_extraction.py
├── baseline_classifier.py
└── outputs/
    └── exercise_1_3/
        ├── features/
        │   ├── train_features.npz
        │   ├── validation_features.npz
        │   └── test_features.npz
        ├── models/
        │   └── selected_linear_svc_pipeline.joblib
        ├── predictions/
        │   ├── selected_validation_predictions.npz
        │   └── test_predictions.npz
        └── results/
            ├── feature_extraction_metadata.json
            ├── validation_model_selection.csv
            ├── selected_baseline.json
            ├── selected_validation_classification_report.json
            ├── test_metrics.json
            └── test_classification_report.json
```

### Module responsibilities

#### `feature_extraction.py`

- resolves CPU or CUDA execution;
- loads the tokenizer and base DistilBERT encoder;
- freezes all Transformer parameters;
- processes the three splits in batches;
- uses dynamic padding and truncation at the architectural limit;
- extracts `last_hidden_state[:, 0, :]`;
- saves features, labels, indices, and extraction metadata.

#### `baseline_classifier.py`

- loads the saved feature archives;
- builds a `StandardScaler + LinearSVC` pipeline;
- evaluates each candidate `C` on validation;
- selects the best pipeline by validation macro-F1;
- saves the selected model and validation predictions;
- evaluates the saved pipeline on test without retraining.

#### `main.py`

Exposes the three CLI commands and passes their arguments to the corresponding experiment functions.

---

# 1. Feature extraction

## Model preparation

The model is loaded and frozen with:

```python
model = AutoModel.from_pretrained(model_checkpoint)
model.requires_grad_(False)
model.eval()
model.to(device)
```

Each forward pass runs inside:

```python
with torch.inference_mode():
    hidden_states = model(**encoding).last_hidden_state
```

Therefore:

- no gradient graph is created;
- no model parameter is updated;
- DistilBERT remains identical to the pretrained checkpoint.

---

## Tokenization policy

For each batch:

```python
encoding = tokenizer(
    batch_texts,
    padding=True,
    truncation=True,
    max_length=model.config.max_position_embeddings,
    return_tensors="pt",
)
```

The policy is:

```text
padding: dynamic inside each batch
truncation: enabled at 512 positions
maximum length: model.config.max_position_embeddings
```

The previous preflight found no sentence above the 512-token limit. Therefore enabling truncation makes the final implementation safer without changing the verified feature values for this dataset.

---

## Extracted representation

The model output has shape:

```text
[batch_size, sequence_length, 768]
```

The implementation selects:

```python
hidden_states[:, 0, :]
```

which produces:

```text
[batch_size, 768]
```

After concatenating all batches, the expected matrices are:

| Split | Feature shape | Label shape |
|---|---:|---:|
| Train | `(8530, 768)` | `(8530,)` |
| Validation | `(1066, 768)` | `(1066,)` |
| Test | `(1066, 768)` | `(1066,)` |

The verified extraction times from the completed run were:

| Split | Time |
|---|---:|
| Train | 12.60 s |
| Validation | 1.59 s |
| Test | 1.64 s |

Execution time depends on the hardware and should not be treated as a model-quality metric.

---

## Saved feature archives

Each `.npz` archive contains:

```text
features → [number_of_examples, 768], float32
labels   → [number_of_examples], int64
indices  → [number_of_examples], int64
```

The archives allow the classifier grid to be evaluated without repeating the expensive Transformer forward pass.

Metadata are saved in:

```text
Exercise1/outputs/exercise_1_3/results/feature_extraction_metadata.json
```

The metadata record:

- checkpoint;
- device;
- batch size;
- feature source;
- hidden size;
- maximum sequence length;
- split sizes, shapes, elapsed times, and archive paths.

---

## Feature-extraction command

From the `DLA_LAB2` root directory:

```powershell
python Exercise1/main.py extract-features --device auto --batch-size 32
```

To request CUDA explicitly:

```powershell
python Exercise1/main.py extract-features --device cuda --batch-size 32
```

Use `--overwrite` only when existing feature archives should intentionally be replaced:

```powershell
python Exercise1/main.py extract-features --device auto --batch-size 32 --overwrite
```

---

# 2. Stable classifier

The downstream model is a Scikit-learn pipeline:

```text
StandardScaler
→ LinearSVC
```
Come verifica esplorativa, le stesse feature DistilBERT sono state utilizzate anche con Linear Discriminant Analysis e regressione logistica. LDA ha ottenuto un’accuracy di validation pari a 0,813321, mentre la regressione logistica, con C=1 e solver lbfgs, ha raggiunto accuracy 0,814259 e macro-F1 0,814180. Entrambe le alternative sono risultate leggermente inferiori alla pipeline StandardScaler + LinearSVC con C=0.01, che ha ottenuto accuracy 0,820826 e macro-F1 0,820742. Per mantenere la soluzione finale semplice e coerente con la consegna, i classificatori esplorativi non sono stati inclusi nel codice definitivo.
## Standardization without leakage

The pipeline is fitted only with training features:

```python
pipeline.fit(train_features, train_labels)
```

`StandardScaler` therefore estimates means and standard deviations from the train split only. Validation and test are transformed using the already fitted scaler.

Keeping the scaler and classifier inside a single `Pipeline` prevents accidental preprocessing leakage.

---

## LinearSVC configuration

```text
classifier: LinearSVC
random_state: 42
dual: False
max_iter: 10000
class_weight: None
```

`dual=False` is appropriate here because the number of training examples is larger than the number of feature dimensions:

```text
8530 examples > 768 features
```

No class weighting is required because the dataset is perfectly balanced.

---

# 3. Validation model selection

The candidate grid is:

```text
C ∈ {0.01, 0.1, 1.0, 10.0}
```

For each value, the implementation:

1. builds a new `StandardScaler + LinearSVC` pipeline;
2. fits it on train features;
3. predicts the validation split;
4. computes accuracy and macro-F1;
5. records fit time, prediction time, and optimizer iterations.

The selected configuration maximizes:

1. validation macro-F1;
2. validation accuracy as the first tie-breaker;
3. the smaller value of `C` as the final tie-breaker.

## Command

```powershell
python Exercise1/main.py select-baseline --c-values 0.01 0.1 1 10 --max-iter 10000
```

## Verified validation results

| C | Accuracy | Macro-F1 | Iterations | Fit time |
|---:|---:|---:|---:|---:|
| **0.01** | **0.820826** | **0.820742** | 9 | 3.339 s |
| 0.1 | 0.818949 | 0.818865 | 10 | 4.559 s |
| 1.0 | 0.818011 | 0.817934 | 10 | 4.447 s |
| 10.0 | 0.818011 | 0.817934 | 10 | 4.411 s |

The selected pipeline is:

```text
StandardScaler + LinearSVC(C=0.01)
```

Selected validation confusion matrix:

```text
[[449, 84],
 [107, 426]]
```

Interpretation:

```text
449 true negatives
84 false positives
107 false negatives
426 true positives
```

The model-selection command does not load the test split.

---

## Model-selection artifacts

```text
Exercise1/outputs/exercise_1_3/results/validation_model_selection.csv
Exercise1/outputs/exercise_1_3/results/selected_baseline.json
Exercise1/outputs/exercise_1_3/results/selected_validation_classification_report.json
Exercise1/outputs/exercise_1_3/models/selected_linear_svc_pipeline.joblib
Exercise1/outputs/exercise_1_3/predictions/selected_validation_predictions.npz
```

The prediction archive contains:

```text
indices
labels
predictions
decision_scores
```

---

# 4. Final test evaluation

After model selection, the saved pipeline is evaluated on the official test split.

The pipeline is loaded from disk and used directly:

```text
pipeline refitted before test: False
test used for model selection: False
```

No new value of `C` can be passed to the test command.

## Command

```powershell
python Exercise1/main.py evaluate-test
```

## Verified final results

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 0.820826 | 0.820742 |
| Test | **0.800188** | **0.800148** |

Test prediction time:

```text
0.052 s
```

Test confusion matrix:

```text
[[434, 99],
 [114, 419]]
```

Interpretation:

```text
434 true negatives
99 false positives
114 false negatives
419 true positives
```

The validation-test gap is approximately 2.06 percentage points for both accuracy and macro-F1. Test performance is therefore lower but remains close to the validation estimate.

Accuracy and macro-F1 are almost identical, which is consistent with the balanced class distribution and relatively similar performance across the two classes.

---

## Test artifacts

```text
Exercise1/outputs/exercise_1_3/results/test_metrics.json
Exercise1/outputs/exercise_1_3/results/test_classification_report.json
Exercise1/outputs/exercise_1_3/predictions/test_predictions.npz
```

To intentionally reproduce the same evaluation and replace existing output files:

```powershell
python Exercise1/main.py evaluate-test --overwrite
```

The test should normally remain closed until the validation configuration has been fixed.

---

# 5. Complete reproduction sequence

Activate the environment and move to the project root:

```powershell
conda activate DLA2026-transformers
cd C:\Users\Alessio\pythonProject\DLA_LAB2
```

Optional syntax check:

```powershell
python -m compileall -q Exercise1
```

Extract all feature matrices:

```powershell
python Exercise1/main.py extract-features --device auto --batch-size 32
```

Select the classifier on validation:

```powershell
python Exercise1/main.py select-baseline --c-values 0.01 0.1 1 10 --max-iter 10000
```

Evaluate the selected pipeline on test:

```powershell
python Exercise1/main.py evaluate-test
```

The required order is:

```text
feature extraction
→ validation model selection
→ final test evaluation
```

---

# 6. Essential safeguards retained

Although the implementation was simplified, it still retains the checks that protect the experiment from common practical errors:

- required train, validation, and test splits must exist;
- extraction batch size must be positive;
- requesting CUDA without CUDA availability raises an error;
- existing experimental artifacts are not overwritten by default;
- the number of feature rows must match the number of labels;
- train and validation feature dimensions must match;
- test feature dimension must match the selected model configuration;
- model selection uses validation only;
- final test evaluation uses the already saved pipeline without refitting.

These checks preserve the scientific protocol without obscuring the core implementation.

---

# 7. Scientific interpretation

The final baseline reaches:

```text
Test accuracy: 0.800188
Test macro-F1: 0.800148
```

This demonstrates that pretrained DistilBERT representations contain substantial sentiment information even when the Transformer is never updated on Rotten Tomatoes.

Only the following components are learned from the dataset:

- the means and standard deviations of `StandardScaler`;
- the coefficients and intercept of `LinearSVC`.

DistilBERT remains frozen. The experiment therefore measures the usefulness of pretrained sentence representations for a downstream linear classifier.

The improvement obtained by changing `C` from `1.0` to `0.01` is small. The correct conclusion is not that `0.01` is universally optimal, but that it performed best within the evaluated grid and validation split.

---

# 8. Limitations

The baseline deliberately remains simple:

1. it uses only the first token from the final layer;
2. it does not compare mean pooling or max pooling;
3. it evaluates only `LinearSVC`;
4. it uses a small grid of `C` values;
5. it does not fine-tune DistilBERT;
6. it uses a single official validation split;
7. it does not estimate variability across multiple seeds;
8. it does not include a qualitative error analysis;
9. the uncased checkpoint discards capitalization information.

These limits are consistent with the purpose of the exercise: establish a transparent and reproducible stable baseline before moving to Transformer fine-tuning.

---

## Final result summary

```text
Dataset:              Cornell Rotten Tomatoes
Feature extractor:    distilbert/distilbert-base-uncased
Feature:              last_hidden_state[:, 0, :]
Feature dimension:    768
Transformer training: none
Classifier:           StandardScaler + LinearSVC
Selected C:           0.01
Validation accuracy:  0.820826
Validation macro-F1:  0.820742
Test accuracy:        0.800188
Test macro-F1:        0.800148
```

---

## Status

```text
Feature extraction:          completed and verified
Validation model selection:  completed and verified
Final test evaluation:       completed and verified
Exercise 1.3:                complete
```

---

## References and external resources

- Hugging Face Transformers documentation.
- Hugging Face Datasets documentation.
- DistilBERT model documentation and checkpoint page.
- Scikit-learn documentation for `StandardScaler`, `Pipeline`, and `LinearSVC`.

## AI assistance disclosure

AI assistance was used to review the refactored code organization and update this documentation. Experimental metrics and tensor shapes reported here originate from the completed project executions and generated artifacts.
