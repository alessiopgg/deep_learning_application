# Exercise 1.2 — Tokenizer and Pretrained DistilBERT Inspection

## Deep Learning Applications — Laboratory 2

This exercise introduces the input and output structures of a pretrained Transformer using the **Cornell Rotten Tomatoes** dataset and the checkpoint:

```text
distilbert/distilbert-base-uncased
```

The objective is to inspect the complete forward path:

```text
raw text
→ tokenizer
→ input_ids and attention_mask
→ DistilBERT base encoder
→ last_hidden_state
→ first-token sentence representation
```

No classifier is trained in this exercise. DistilBERT is used only in evaluation mode and the forward pass is executed with gradients disabled.

The current refactored implementation keeps only the information required by the exercise. The previous validation helper functions and highly detailed diagnostic prints were removed to make the code shorter and easier to follow.

---

## Relevant files

```text
Exercise1/
├── main.py
├── data.py
└── transformer_inspection.py
```

### Responsibilities

- `data.py` loads the official Rotten Tomatoes splits.
- `transformer_inspection.py` loads the tokenizer and base encoder and performs single-example and batch inspections.
- `main.py` selects real dataset examples and exposes the two CLI commands.

This exercise does not create output files. Its results are printed directly to the console.

---

## Pretrained components

The tokenizer and model are loaded with:

```python
AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
)

AutoModel.from_pretrained(
    "distilbert/distilbert-base-uncased"
)
```

`AutoTokenizer` selects the tokenizer associated with the checkpoint.

`AutoModel` loads the base `DistilBertModel` encoder without a sentiment-classification head. Therefore, the model produces contextual representations rather than positive/negative predictions.

The model is placed in evaluation mode:

```python
model.eval()
```

The forward pass is executed inside:

```python
with torch.inference_mode():
    outputs = model(**encoding)
```

No `backward()` call, optimizer, loss, or parameter update is used.

---

# Single-example inspection

## Command

```powershell
python Exercise1/main.py inspect-transformer
```

The command always selects:

```text
split: train
index: 0
```

The verified example has label `1`, corresponding to positive sentiment.

## Tokenization

The tokenizer:

- converts the text to WordPiece tokens;
- adds the special tokens `[CLS]` and `[SEP]`;
- returns PyTorch tensors;
- creates `input_ids` and `attention_mask`.

The refactored command prints the final token sequence passed to the model, including special tokens.

Some words are divided into subwords, for example:

```text
schwarzenegger → schwarz + ##ene + ##gger
claud          → cl + ##aud
damme          → dam + ##me
segal          → sega + ##l
```

The `##` prefix means that the fragment continues the previous subword.

This confirms that:

```text
number of words ≠ number of Transformer tokens
```

## `input_ids`

For the verified example:

```text
input_ids shape: (1, 47)
```

General form:

```text
[batch_size, sequence_length]
```

Interpretation:

- `1`: one sentence in the batch;
- `47`: token positions including `[CLS]` and `[SEP]`.

`input_ids` contain integer vocabulary indices. They are not yet contextual semantic representations.

## `attention_mask`

The mask has the same shape as `input_ids`:

```text
attention_mask shape: (1, 47)
```

Interpretation:

```text
1 → real token
0 → padding position
```

A single sentence does not require padding, so the mask contains only ones.

---

## DistilBERT output

The model output contains:

```python
outputs.last_hidden_state
```

For the verified sentence:

```text
last_hidden_state shape: (1, 47, 768)
```

General form:

```text
[batch_size, sequence_length, hidden_size]
```

DistilBERT therefore produces one contextual vector of `768` values for every sequence position.

The representation of a token depends on the complete sentence. This differs from `input_ids`, which are fixed vocabulary identifiers.

---

## First-token sentence representation

The implementation extracts:

```python
cls_features = outputs.last_hidden_state[:, 0, :]
```

The position `0` corresponds to the first special token, `[CLS]`.

For one sentence:

```text
CLS feature shape: (1, 768)
```

Shape transformation:

```text
last_hidden_state: [1, L, 768]
select position 0
→ CLS features: [1, 768]
```

This vector is used as the fixed-size sentence representation in Exercise 1.3.

---

# Batch inspection and dynamic padding

## Command

```powershell
python Exercise1/main.py inspect-transformer-batch
```

`main.py` examines the first 50 training examples and selects:

- the shortest sentence according to whitespace-separated word count;
- the longest sentence according to the same criterion.

This produces a small batch with deliberately different sequence lengths.

In the verified execution, the selected examples were:

```text
short example index: 26
long example index: 38
```

The short text was:

```text
spiderman rocks
```

After WordPiece tokenization and special tokens, it contains five real token positions.

The long example contains 54 real token positions.

---

## Dynamic padding

The tokenizer is called with:

```python
padding=True
```

This pads every sequence only to the longest sequence in the current batch.

For the verified pair:

```text
short sequence: 5 real tokens
long sequence: 54 real tokens
batch length: 54
```

The short sequence therefore receives 49 `[PAD]` positions.

The resulting shapes are:

```text
input_ids:          (2, 54)
attention_mask:     (2, 54)
last_hidden_state:  (2, 54, 768)
CLS feature matrix: (2, 768)
```

The `attention_mask` distinguishes real text from padding:

```text
short sentence: 1 1 1 1 1 0 0 ... 0
long sentence:  1 1 1 1 1 1 1 ... 1
```

DistilBERT returns a rectangular output tensor, including vectors at padded positions. These positions are not treated as real textual content.

---

## Shape summary

### Single sentence

```text
input_ids:          [1, L]
attention_mask:     [1, L]
last_hidden_state:  [1, L, 768]
CLS features:       [1, 768]
```

### Batch of `B` sentences

```text
input_ids:          [B, Lmax]
attention_mask:     [B, Lmax]
last_hidden_state:  [B, Lmax, 768]
CLS features:       [B, 768]
```

`Lmax` is the maximum tokenized length inside the current batch.

---

## What the exercise verifies

The current implementation verifies:

- loading of `AutoTokenizer`;
- loading of the base `AutoModel` encoder;
- use of real dataset examples;
- WordPiece tokenization;
- addition of special tokens;
- construction of `input_ids` and `attention_mask`;
- conversion to PyTorch tensors;
- inference without gradients;
- inspection of `last_hidden_state`;
- extraction of the first-token representation;
- batching with dynamic padding;
- interpretation of all relevant tensor shapes;
- absence of a sentiment-classification head.

---

## What is deliberately not done here

Exercise 1.2 does not perform:

- complete feature extraction for all splits;
- classifier training;
- validation-based hyperparameter selection;
- test evaluation;
- fine-tuning of DistilBERT;
- loss computation;
- optimizer updates.

These operations belong to Exercise 1.3 or later exercises.

---

## Running the experiment

From the `DLA_LAB2` root directory:

```powershell
conda activate DLA2026-transformers
cd C:\Users\Alessio\pythonProject\DLA_LAB2
```

Optional syntax check:

```powershell
python -m compileall -q Exercise1
```

Run the single-example inspection:

```powershell
python Exercise1/main.py inspect-transformer
```

Run the batch and padding inspection:

```powershell
python Exercise1/main.py inspect-transformer-batch
```

---

## Warning notes

### Hugging Face authentication warning

The public checkpoint can be downloaded without an `HF_TOKEN`. Anonymous requests can have lower rate limits, but the warning does not change the model output.

### Windows symlink warning

The Hugging Face cache may report limited symlink support on Windows. This can increase disk usage but does not change the experiment.

### Unexpected pretrained-head parameters

A loading report may mention parameters associated with the masked-language-modeling head. `AutoModel` loads only the base DistilBERT encoder, so those head parameters are not required for this exercise.

---

## Connection with Exercise 1.3

The central result is the fixed-size representation:

```python
last_hidden_state[:, 0, :]
```

For the full dataset, the expected feature matrices are:

```text
train:      [8530, 768]
validation: [1066, 768]
test:       [1066, 768]
```

Exercise 1.3 computes these matrices and uses them with a classical linear classifier.

---

## Status

```text
Exercise 1.2: implemented, executed, and verified
Exercise 1.3: implemented, executed, and verified
```

---

## References and external resources

- Hugging Face Transformers documentation.
- DistilBERT model documentation and checkpoint page.
- Cornell Rotten Tomatoes dataset repository on the Hugging Face Hub.

## AI assistance disclosure

AI assistance was used to review the code organization and update this documentation. Tensor shapes and examples reported here originate from the project executions.
