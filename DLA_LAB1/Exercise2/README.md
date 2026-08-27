# Exercise 2 — Compact and Reproducible Training Pipeline

Exercise 2 consolidates the GTSRB fine-tuning workflow from Exercise 1.3 into a configurable and reproducible pipeline. The scientific protocol is unchanged: GTSRB classification with pretrained ResNet-18/50, linear or MLP heads, the same fine-tuning strategies, Cross Entropy, AdamW, stratified train/validation splitting, and a separate final test evaluation.

## Structure

```text
Exercise2/
├── configuration.py     # OmegaConf loading, CLI overrides, validation
├── data.py              # GTSRB split and DataLoaders
├── models.py            # ResNet construction and freezing policies
├── training.py          # runtime, optimizer, loops, checkpointing
├── main.py              # training / smoke-test entry point
├── evaluate_test.py     # independent official test evaluation
├── configs/
│   └── default.yaml
├── assets/
└── outputs/
```

The previous small modules `runtime.py`, `optimization.py`, `engine.py`, `checkpointing.py`, and `experiment_paths.py` were merged into the modules that actually use their logic. This keeps the separation of responsibilities without fragmenting a relatively small laboratory pipeline.

## Configuration

Default values are stored only in `configs/default.yaml`. OmegaConf merges command-line overrides on top and validates supported values and relevant relationships.

```bash
python Exercise2/main.py \
  model.name=resnet50 \
  data.batch_size=16 \
  model.classifier_type=mlp \
  model.fine_tuning_strategy=classifier \
  training.epochs=10
```

Supported values:

| Component | Values |
|---|---|
| Backbone | `resnet18`, `resnet50` |
| Head | `linear`, `mlp` |
| Fine-tuning | `classifier`, `last_block`, `full` |
| Loss | Cross Entropy |
| Optimizer | AdamW |
| Checkpoint metric | validation loss, accuracy, macro-F1 |
| Device | `auto`, `cpu`, `cuda`, `cuda:<index>` |

Checkpoint mode must remain coherent with the monitored metric:

```text
validation_loss      -> min
validation_accuracy  -> max
validation_macro_f1  -> max
```

## Data and preprocessing

The official GTSRB training split is divided with the same stratified 80/20 protocol and seed `42` used previously:

| Split | Images |
|---|---:|
| Internal training | 21,312 |
| Validation | 5,328 |
| Official test | 12,630 |
| Classes | 43 |

Preprocessing comes directly from the selected Torchvision pretrained weights through `weights.transforms()`. No custom augmentation is introduced.

## Fine-tuning

Strategies are unchanged:

| Strategy | Trainable modules |
|---|---|
| `classifier` | classifier head |
| `last_block` | `layer4` + classifier |
| `full` | complete network |

Frozen BatchNorm stages remain in evaluation mode during selective fine-tuning so their running statistics are not modified.

The default optimizer remains AdamW with differentiated learning rates:

- backbone: `1e-4`;
- classifier: `1e-3`;
- weight decay: `1e-4`.

Training and validation report loss, accuracy and macro-F1. The best checkpoint is selected only from validation metrics.

## Smoke test

```bash
python Exercise2/main.py experiment.smoke_test_batches=2
```

This runs the real data/model/optimizer pipeline on a limited number of train and validation batches and checks forward, backward, optimizer step and evaluation without starting a full experiment.

## Full training

Default run:

```bash
python Exercise2/main.py
```

ResNet-50 example:

```bash
python Exercise2/main.py \
  model.name=resnet50 \
  data.batch_size=16 \
  model.fine_tuning_strategy=last_block \
  model.classifier_type=linear \
  training.epochs=5
```

Macro-F1 checkpoint selection:

```bash
python Exercise2/main.py \
  checkpoint.monitor=validation_macro_f1 \
  checkpoint.mode=max
```

Deterministic execution:

```bash
python Exercise2/main.py experiment.deterministic=true
```

## Checkpointing

The best checkpoint still stores:

- selected epoch and validation metric;
- model state dictionary;
- optimizer state dictionary;
- complete resolved configuration.

State is moved to CPU before serialization and writing remains atomic through a temporary `best_model.pt.tmp` file.

## Separate test evaluation

```bash
python Exercise2/evaluate_test.py \
  --checkpoint Exercise2/outputs/runs/<run_id>/best_model.pt
```

Optional explicit device:

```bash
python Exercise2/evaluate_test.py \
  --checkpoint Exercise2/outputs/runs/<run_id>/best_model.pt \
  --device cuda:0
```

The test script reconstructs preprocessing, loaders and model from the checkpoint configuration, then produces:

```text
test_metrics.json
classification_report.json
predictions.npz
```

Keeping test evaluation separate prevents the official test split from being used for model selection.

## Reference run already obtained

The previously completed reference run remains the experimental reference; this refactor does not redefine or replace its results.

| Component | Value |
|---|---|
| Run name | `resnet18-lastblock-mlp` |
| Backbone | ResNet-18 |
| Head | MLP |
| Strategy | `last_block` |
| Epochs | 10 |
| Batch size | 32 |
| Checkpoint monitor | `validation_macro_f1` |
| Checkpoint mode | `max` |
| Seed | 42 |

Selected checkpoint: epoch **10**.

| Metric | Value |
|---|---:|
| Best validation macro-F1 | 0.9972 |
| Test loss | 0.1416 |
| Test accuracy | 0.9645 |
| Test macro-F1 | 0.9590 |
| Test samples | 12,630 |
| Test evaluation time | 9.87 s |

These values come from the existing experiment artifacts and are not recomputed by the refactor.

## Reproducibility and limitations

Python, NumPy and PyTorch RNGs are seeded; DataLoader workers and the training generator are also seeded. With `experiment.deterministic=true`, deterministic PyTorch algorithms are requested and cuDNN benchmarking is disabled.

The reference result uses one seed and non-deterministic execution. The internal validation split is easier than the official test set. Repeated test evaluation should therefore be avoided during model selection.

The unused W&B configuration previously present in the Exercise 2 schema was removed: unlike Exercise 1.3, this pipeline never initialized or logged to W&B. Removing those dead options does not alter an executed experiment.

## References and AI assistance

Main external resources: GTSRB, `torchvision.datasets.GTSRB`, Torchvision pretrained ResNet weights/transforms, PyTorch, OmegaConf and Scikit-learn metrics.

ChatGPT was used to discuss the assignment, explain pipeline concepts, review software organization, assist debugging, and revise documentation. Suggestions were reviewed and validated by the author; reported numerical results come from experiment artifacts.
