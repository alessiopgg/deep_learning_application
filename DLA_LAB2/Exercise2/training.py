from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from model import load_sequence_classifier


DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42


def compute_classification_metrics(
        evaluation: EvalPrediction,
) -> dict[str, float]:
    """Compute metrics from classification logits and integer labels."""
    logits = evaluation.predictions
    labels = evaluation.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def build_training_arguments(
        output_dir: Path,
        epochs: float = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
        eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = DEFAULT_SEED,
) -> TrainingArguments:
    """Create the initial full-fine-tuning configuration."""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=seed,
        data_seed=seed,
    )


def build_trainer(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        tokenized_dataset: DatasetDict,
        training_arguments: TrainingArguments,
) -> Trainer:
    """Build the Hugging Face Trainer used for training and validation."""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_classification_metrics,
    )


def run_full_fine_tuning(
        tokenized_dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: Path,
        epochs: float = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
        eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = DEFAULT_SEED,
) -> None:
    """Fine-tune all DistilBERT parameters and evaluate on validation."""
    checkpoint_dir = output_dir / "checkpoints"
    final_model_dir = output_dir / "best_model"

    model = load_sequence_classifier()
    total_parameters, trainable_parameters = count_parameters(model)

    if total_parameters != trainable_parameters:
        raise ValueError(
            "Exercise 2 requires full fine-tuning, but some parameters "
            "are frozen."
        )

    training_arguments = build_training_arguments(
        output_dir=checkpoint_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        weight_decay=weight_decay,
        seed=seed,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        training_arguments=training_arguments,
    )

    print("\n=== Exercise 2.3: full DistilBERT fine-tuning ===")
    print(f"Training examples: {len(tokenized_dataset['train'])}")
    print(f"Validation examples: {len(tokenized_dataset['validation'])}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train batch size per device: {train_batch_size}")
    print(f"Evaluation batch size per device: {eval_batch_size}")
    print(f"Weight decay: {weight_decay}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Mixed precision fp16: {training_arguments.fp16}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print("Test split evaluation: False")

    train_result = trainer.train()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    validation_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["validation"],
        metric_key_prefix="validation",
    )
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\nBest model saved in: {final_model_dir}")
    print(f"Training artifacts saved in: {checkpoint_dir}")
    print("The test split was not evaluated.")


def run_test_evaluation(
        tokenized_dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: Path,
        eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        seed: int = DEFAULT_SEED,
) -> None:
    """Evaluate the already selected best model once on the test split."""
    final_model_dir = output_dir / "best_model"
    test_output_dir = output_dir / "test_evaluation"

    if not final_model_dir.exists():
        raise FileNotFoundError(
            f"Best model not found in {final_model_dir}. "
            "Run the train command first."
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        final_model_dir
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    evaluation_arguments = TrainingArguments(
        output_dir=str(test_output_dir),
        per_device_eval_batch_size=eval_batch_size,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=evaluation_arguments,
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_classification_metrics,
    )

    print("\n=== Exercise 2.3: final test evaluation ===")
    print(f"Model directory: {final_model_dir}")
    print(f"Test examples: {len(tokenized_dataset['test'])}")
    print(f"Evaluation batch size per device: {eval_batch_size}")

    test_metrics = trainer.evaluate(metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    print(f"\nTest metrics saved in: {test_output_dir}")