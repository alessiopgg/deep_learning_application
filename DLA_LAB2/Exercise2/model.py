import torch
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from data import MODEL_CHECKPOINT


NUM_LABELS = 2
ID_TO_LABEL = {
    0: "negative",
    1: "positive",
}
LABEL_TO_ID = {
    label: class_id
    for class_id, label in ID_TO_LABEL.items()
}


def load_sequence_classifier(
        model_checkpoint: str = MODEL_CHECKPOINT,
) -> PreTrainedModel:
    """Load DistilBERT with a new binary sequence-classification head."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )


def inspect_sequence_classifier(
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
) -> None:
    """Run an inference batch and inspect the classification output."""
    if not texts:
        raise ValueError("At least one text is required for model inspection.")

    encoded_batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )

    model.eval()

    with torch.inference_mode():
        outputs = model(**encoded_batch)

    logits = outputs.logits
    predicted_classes = logits.argmax(dim=-1)

    expected_shape = (len(texts), NUM_LABELS)

    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Expected logits shape {expected_shape}, "
            f"found {tuple(logits.shape)}."
        )

    print("\n=== Exercise 2.2: sequence-classification model inspection ===")
    print(f"Checkpoint: {MODEL_CHECKPOINT}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Number of labels: {model.config.num_labels}")
    print(f"Label mapping: {model.config.id2label}")
    print(f"Batch size: {len(texts)}")
    print(f"input_ids shape: {tuple(encoded_batch['input_ids'].shape)}")
    print(
        "attention_mask shape: "
        f"{tuple(encoded_batch['attention_mask'].shape)}"
    )
    print(f"logits shape: {tuple(logits.shape)}")

    for index, text in enumerate(texts):
        predicted_id = int(predicted_classes[index])
        predicted_label = model.config.id2label[predicted_id]

        print(f"\n--- Batch position {index} ---")
        print(f"Text: {text!r}")
        print(f"Logits: {logits[index].tolist()}")
        print(f"Temporary predicted class: {predicted_id} ({predicted_label})")

    print(
        "\nThe classification head is newly initialized, so these "
        "predictions are not meaningful before fine-tuning."
    )