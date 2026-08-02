import torch
from transformers import AutoModel, AutoTokenizer


MODEL_CHECKPOINT = "distilbert/distilbert-base-uncased"


def load_pretrained_components(
    model_checkpoint: str = MODEL_CHECKPOINT,
):
    """Load the DistilBERT tokenizer and base encoder."""
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint)
    model.eval()
    return tokenizer, model


def run_transformer_inspection(
    text: str,
    label: int,
    split_name: str,
    example_index: int,
    model_checkpoint: str = MODEL_CHECKPOINT,
) -> None:
    """Inspect tokenization and DistilBERT output for one example."""
    tokenizer, model = load_pretrained_components(model_checkpoint)

    encoding = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0].tolist()
    )

    with torch.inference_mode():
        outputs = model(**encoding)

    hidden_states = outputs.last_hidden_state
    cls_features = hidden_states[:, 0, :]

    print("\n=== Exercise 1.2: single-example inspection ===")
    print(f"Checkpoint: {model_checkpoint}")
    print(f"Split/index: {split_name}/{example_index}")
    print(f"Label: {label}")
    print(f"Text: {text!r}")
    print(f"Tokens: {tokens}")
    print(f"input_ids shape: {tuple(encoding['input_ids'].shape)}")
    print(f"input_ids: {encoding['input_ids'].tolist()}")
    print(f"attention_mask: {encoding['attention_mask'].tolist()}")
    print(f"last_hidden_state shape: {tuple(hidden_states.shape)}")
    print(f"CLS feature shape: {tuple(cls_features.shape)}")
    print(f"First 10 CLS values: {cls_features[0, :10]}")
    print(
        "AutoModel returns contextual token representations, "
        "not sentiment predictions."
    )


def run_transformer_batch_inspection(
    texts: list[str],
    labels: list[int],
    split_name: str,
    example_indices: list[int],
    model_checkpoint: str = MODEL_CHECKPOINT,
) -> None:
    """Inspect dynamic padding and outputs for a small text batch."""
    tokenizer, model = load_pretrained_components(model_checkpoint)

    encoding = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    with torch.inference_mode():
        outputs = model(**encoding)

    hidden_states = outputs.last_hidden_state
    cls_features = hidden_states[:, 0, :]

    print("\n=== Exercise 1.2: batch inspection ===")
    print(f"Checkpoint: {model_checkpoint}")
    print(f"input_ids shape: {tuple(encoding['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(encoding['attention_mask'].shape)}")
    print(f"last_hidden_state shape: {tuple(hidden_states.shape)}")
    print(f"CLS feature matrix shape: {tuple(cls_features.shape)}")

    for batch_index, (text, label, dataset_index) in enumerate(
        zip(texts, labels, example_indices)
    ):
        tokens = tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][batch_index].tolist()
        )
        real_tokens = int(encoding["attention_mask"][batch_index].sum())
        padding_tokens = encoding["attention_mask"].shape[1] - real_tokens

        print(f"\n--- Batch position {batch_index} ---")
        print(f"Split/index: {split_name}/{dataset_index}")
        print(f"Label: {label}")
        print(f"Text: {text!r}")
        print(f"Tokens: {tokens}")
        print(f"Real tokens: {real_tokens}")
        print(f"Padding tokens: {padding_tokens}")
        print(f"First 10 CLS values: {cls_features[batch_index, :10]}")
