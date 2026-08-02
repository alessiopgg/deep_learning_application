import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.tokenization_utils_base import BatchEncoding


MODEL_CHECKPOINT = "distilbert/distilbert-base-uncased"


def load_pretrained_components(
        model_checkpoint: str = MODEL_CHECKPOINT,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load the tokenizer and the base pre-trained Transformer model.

    AutoModel loads the DistilBERT encoder without adding a
    classification head.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint
    )

    model = AutoModel.from_pretrained(
        model_checkpoint
    )

    model.eval()

    return tokenizer, model


def tokenize_single_text(
        text: str,
        tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[str], list[str], BatchEncoding]:
    """
    Tokenize one text without padding or truncation.
    """
    tokenizer_tokens = tokenizer.tokenize(text)

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )

    validate_encoding(
        encoding=encoding,
        expected_batch_size=1,
    )

    model_input_tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0].tolist()
    )

    return (
        tokenizer_tokens,
        model_input_tokens,
        encoding,
    )


def tokenize_text_batch(
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[list[str]], list[list[str]], BatchEncoding]:
    """
    Tokenize a batch of texts using dynamic padding.

    All sequences are padded to the length of the longest
    tokenized text in this specific batch.
    """
    if len(texts) < 2:
        raise ValueError(
            "Batch inspection requires at least two texts."
        )

    for text_index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"Text at batch position {text_index} must be "
                f"a string, but received {type(text).__name__}."
            )

        if not text.strip():
            raise ValueError(
                f"Text at batch position {text_index} is empty."
            )

    tokenizer_tokens = [
        tokenizer.tokenize(text)
        for text in texts
    ]

    encoding = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )

    validate_encoding(
        encoding=encoding,
        expected_batch_size=len(texts),
    )

    model_input_tokens = [
        tokenizer.convert_ids_to_tokens(
            input_ids_row.tolist()
        )
        for input_ids_row in encoding["input_ids"]
    ]

    return (
        tokenizer_tokens,
        model_input_tokens,
        encoding,
    )


def validate_encoding(
        encoding: BatchEncoding,
        expected_batch_size: int,
) -> None:
    """
    Verify the fundamental tokenizer-output structure.
    """
    required_fields = (
        "input_ids",
        "attention_mask",
    )

    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in encoding
    ]

    if missing_fields:
        raise ValueError(
            "The tokenizer output is missing the required fields: "
            f"{missing_fields}"
        )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    if input_ids.ndim != 2:
        raise ValueError(
            "Expected input_ids to have two dimensions "
            "[batch_size, sequence_length], "
            f"but received shape {tuple(input_ids.shape)}."
        )

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "Expected attention_mask and input_ids to have "
            "the same shape, but received "
            f"{tuple(attention_mask.shape)} and "
            f"{tuple(input_ids.shape)}."
        )

    if input_ids.shape[0] != expected_batch_size:
        raise ValueError(
            "Unexpected tokenizer batch size: "
            f"{input_ids.shape[0]} instead of "
            f"{expected_batch_size}."
        )


def execute_forward_pass(
        model: PreTrainedModel,
        encoding: BatchEncoding,
) -> BaseModelOutput:
    """
    Pass tokenized inputs through DistilBERT without gradients.
    """
    model_device = next(model.parameters()).device

    model_inputs = {
        "input_ids": encoding["input_ids"].to(model_device),
        "attention_mask": encoding[
            "attention_mask"
        ].to(model_device),
    }

    with torch.inference_mode():
        outputs = model(**model_inputs)

    if not hasattr(outputs, "last_hidden_state"):
        raise ValueError(
            "The model output does not contain "
            "last_hidden_state."
        )

    return outputs


def validate_model_output(
        input_ids: torch.Tensor,
        last_hidden_state: torch.Tensor,
) -> None:
    """
    Check that output batch and sequence dimensions match the input.
    """
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "Expected last_hidden_state to have three dimensions "
            "[batch_size, sequence_length, hidden_size], "
            f"but received shape "
            f"{tuple(last_hidden_state.shape)}."
        )

    input_batch_size = input_ids.shape[0]
    input_sequence_length = input_ids.shape[1]

    output_batch_size = last_hidden_state.shape[0]
    output_sequence_length = last_hidden_state.shape[1]

    if output_batch_size != input_batch_size:
        raise ValueError(
            "The output batch size does not match the input "
            f"batch size: {output_batch_size} != "
            f"{input_batch_size}."
        )

    if output_sequence_length != input_sequence_length:
        raise ValueError(
            "The output sequence length does not match the "
            f"input sequence length: "
            f"{output_sequence_length} != "
            f"{input_sequence_length}."
        )


def run_transformer_inspection(
        text: str,
        label: int,
        split_name: str,
        example_index: int,
        model_checkpoint: str = MODEL_CHECKPOINT,
) -> None:
    """
    Inspect DistilBERT using one real dataset example.
    """
    tokenizer, model = load_pretrained_components(
        model_checkpoint=model_checkpoint
    )

    (
        tokenizer_tokens,
        model_input_tokens,
        encoding,
    ) = tokenize_single_text(
        text=text,
        tokenizer=tokenizer,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    model_device = next(model.parameters()).device

    print("\n=== Exercise 1.2: pre-trained components ===")
    print(f"Checkpoint: {model_checkpoint}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Model type: {type(model).__name__}")
    print(f"Model device: {model_device}")
    print(f"Model training mode: {model.training}")

    print("\n=== Selected dataset example ===")
    print(f"Split: {split_name}")
    print(f"Example index: {example_index}")
    print(f"Text type: {type(text).__name__}")
    print(f"Label type: {type(label).__name__}")
    print(f"Label: {label}")
    print(f"Text: {text!r}")

    print("\n=== Tokenizer output ===")
    print(f"Encoding type: {type(encoding).__name__}")

    print(
        "Tokens before adding special tokens "
        f"({len(tokenizer_tokens)}):"
    )
    print(tokenizer_tokens)

    print(
        "\nTokens passed to the model, including special tokens "
        f"({len(model_input_tokens)}):"
    )
    print(model_input_tokens)

    print("\n=== input_ids ===")
    print(f"Object type: {type(input_ids).__name__}")
    print(f"Shape: {tuple(input_ids.shape)}")
    print(f"Dtype: {input_ids.dtype}")
    print(f"Device: {input_ids.device}")
    print(f"Values: {input_ids.tolist()}")

    print("\n=== attention_mask ===")
    print(f"Object type: {type(attention_mask).__name__}")
    print(f"Shape: {tuple(attention_mask.shape)}")
    print(f"Dtype: {attention_mask.dtype}")
    print(f"Device: {attention_mask.device}")
    print(f"Values: {attention_mask.tolist()}")

    print("\n=== Input shape interpretation ===")
    print(
        "input_ids: "
        "[batch_size, sequence_length] = "
        f"{tuple(input_ids.shape)}"
    )
    print(
        "attention_mask: "
        "[batch_size, sequence_length] = "
        f"{tuple(attention_mask.shape)}"
    )
    print(f"Batch size: {input_ids.shape[0]}")
    print(
        "Sequence length: "
        f"{input_ids.shape[1]} tokens, including special tokens"
    )

    outputs = execute_forward_pass(
        model=model,
        encoding=encoding,
    )

    last_hidden_state = outputs.last_hidden_state

    validate_model_output(
        input_ids=input_ids,
        last_hidden_state=last_hidden_state,
    )

    first_token_representation = (
        last_hidden_state[:, 0, :]
    )

    print("\n=== Model output ===")
    print(f"Output type: {type(outputs).__name__}")
    print(f"Output fields: {list(outputs.keys())}")

    print("\n=== last_hidden_state ===")
    print(
        f"Object type: "
        f"{type(last_hidden_state).__name__}"
    )
    print(f"Shape: {tuple(last_hidden_state.shape)}")
    print(f"Dtype: {last_hidden_state.dtype}")
    print(f"Device: {last_hidden_state.device}")
    print(
        "Requires gradient: "
        f"{last_hidden_state.requires_grad}"
    )

    print(
        "\nSmall output sample "
        "[first two tokens, first eight components]:"
    )
    print(
        last_hidden_state[
        0,
        :2,
        :8,
        ]
    )

    print("\n=== First-token representation ===")
    print(
        "Selected position: "
        "last_hidden_state[:, 0, :]"
    )
    print(
        f"Token at position 0: {model_input_tokens[0]}"
    )
    print(
        f"Shape: "
        f"{tuple(first_token_representation.shape)}"
    )
    print(
        f"Dtype: "
        f"{first_token_representation.dtype}"
    )
    print(
        f"Device: "
        f"{first_token_representation.device}"
    )
    print(
        "First 10 components of the first-token vector:"
    )
    print(
        first_token_representation[
        0,
        :10,
        ]
    )

    print("\n=== Output shape interpretation ===")
    print(
        "last_hidden_state: "
        "[batch_size, sequence_length, hidden_size] = "
        f"{tuple(last_hidden_state.shape)}"
    )
    print(
        "First-token representation: "
        "[batch_size, hidden_size] = "
        f"{tuple(first_token_representation.shape)}"
    )
    print(
        "Each sequence position now has one contextual "
        "representation produced by the final Transformer layer."
    )
    print(
        "No classification prediction was produced because "
        "AutoModel loads the base encoder without a "
        "classification head."
    )


def run_transformer_batch_inspection(
        texts: list[str],
        labels: list[int],
        split_name: str,
        example_indices: list[int],
        model_checkpoint: str = MODEL_CHECKPOINT,
) -> None:
    """
    Inspect dynamic padding and model outputs for a text batch.
    """
    if not (
            len(texts)
            == len(labels)
            == len(example_indices)
    ):
        raise ValueError(
            "Texts, labels and example indices must have "
            "the same length."
        )

    tokenizer, model = load_pretrained_components(
        model_checkpoint=model_checkpoint
    )

    (
        tokenizer_tokens,
        model_input_tokens,
        encoding,
    ) = tokenize_text_batch(
        texts=texts,
        tokenizer=tokenizer,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    print("\n=== Exercise 1.2: batch inspection ===")
    print(f"Checkpoint: {model_checkpoint}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Model type: {type(model).__name__}")
    print(
        f"Model device: "
        f"{next(model.parameters()).device}"
    )
    print(f"Batch size: {len(texts)}")
    print(f"Padding strategy: dynamic batch padding")
    print(f"Padding token: {tokenizer.pad_token!r}")
    print(f"Padding token ID: {tokenizer.pad_token_id}")

    print("\n=== Selected dataset examples ===")

    for batch_index, (
            text,
            label,
            example_index,
            original_tokens,
            padded_tokens,
    ) in enumerate(
        zip(
            texts,
            labels,
            example_indices,
            tokenizer_tokens,
            model_input_tokens,
        )
    ):
        mask_row = attention_mask[batch_index]

        real_token_count = int(
            mask_row.sum().item()
        )

        padding_token_count = int(
            mask_row.numel() - real_token_count
        )

        print(
            f"\n--- Batch position {batch_index} ---"
        )
        print(f"Split: {split_name}")
        print(f"Dataset index: {example_index}")
        print(f"Label: {label}")
        print(f"Text: {text!r}")

        print(
            "Tokens before special tokens: "
            f"{len(original_tokens)}"
        )
        print(original_tokens)

        print(
            "Tokens after special tokens and padding: "
            f"{len(padded_tokens)}"
        )
        print(padded_tokens)

        print(
            f"Real token positions: {real_token_count}"
        )
        print(
            f"Padding positions: {padding_token_count}"
        )

    print("\n=== Batched input_ids ===")
    print(f"Shape: {tuple(input_ids.shape)}")
    print(f"Dtype: {input_ids.dtype}")
    print(f"Device: {input_ids.device}")
    print(f"Values:")
    print(input_ids)

    print("\n=== Batched attention_mask ===")
    print(f"Shape: {tuple(attention_mask.shape)}")
    print(f"Dtype: {attention_mask.dtype}")
    print(f"Device: {attention_mask.device}")
    print(f"Values:")
    print(attention_mask)

    print("\n=== Padding interpretation ===")

    for batch_index in range(len(texts)):
        real_token_count = int(
            attention_mask[batch_index].sum().item()
        )

        padding_token_count = (
                attention_mask.shape[1]
                - real_token_count
        )

        print(
            f"Batch position {batch_index}: "
            f"{real_token_count} real tokens, "
            f"{padding_token_count} padding tokens"
        )

    outputs = execute_forward_pass(
        model=model,
        encoding=encoding,
    )

    last_hidden_state = outputs.last_hidden_state

    validate_model_output(
        input_ids=input_ids,
        last_hidden_state=last_hidden_state,
    )

    first_token_representations = (
        last_hidden_state[:, 0, :]
    )

    print("\n=== Batched model output ===")
    print(f"Output type: {type(outputs).__name__}")
    print(f"Output fields: {list(outputs.keys())}")

    print("\n=== Batched last_hidden_state ===")
    print(
        "Shape: "
        f"{tuple(last_hidden_state.shape)}"
    )
    print(f"Dtype: {last_hidden_state.dtype}")
    print(f"Device: {last_hidden_state.device}")
    print(
        "Requires gradient: "
        f"{last_hidden_state.requires_grad}"
    )

    print(
        "\nFirst-token representations shape: "
        f"{tuple(first_token_representations.shape)}"
    )

    for batch_index in range(len(texts)):
        print(
            f"Batch position {batch_index}, "
            "first 10 components:"
        )
        print(
            first_token_representations[
            batch_index,
            :10,
            ]
        )

    print("\n=== Batch shape interpretation ===")
    print(
        "input_ids: "
        "[batch_size, padded_sequence_length] = "
        f"{tuple(input_ids.shape)}"
    )
    print(
        "attention_mask: "
        "[batch_size, padded_sequence_length] = "
        f"{tuple(attention_mask.shape)}"
    )
    print(
        "last_hidden_state: "
        "[batch_size, padded_sequence_length, hidden_size] = "
        f"{tuple(last_hidden_state.shape)}"
    )
    print(
        "First-token representations: "
        "[batch_size, hidden_size] = "
        f"{tuple(first_token_representations.shape)}"
    )
    print(
        "Padding positions also have output vectors, but they "
        "must not be treated as real textual representations."
    )