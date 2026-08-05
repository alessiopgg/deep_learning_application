from __future__ import annotations

import argparse
from typing import Any

from Exercise3.data import (
    DEFAULT_DATASET_ID,
    DEFAULT_EDA_LIMIT,
    DEFAULT_NUM_EXAMPLES,
    inspect_dataset,
)
from Exercise3.evaluation import (
    DEFAULT_EVALUATION_BATCH_SIZE,
    DEFAULT_EVALUATION_OUTPUT_DIR,
    DEFAULT_EVALUATION_SPLIT,
    evaluate_text_to_image,
)
from Exercise3.indexing import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_INDEX_BATCH_SIZE,
    DEFAULT_INDEX_OUTPUT_DIR,
    DEFAULT_INDEX_SPLITS,
    build_image_index,
)
from Exercise3.model import (
    DEFAULT_CLIP_MODEL_ID,
    DEFAULT_CLIP_NUM_SAMPLES,
    DEFAULT_CLIP_SPLIT,
    inspect_clip,
)
from Exercise3.retrieval import DEFAULT_TOP_K, search_image_index


def _add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")


def _add_index_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_OUTPUT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise 3.3 - text-to-image retrieval."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dataset_parser = commands.add_parser(
        "inspect-dataset",
        help="Inspect Flickr8k and run a compact EDA.",
    )
    dataset_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    dataset_parser.add_argument("--streaming", action="store_true")
    dataset_parser.add_argument(
        "--eda-limit",
        type=int,
        default=DEFAULT_EDA_LIMIT,
    )
    dataset_parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
    )
    dataset_parser.add_argument(
        "--output-dir",
        default="Exercise3/outputs/dataset_inspection",
    )

    clip_parser = commands.add_parser(
        "inspect-clip",
        help="Inspect CLIP preprocessing and similarities.",
    )
    clip_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    clip_parser.add_argument("--split", default=DEFAULT_CLIP_SPLIT)
    clip_parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_CLIP_NUM_SAMPLES,
    )
    clip_parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL_ID)
    _add_device(clip_parser)

    index_parser = commands.add_parser(
        "build-index",
        help="Build and persist the normalized CLIP image index.",
    )
    index_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    index_parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
    )
    index_parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_INDEX_SPLITS),
    )
    index_parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL_ID)
    index_parser.add_argument("--limit", type=int)
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INDEX_BATCH_SIZE,
    )
    index_parser.add_argument(
        "--output-dir",
        default=DEFAULT_INDEX_OUTPUT_DIR,
    )
    index_parser.add_argument("--force", action="store_true")
    _add_device(index_parser)

    search_parser = commands.add_parser(
        "search",
        help="Search an existing image index with a text query.",
    )
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    search_parser.add_argument("--model-id")
    _add_index_dir(search_parser)
    _add_device(search_parser)

    evaluation_parser = commands.add_parser(
        "evaluate-retrieval",
        help="Evaluate caption-to-image retrieval.",
    )
    evaluation_parser.add_argument(
        "--split",
        default=DEFAULT_EVALUATION_SPLIT,
    )
    evaluation_parser.add_argument("--limit", type=int)
    evaluation_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EVALUATION_BATCH_SIZE,
    )
    evaluation_parser.add_argument(
        "--output-dir",
        default=DEFAULT_EVALUATION_OUTPUT_DIR,
    )
    evaluation_parser.add_argument("--force", action="store_true")
    _add_index_dir(evaluation_parser)
    _add_device(evaluation_parser)

    app_parser = commands.add_parser(
        "launch-app",
        help="Launch the Gradio retrieval application.",
    )
    app_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    app_parser.add_argument("--server-name", default="127.0.0.1")
    app_parser.add_argument("--server-port", type=int)
    app_parser.add_argument("--share", action="store_true")
    app_parser.add_argument("--inbrowser", action="store_true")
    _add_index_dir(app_parser)
    _add_device(app_parser)

    return parser


def _run_command(args: argparse.Namespace) -> Any:
    if args.command == "inspect-dataset":
        return inspect_dataset(
            dataset_id=args.dataset_id,
            streaming=args.streaming,
            eda_limit=args.eda_limit,
            num_examples=args.num_examples,
            output_dir=args.output_dir,
        )
    if args.command == "inspect-clip":
        return inspect_clip(
            dataset_id=args.dataset_id,
            split=args.split,
            num_samples=args.num_samples,
            model_id=args.model_id,
            requested_device=args.device,
        )
    if args.command == "build-index":
        return build_image_index(
            dataset_id=args.dataset_id,
            dataset_config=args.dataset_config,
            splits=args.splits,
            model_id=args.model_id,
            limit=args.limit,
            batch_size=args.batch_size,
            requested_device=args.device,
            output_dir=args.output_dir,
            force=args.force,
        )
    if args.command == "search":
        return search_image_index(
            query=args.query,
            top_k=args.top_k,
            index_dir=args.index_dir,
            requested_model_id=args.model_id,
            requested_device=args.device,
        )
    if args.command == "evaluate-retrieval":
        return evaluate_text_to_image(
            index_dir=args.index_dir,
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            requested_device=args.device,
            output_dir=args.output_dir,
            force=args.force,
        )
    if args.command == "launch-app":
        from Exercise3.app import launch_app

        return launch_app(
            index_dir=args.index_dir,
            requested_device=args.device,
            top_k=args.top_k,
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            inbrowser=args.inbrowser,
        )
    raise ValueError(f"Unsupported command: {args.command}")


def main() -> None:
    _run_command(build_parser().parse_args())


if __name__ == "__main__":
    main()
