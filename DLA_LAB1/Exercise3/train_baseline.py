"""CLI entry point for Step 11 Faster R-CNN baseline training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from Exercise3.data_pipeline.loaders import build_detection_dataloaders
from Exercise3.data_pipeline.loading import (
    DATASET_CONFIGURATION,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DEFAULT_CACHE_DIR,
    load_detection_dataset,
)
from Exercise3.models.faster_rcnn import (
    FasterRCNNBaselineConfig,
    build_faster_rcnn_baseline,
    summarize_faster_rcnn,
)
from Exercise3.training.checkpointing import load_checkpoint
from Exercise3.training.configuration import (
    load_training_config,
    parse_config_arguments,
    resolve_cache_dir,
    resolve_output_root,
    save_resolved_config,
    validate_resume_compatibility,
)
from Exercise3.training.engine import (
    build_grad_scaler,
    resolve_device,
    set_reproducibility,
)
from Exercise3.training.tracking import ExperimentTracker
from Exercise3.training.trainer import (
    build_optimizer,
    build_runtime_metadata,
    build_scheduler,
    create_run_directory,
    fit_detector,
)


def main() -> None:
    config_path, overrides = parse_config_arguments()
    config = load_training_config(config_path, overrides)
    device = resolve_device(config.experiment.device)
    amp_enabled = config.training.amp and device.type == "cuda"
    config.training.amp = amp_enabled
    set_reproducibility(config.experiment.seed, config.experiment.deterministic)

    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        torch.cuda.set_device(index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(index)

    cache_dir = resolve_cache_dir(config)
    dataset_dict, resolved_cache_dir = load_detection_dataset(
        DEFAULT_CACHE_DIR if cache_dir is None else cache_dir
    )
    loaders = build_detection_dataloaders(
        dataset_dict,
        train_batch_size=config.data.train_batch_size,
        evaluation_batch_size=config.data.evaluation_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
        persistent_workers=config.data.persistent_workers,
        seed=config.experiment.seed,
    )

    model_config = FasterRCNNBaselineConfig(
        architecture=config.model.architecture,
        weights=config.model.weights,
        num_classes=config.model.num_classes,
        backbone_source=config.model.backbone_source,
        gtsrb_checkpoint=config.model.gtsrb_checkpoint,
        required_gtsrb_strategy=config.model.required_gtsrb_strategy,
        trainable_backbone=config.model.trainable_backbone,
        freeze_backbone=config.model.freeze_backbone,
        seed=config.experiment.seed,
        progress=config.model.progress,
    )
    model, model_metadata = build_faster_rcnn_baseline(model_config)
    model.to(device)
    model_audit = summarize_faster_rcnn(model, model_metadata)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = build_grad_scaler(
        device,
        enabled=amp_enabled,
        initial_scale=config.training.amp_initial_scale,
    )

    resume_path = (
        None
        if config.experiment.resume_from is None
        else Path(config.experiment.resume_from).expanduser().resolve()
    )
    resume_checkpoint = (
        None if resume_path is None else load_checkpoint(resume_path, device)
    )
    if resume_checkpoint is not None:
        validate_resume_compatibility(config, resume_checkpoint["config"])
    output_root = resolve_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name, run_dir = create_run_directory(
        output_root,
        config.experiment.run_name,
        resume_path,
    )
    save_resolved_config(config, run_dir)

    runtime_metadata = build_runtime_metadata(
        device=device,
        model_metadata=model_audit,
        loader_settings=loaders.settings.to_dict(),
        dataset_metadata={
            "repository": DATASET_REPOSITORY,
            "configuration": DATASET_CONFIGURATION,
            "revision": DATASET_REVISION,
            "resolved_cache_dir": str(resolved_cache_dir),
            "train_images": len(loaders.datasets.train),
            "validation_images": len(loaders.datasets.validation),
            "test_images": len(loaders.datasets.test),
            "test_used_during_training": False,
        },
    )
    (run_dir / "runtime_metadata.json").write_text(
        json.dumps(runtime_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    resume_wandb_id = (
        None
        if resume_checkpoint is None
        else resume_checkpoint.get("wandb_run_id")
    )
    tracker = ExperimentTracker(
        enabled=config.tracking.use_wandb,
        project=config.tracking.project,
        entity=config.tracking.entity,
        group=config.tracking.group,
        mode=config.tracking.mode,
        run_name=run_name,
        config=config.to_dict(),
        resume_run_id=resume_wandb_id,
        log_best_checkpoint=config.tracking.log_best_checkpoint,
    )

    print("\n=== Exercise 3.3 - Step 11: baseline training ===")
    print(f"Run: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Epoch target: {config.training.epochs}")
    print(f"Train batch size: {config.data.train_batch_size}")
    print(f"Validation batch size: {config.data.evaluation_batch_size}")
    print(f"Train batches limit: {config.training.max_train_batches}")
    print(f"Validation batches limit: {config.training.max_validation_batches}")
    print(f"W&B enabled: {config.tracking.use_wandb}")
    print(f"Resume checkpoint: {resume_path}")
    print("Test evaluation during training: False")

    try:
        summary = fit_detector(
            model=model,
            train_loader=loaders.train,
            validation_loader=loaders.validation,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
            run_name=run_name,
            run_dir=run_dir,
            tracker=tracker,
            resume_checkpoint=resume_checkpoint,
        )
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    else:
        tracker.finish(exit_code=0)

    print("\nTraining completed.")
    print(f"Best epoch: {summary['best_epoch']}")
    print(
        "Best validation total loss: "
        f"{summary['best_validation_total_loss']:.6f}"
    )
    print(f"Best checkpoint: {summary['best_checkpoint']}")
    print(f"Scientific full run: {summary['scientific_run']}")
    print("Test evaluated: False")


if __name__ == "__main__":
    main()
