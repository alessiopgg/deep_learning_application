"""Entry point for the compact Exercise 2 training pipeline."""

from datetime import datetime

from configuration import config_to_yaml, load_config
from data import create_dataloaders, print_data_summary, resolve_path
from models import create_input_transform, create_model, print_model_summary
from training import (
    create_training_components,
    describe_device,
    evaluate,
    fit,
    print_epoch_metrics,
    print_optimization_summary,
    resolve_device,
    set_reproducibility,
    train_one_epoch,
)


def create_run_directory(config):
    name = config.experiment.run_name
    if name is None:
        name = (
            f"{config.model.name}-"
            f"{config.model.fine_tuning_strategy}-"
            f"{config.model.classifier_type}"
        )
    else:
        name = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(name).strip()).strip("-")
        if not name:
            raise ValueError("experiment.run_name must contain a letter or digit.")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{name}"
    run_dir = resolve_path(config.paths.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def main() -> None:
    config = load_config()
    print("\n=== Exercise 2 configuration ===")
    print(config_to_yaml(config))

    set_reproducibility(config.experiment.seed, config.experiment.deterministic)
    device = resolve_device(config.experiment.device)
    print("=== Exercise 2 runtime ===")
    print(f"Selected device: {describe_device(device)}")
    print(f"Seed: {config.experiment.seed}")
    print(f"Deterministic execution: {config.experiment.deterministic}")

    transform = create_input_transform(config.model.name)
    loaders, data_info = create_dataloaders(config, transform, device)
    print_data_summary(data_info)

    model, model_info = create_model(config, device)
    print_model_summary(model_info)
    criterion, optimizer, optimization_info = create_training_components(model, config)
    print_optimization_summary(optimization_info)

    smoke_batches = int(config.experiment.smoke_test_batches)
    if smoke_batches > 0:
        print(f"\nRunning smoke test over {smoke_batches} training and validation batch(es).")
        training_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
            config.model.fine_tuning_strategy,
            config.logging.batch_interval,
            max_batches=smoke_batches,
        )
        validation_metrics, _, _ = evaluate(
            model, loaders["validation"], criterion, device,
            max_batches=smoke_batches,
        )
        print_epoch_metrics("Smoke-test training metrics", training_metrics)
        print_epoch_metrics("Smoke-test validation metrics", validation_metrics)
        print("\nEngine smoke test completed successfully.")
        return

    run_id, run_dir = create_run_directory(config)
    print("\n=== Exercise 2 run ===")
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")

    fit(
        model,
        loaders["train"],
        loaders["validation"],
        criterion,
        optimizer,
        device,
        config,
        run_dir / "best_model.pt",
    )


if __name__ == "__main__":
    main()
