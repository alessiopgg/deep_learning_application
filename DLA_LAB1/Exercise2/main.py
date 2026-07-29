"""Entry point for the Exercise 2 experimental pipeline."""

from configuration import config_to_yaml, load_config
from data import create_dataloaders, print_data_summary
from engine import evaluate, print_epoch_metrics, train_one_epoch
from experiment_paths import create_run_paths
from models import (
    create_input_transform,
    create_model,
    print_model_summary,
)
from optimization import (
    create_training_components,
    print_optimization_summary,
)
from runtime import describe_device, resolve_device, set_reproducibility
from training import fit


def main() -> None:
    """Prepare the pipeline and optionally execute a short smoke test."""
    config = load_config()

    print("\n=== Exercise 2 configuration ===")
    print(config_to_yaml(config))

    set_reproducibility(
        seed=config.experiment.seed,
        deterministic=config.experiment.deterministic,
    )
    device = resolve_device(config.experiment.device)

    print("=== Exercise 2 runtime ===")
    print(f"Selected device: {describe_device(device)}")
    print(f"Seed: {config.experiment.seed}")
    print(f"Deterministic execution: {config.experiment.deterministic}")

    transform = create_input_transform(config.model.name)
    data_loaders = create_dataloaders(
        config=config,
        transform=transform,
        device=device,
    )
    print_data_summary(data_loaders)

    model_bundle = create_model(
        config=config,
        device=device,
    )
    print_model_summary(model_bundle)

    components = create_training_components(
        model=model_bundle.model,
        config=config,
    )
    print_optimization_summary(components)

    smoke_test_batches = config.experiment.smoke_test_batches
    if smoke_test_batches > 0:
        print(
            "\nRunning an engine smoke test over "
            f"{smoke_test_batches} training and validation batch(es)."
        )

        training_metrics = train_one_epoch(
            model=model_bundle.model,
            dataloader=data_loaders.train,
            criterion=components.criterion,
            optimizer=components.optimizer,
            device=device,
            fine_tuning_strategy=(
                config.model.fine_tuning_strategy
            ),
            log_interval=config.logging.batch_interval,
            max_batches=smoke_test_batches,
        )
        validation_result = evaluate(
            model=model_bundle.model,
            dataloader=data_loaders.validation,
            criterion=components.criterion,
            device=device,
            max_batches=smoke_test_batches,
        )

        print_epoch_metrics(
            "Smoke-test training metrics",
            training_metrics,
        )
        print_epoch_metrics(
            "Smoke-test validation metrics",
            validation_result.metrics,
        )
        print("\nEngine smoke test completed successfully.")
        return

    run_paths = create_run_paths(config)
    print("\n=== Exercise 2 run ===")
    print(f"Run ID: {run_paths.run_id}")
    print(f"Run directory: {run_paths.run_dir}")

    fit(
        model=model_bundle.model,
        train_loader=data_loaders.train,
        validation_loader=data_loaders.validation,
        criterion=components.criterion,
        optimizer=components.optimizer,
        device=device,
        config=config,
        checkpoint_path=run_paths.checkpoint_path,
    )


if __name__ == "__main__":
    main()
