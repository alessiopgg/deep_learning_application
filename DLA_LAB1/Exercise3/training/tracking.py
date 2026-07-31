"""Optional Weights & Biases integration without a hard runtime dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ExperimentTracker:
    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        group: str,
        mode: str,
        run_name: str,
        config: dict[str, Any],
        resume_run_id: str | None,
        log_best_checkpoint: bool,
    ) -> None:
        self.enabled = enabled
        self.run: Any | None = None
        self.log_best_checkpoint_enabled = log_best_checkpoint

        if not enabled:
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "tracking.use_wandb=true but the wandb package is not installed."
            ) from error

        model_config = config.get("model", {})
        backbone_source = str(model_config.get("backbone_source", "coco"))
        trainable_backbone = str(
            model_config.get("trainable_backbone", "frozen")
        )
        init_arguments: dict[str, Any] = {
            "project": project,
            "entity": entity,
            "group": group,
            "mode": mode,
            "name": run_name,
            "config": config,
            "job_type": "faster-rcnn-training",
            "tags": [
                "exercise-3-3",
                "faster-rcnn",
                f"backbone-{backbone_source}",
                f"trainable-{trainable_backbone}",
            ],
        }
        if resume_run_id is not None:
            init_arguments["id"] = resume_run_id
            init_arguments["resume"] = "allow"
        self.run = wandb.init(**init_arguments)
        if self.run is None:
            raise RuntimeError("wandb.init() did not return a Run object.")

        self.run.define_metric("global_step")
        self.run.define_metric("train_batch_*", step_metric="global_step")
        self.run.define_metric("epoch")
        self.run.define_metric("train_epoch_*", step_metric="epoch")
        self.run.define_metric("validation_*", step_metric="epoch")

    @property
    def run_id(self) -> str | None:
        return None if self.run is None else str(self.run.id)

    def log_batch(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(values)

    def log_epoch(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(values)

    def update_summary(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key, value in values.items():
            self.run.summary[key] = value

    def log_best_checkpoint(self, path: Path, run_name: str) -> None:
        if self.run is not None and self.log_best_checkpoint_enabled:
            self.run.log_model(path=str(path), name=f"{run_name}-best")

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
